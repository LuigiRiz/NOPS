import os
import sys
from itertools import chain as chain_iterators

import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

from models.multiheadminkunet import MultiHeadMinkUnet
from utils.collation import (
    collation_fn_restricted_dataset,
    collation_fn_restricted_dataset_two_samples,
)
from utils.dataset import dataset_wrapper, get_dataset
from utils.scheduler import LinearWarmupCosineAnnealingLR
from utils.sinkhorn_knopp import SinkhornKnopp


class Discoverer(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):

        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )

        self.model = MultiHeadMinkUnet(
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads
        )

        self.label_mapping = label_mapping
        self.label_mapping_inv = label_mapping_inv
        self.unknown_label = unknown_label

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print(f'Missing: {missing_keys}', f'Unexpected: {unexpected_keys}')

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.initial_epsilon_sk
        )

        self.sk_queue = None
        self.sk_indices = []

        self.loss_per_head = torch.zeros(self.hparams.num_heads, device=self.device)

        # wCE as loss
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        weights = torch.ones(len(self.label_mapping)) / len(self.label_mapping)
        self.criterion.weight = weights

        self.valid_criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(len(self.label_mapping)) / len(self.label_mapping)
        self.valid_criterion.weight = weights

        # Mapping numeric_label -> word_label
        dataset_config_file = self.hparams.dataset_config
        with open(dataset_config_file, "r") as f:
            dataset_config = yaml.safe_load(f)
        map_inv = dataset_config["learning_map_inv"]
        lab_dict = dataset_config["labels"]
        label_dict = {}
        for new_label, old_label in map_inv.items():
            label_dict[new_label] = lab_dict[old_label]
        self.label_dict = label_dict

        return

    def configure_optimizers(self):
        if self.hparams.pretrained is not None:
            encoder_params = self.model.encoder.parameters()
            rest_params = chain_iterators(
                self.model.head_lab.parameters(), self.model.head_unlab.parameters()
            )
            if hasattr(self.model, "head_unlab_over"):
                rest_params = chain_iterators(
                    rest_params, self.model.head_unlab_over.parameters()
                )
            optimizer = optim.SGD(
                [
                    {"params": rest_params, "lr": self.hparams.train_lr},
                    {"params": encoder_params},
                ],
                lr=self.hparams.finetune_lr,
                momentum=self.hparams.momentum_for_optim,
                weight_decay=self.hparams.weight_decay_for_optim,
            )
        else:
            optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams.train_lr,
                momentum=self.hparams.momentum_for_optim,
                weight_decay=self.hparams.weight_decay_for_optim,
            )

        if self.hparams.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.epochs,
                warmup_start_lr=self.hparams.min_lr,
                eta_min=self.hparams.min_lr,
            )

            return [optimizer], [scheduler]

        return optimizer

    def on_train_start(self):
        # Compute/load weights for weighted CE loss
        if not os.path.exists("weights.pt"):
            dataset = get_dataset(self.hparams.dataset)(
                config_file=self.hparams.dataset_config,
                split="train",
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
            )

            weights = torch.zeros((self.hparams.num_classes), device=self.device)

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=False,
            )

            # Split each unknown point across the 5 (or 4) unknown classes
            unk_labels_num = self.hparams.num_unlabeled_classes
            with tqdm(
                total=len(dataloader),
                desc="Evaluating weights for wCE",
                file=sys.stdout,
            ) as pbar:
                for _, _, _, _, labels, _ in dataloader:
                    for label in set(self.label_mapping.values()):
                        n_points = (labels == label).nonzero().numel()
                        if label != self.unknown_label:
                            weights[label] += n_points
                        else:
                            weights[-unk_labels_num:] += n_points / unk_labels_num
                    pbar.update()

            weights += 1
            weights = 1 / weights
            weights = weights / torch.sum(weights)
            self.criterion.weight = weights
            torch.save(weights, "weights.pt")
        else:
            print("\nLoading weights.pt ...", flush=True)
            weights = torch.load("weights.pt").to(self.device)
            self.criterion.weight = weights

    def train_dataloader(self):

        dataset = get_dataset(self.hparams.dataset)(
            config_file=self.hparams.dataset_config,
            split="train",
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
        )

        dataset = dataset_wrapper(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset_two_samples,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):

        dataset = get_dataset(self.hparams.dataset)(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
        )

        return dataloader

    def on_train_epoch_start(self):
        # Reset best_head tracker
        self.loss_per_head = torch.zeros_like(self.loss_per_head, device=self.device)

        # Compute the actual epsilon for Sinkhorn-Knopp
        if self.hparams.adapting_epsilon_sk and self.hparams.epochs > 1:
            eps_0 = self.hparams.initial_epsilon_sk
            eps_n = self.hparams.final_epsilon_sk
            n_ep = self.hparams.epochs
            act_ep = self.current_epoch
            self.sk.epsilon = eps_0 + act_ep * (eps_n - eps_0) / (n_ep - 1)

    def training_step(self, data, _):
        def get_uncertainty_mask(preds: torch.Tensor, p=0.5):
            """
            returns a boolean mask selecting the p-th percentile of the predictions with highest confidence for each class

            :param preds: Tensor of predicted logits (N x Nc)
            :param p: float describing the percentile to use in the selection
            """

            self.log(f"utils/tot_p", preds.shape[0])

            # init mask
            uncertainty_mask = torch.zeros(
                preds.shape[0], dtype=torch.bool, device=self.device
            )

            # get hard predictions
            hard_preds = preds.argmax(dim=-1)

            # generate indexes for consistent mapping
            indexes = torch.arange(preds.shape[0], device=self.device)

            # for each novel class
            for un_tmp in range(self.hparams.num_unlabeled_classes):
                # select points with given novel class
                un_idx_tmp = hard_preds == un_tmp

                if (un_idx_tmp.sum() * p).int() > 0:
                    # select confident novel pts
                    un_conf = preds[un_idx_tmp][:, un_tmp]
                    un_sel_tmp = indexes[un_idx_tmp]

                    # sort them
                    sorted_conf_tmp, sorted_idx_tmp = torch.sort(un_conf)
                    un_conf = un_conf[sorted_idx_tmp]
                    un_sel_tmp = un_sel_tmp[sorted_idx_tmp]

                    # get percentile idx
                    perc_tmp = (un_idx_tmp.sum() * p).int()

                    # update th
                    un_th_tmp = un_conf[perc_tmp]

                    # find valid pts
                    mask_tmp = un_conf > un_th_tmp

                    self.log(f"utils/thr_{un_tmp}", un_th_tmp)
                    self.log(
                        f"utils/perc_{un_tmp}", mask_tmp.sum() / un_sel_tmp.shape[0]
                    )
                    self.log(f"utils/tot_p_{un_tmp}", un_sel_tmp.shape[0])

                    uncertainty_mask[un_sel_tmp[mask_tmp]] = 1

            return uncertainty_mask

        nlc = self.hparams.num_labeled_classes

        (
            coords,
            feats,
            _,
            selected_idx,
            mapped_labels,
            coords1,
            feats1,
            _,
            selected_idx1,
            mapped_labels1,
            pcd_indexes,
        ) = data

        pcd_masks = []
        pcd_masks1 = []
        for i in range(pcd_indexes.shape[0]):
            pcd_masks.append(coords[:, 0] == i)
            pcd_masks1.append(coords1[:, 0] == i)

        # Forward
        coords = coords.int()
        coords1 = coords1.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)
        sp_tensor1 = ME.SparseTensor(features=feats1.float(), coordinates=coords1)

        # Clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        out1 = self.model(sp_tensor1)

        # Gather outputs
        out["logits_lab"] = (
            out["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)
        )
        out1["logits_lab"] = (
            out1["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)
        )
        logits = torch.cat([out["logits_lab"], out["logits_unlab"]], dim=-1)
        logits1 = torch.cat([out1["logits_lab"], out1["logits_unlab"]], dim=-1)
        if self.hparams.overcluster_factor is not None:
            logits_over = torch.cat(
                [out["logits_lab"], out["logits_unlab_over"]], dim=-1
            )
            logits_over1 = torch.cat(
                [out1["logits_lab"], out1["logits_unlab_over"]], dim=-1
            )

        mask_lab = mapped_labels != self.unknown_label
        mask_lab1 = mapped_labels1 != self.unknown_label

        # Generate one-hot targets for the base points
        targets_lab = (
            F.one_hot(
                mapped_labels[mask_lab].to(torch.long),
                num_classes=self.hparams.num_labeled_classes,
            )
            .float()
            .to(self.device)
        )
        targets_lab1 = (
            F.one_hot(
                mapped_labels1[mask_lab1].to(torch.long),
                num_classes=self.hparams.num_labeled_classes,
            )
            .float()
            .to(self.device)
        )

        # Generate empty targets for all the points
        targets = torch.zeros_like(logits)
        targets1 = torch.zeros_like(logits1)
        if self.hparams.overcluster_factor is not None:
            targets_over = torch.zeros_like(logits_over)
            targets_over1 = torch.zeros_like(logits_over1)

        # Generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        act_queue = (
            None
            if self.current_epoch < self.hparams.queue_start_epoch
            else self.sk_queue
        )
        for h in range(self.hparams.num_heads):
            # Insert the one-hot labels
            targets[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            targets1[h, mask_lab1, :nlc] = targets_lab1.type_as(targets1)

            if self.hparams.use_uncertainty_queue or self.hparams.use_uncertainty_loss:
                # Get masks for certain points
                unc_mask = get_uncertainty_mask(
                    out["logits_unlab"][h][~mask_lab].detach(),
                    p=self.hparams.uncertainty_percentile,
                )
                unc_mask1 = get_uncertainty_mask(
                    out1["logits_unlab"][h][~mask_lab1].detach(),
                    p=self.hparams.uncertainty_percentile,
                )
                if h == 0:
                    unc_mask_overall = unc_mask
                    unc_mask_overall1 = unc_mask1
                else:
                    unc_mask_overall = torch.logical_and(unc_mask_overall, unc_mask)
                    unc_mask_overall1 = torch.logical_and(unc_mask_overall1, unc_mask1)

            if self.hparams.use_uncertainty_loss:
                # Get predictions from Sinkhorn only for high-confidence points
                pred_sk = self.sk(
                    out["feats"][~mask_lab][unc_mask],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)
                pred_sk1 = self.sk(
                    out1["feats"][~mask_lab1][unc_mask1],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)

                new_mask_unlab = ~mask_lab.clone()
                new_mask_unlab[new_mask_unlab == True] = unc_mask
                new_mask_unlab1 = ~mask_lab1.clone()
                new_mask_unlab1[new_mask_unlab1 == True] = unc_mask1
                # Use sinkhorn labels only with the confident points (unconfident ones remain zero_labelled)
                targets[h, new_mask_unlab, nlc:] = pred_sk
                targets1[h, new_mask_unlab1, nlc:] = pred_sk1
            else:
                # Insert sinkhorn labels
                targets[h, ~mask_lab, nlc:] = self.sk(
                    out["feats"][~mask_lab],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)
                targets1[h, ~mask_lab1, nlc:] = self.sk(
                    out1["feats"][~mask_lab1],
                    self.model.head_unlab.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)

            if self.hparams.overcluster_factor is not None:
                # Manage also overclustering heads
                targets_over[h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[h, ~mask_lab, nlc:] = self.sk(
                    out["feats"][~mask_lab],
                    self.model.head_unlab_over.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets)
                targets_over1[h, mask_lab1, :nlc] = targets_lab1.type_as(targets1)
                targets_over1[h, ~mask_lab1, nlc:] = self.sk(
                    out1["feats"][~mask_lab1],
                    self.model.head_unlab_over.prototypes[h].prototypes.kernel.data,
                    queue=act_queue,
                ).type_as(targets1)

        # Evaluate loss
        loss_cluster = self.loss(
            logits, targets1, selected_idx, selected_idx1, pcd_masks, pcd_masks1
        )
        loss_cluster += self.loss(
            logits1, targets, selected_idx1, selected_idx, pcd_masks1, pcd_masks
        )

        if self.hparams.overcluster_factor is not None:
            loss_overcluster = self.loss(
                logits_over,
                targets_over1,
                selected_idx,
                selected_idx1,
                pcd_masks,
                pcd_masks1,
            )
            loss_overcluster += self.loss(
                logits_over1,
                targets_over,
                selected_idx1,
                selected_idx,
                pcd_masks1,
                pcd_masks,
            )
        else:
            loss_overcluster = loss_cluster

        # Keep track of the loss for each head
        self.loss_per_head += loss_cluster.clone().detach()

        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        loss = (loss_cluster + loss_overcluster) / 2

        # logging
        results = {
            "train/loss": loss.detach(),
            "train/loss_cluster": loss_cluster.detach(),
        }

        if self.hparams.overcluster_factor is not None:
            results["train/loss_overcluster"] = loss_overcluster.detach()

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        if self.hparams.queue_start_epoch != -1:
            if self.hparams.use_uncertainty_queue:
                self.update_queue(
                    torch.cat(
                        (
                            out["feats"][~mask_lab][unc_mask_overall],
                            out1["feats"][~mask_lab1][unc_mask_overall1],
                        )
                    )
                )
            else:
                self.update_queue(
                    torch.cat((out["feats"][~mask_lab], out1["feats"][~mask_lab1]))
                )

        return loss

    def update_queue(self, feats: torch.Tensor):
        """
        Updates self.queue with the features of the novel points in the current batch

        :param feats: the features for the novel points in the current batch
        """
        feats = feats.detach()
        if not self.hparams.use_uncertainty_queue:
            n_feats_to_retain = int(feats.shape[0] * self.hparams.queue_percentage)
            mask = torch.randperm(feats.shape[0])[:n_feats_to_retain]
        else:
            n_feats_to_retain = feats.shape[0]
            mask = torch.ones(n_feats_to_retain, device=feats.device, dtype=torch.bool)
        if self.sk_queue is None:
            self.sk_queue = feats[mask]
            self.sk_indices.append(n_feats_to_retain)
            return

        if len(self.sk_indices) < self.hparams.queue_batches:
            self.sk_queue = torch.vstack((feats[mask], self.sk_queue))
            self.sk_indices.insert(0, n_feats_to_retain)
        else:
            self.sk_queue = torch.vstack(
                (feats[mask], self.sk_queue[: -self.sk_indices[-1]])
            )
            self.sk_indices.insert(0, n_feats_to_retain)
            del self.sk_indices[-1]

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        idx_logits: torch.Tensor,
        idx_targets: torch.Tensor,
        pcd_mask_logits: torch.Tensor,
        pcd_mask_targets: torch.Tensor,
    ):
        """
        Evaluates the loss function of the predicted logits w.r.t. the targets

        :param logits: predicted logits for the first augmentation of the point clouds
        :param targets: targets for the second augmentation of the point clouds
        :param idx_logits: indexes of the selected points in the first augmentation of the point clouds
        :param idx_targets: indexes of the selected points in the second augmentation of the point clouds
        :param pcd_mask_logits: mask to separate the different point clouds in the batch
        :param pcd_mask_targets: mask to separate the different point clouds in the batch
        """

        if self.criterion.weight.shape[0] != targets.shape[2]:
            weight_bck = self.criterion.weight.clone()
            weight_new = torch.zeros(targets.shape[2], device=self.device)
            weight_new[: self.hparams.num_labeled_classes] = weight_bck[
                : self.hparams.num_labeled_classes
            ]
            new_weight_tmp = weight_bck[-1] / self.hparams.overcluster_factor
            weight_new[
                -self.hparams.num_unlabeled_classes * self.hparams.overcluster_factor :
            ] = new_weight_tmp
            self.criterion.weight = weight_new
        else:
            weight_bck = None

        heads_loss = None

        for head in range(self.hparams.num_heads):
            head_loss = None
            for pcd in range(len(pcd_mask_logits)):
                pcd_logits = logits[head][pcd_mask_logits[pcd]]
                pcd_targets = targets[head][pcd_mask_targets[pcd]]
                ####
                logit_shape = pcd_logits.shape[0]
                target_shape = pcd_targets.shape[0]
                ####
                mask_logits = torch.isin(
                    idx_logits[pcd_mask_logits[pcd]], idx_targets[pcd_mask_targets[pcd]]
                )
                mask_targets = torch.isin(
                    idx_targets[pcd_mask_targets[pcd]], idx_logits[pcd_mask_logits[pcd]]
                )
                pcd_logits = pcd_logits[mask_logits]
                pcd_targets = pcd_targets[mask_targets]
                ####
                perc_to_log = (
                    pcd_logits.shape[0] / logit_shape
                    + pcd_targets.shape[0] / target_shape
                ) / 2
                # print(perc_to_log)
                self.log("utils/points_in_common", perc_to_log)
                ####

                loss = self.criterion(pcd_logits, pcd_targets)
                if self.hparams.use_uncertainty_loss:
                    loss = loss[loss > 0]
                # pre-compute data for wCE
                multiplier = 1 / ((self.criterion.weight * pcd_targets).sum(1)).sum(0)
                loss *= multiplier
                loss = loss.sum()
                if head_loss is None:
                    head_loss = loss
                else:
                    head_loss = torch.hstack((head_loss, loss))

            if heads_loss is None:
                heads_loss = head_loss.mean()
            else:
                heads_loss = torch.hstack((heads_loss, head_loss.mean()))

        if weight_bck is not None:
            self.criterion.weight = weight_bck

        return heads_loss

    def on_validation_epoch_start(self):
        # Run the hungarian algorithm to map each novel class to the related semantic class
        if (
            self.hparams.hungarian_at_each_step
            or len(self.label_mapping_inv) < self.hparams.num_classes
        ):
            cost_matrix = torch.zeros(
                (
                    self.hparams.num_unlabeled_classes,
                    self.hparams.num_unlabeled_classes,
                ),
                device=self.device,
            )

            dataset = get_dataset(self.hparams.dataset)(
                config_file=self.hparams.dataset_config,
                split="valid",
                voxel_size=self.hparams.voxel_size,
                label_mapping=self.label_mapping,
            )

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
            )

            real_labels_to_be_matched = [
                label
                for label in self.label_mapping
                if self.label_mapping[label] == self.unknown_label
            ]

            with tqdm(
                total=len(dataloader), desc="Cost matrix build-up", file=sys.stdout
            ) as pbar:
                for step, data in enumerate(dataloader):
                    coords, feats, real_labels, _, mapped_labels, _ = data

                    # Forward
                    coords = coords.int().to(self.device)
                    feats = feats.to(self.device)
                    real_labels = real_labels.to(self.device)

                    sp_tensor = ME.SparseTensor(
                        features=feats.float(), coordinates=coords
                    )

                    # Must clear cache at regular interval
                    if self.global_step % self.hparams.clear_cache_int == 0:
                        torch.cuda.empty_cache()

                    out = self.model(sp_tensor)

                    best_head = torch.argmin(self.loss_per_head)

                    mask_unknown = mapped_labels == self.unknown_label

                    preds = out["logits_unlab"][best_head]
                    preds = torch.argmax(preds[mask_unknown].softmax(1), dim=1)

                    for pseudo_label in range(self.hparams.num_unlabeled_classes):
                        mask_pseudo = preds == pseudo_label
                        for j, real_label in enumerate(real_labels_to_be_matched):
                            mask_real = real_labels[mask_unknown] == real_label
                            cost_matrix[pseudo_label, j] += torch.logical_and(
                                mask_pseudo, mask_real
                            ).sum()

                    pbar.update()

            cost_matrix = cost_matrix / (
                torch.negative(cost_matrix)
                + torch.sum(cost_matrix, dim=0)
                + torch.sum(cost_matrix, dim=1).unsqueeze(1)
                + 1e-5
            )

            # Hungarian
            cost_matrix = cost_matrix.cpu()
            row_ind, col_ind = linear_sum_assignment(
                cost_matrix=cost_matrix, maximize=True
            )
            label_mapping = {
                row_ind[i] + self.unknown_label: real_labels_to_be_matched[col_ind[i]]
                for i in range(len(row_ind))
            }
            self.label_mapping_inv.update(label_mapping)

        # Reorder weights for validation loss
        weights = self.criterion.weight.clone()
        sorted_label_mapping_inv = dict(
            sorted(self.label_mapping_inv.items(), key=lambda item: item[1])
        )
        sorter = list(sorted_label_mapping_inv.keys())
        self.valid_criterion.weight = weights[sorter]

        return

    def validation_step(self, data, _):
        coords, feats, real_labels, _, _, _ = data

        # Forward
        coords = coords.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)

        # Must clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)

        best_head = torch.argmin(self.loss_per_head)

        preds = torch.cat([out["logits_lab"], out["logits_unlab"][best_head]], dim=-1)

        sorted_label_mapping_inv = dict(
            sorted(self.label_mapping_inv.items(), key=lambda item: item[1])
        )
        sorter = list(sorted_label_mapping_inv.keys())

        preds = preds[:, sorter]

        loss = self.valid_criterion(preds, real_labels.long())

        gt_labels = real_labels
        avail_labels = torch.unique(gt_labels).long()
        _, pred_labels = torch.max(torch.softmax(preds.detach(), dim=1), dim=1)
        IoU = jaccard_index(gt_labels, pred_labels, reduction="none")
        IoU = IoU[avail_labels]

        self.log("valid/loss", loss, on_epoch=True, sync_dist=True, rank_zero_only=True)
        IoU_to_log = {
            f"valid/IoU/{self.label_dict[int(avail_labels[i])]}": label_IoU
            for i, label_IoU in enumerate(IoU)
        }
        for label, value in IoU_to_log.items():
            self.log(label, value, on_epoch=True, sync_dist=True, rank_zero_only=True)

        return loss

    def on_save_checkpoint(self, checkpoint):
        # Maintain info about best head when saving checkpoints
        checkpoint["loss_per_head"] = self.loss_per_head
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.loss_per_head = checkpoint.get(
            "loss_per_head",
            torch.zeros(
                checkpoint["hyper_parameters"]["num_heads"], device=self.device
            ),
        )
        return super().on_load_checkpoint(checkpoint)
