import os
import sys

import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

from models.multiheadminkunet import MultiHeadMinkUnet
from utils.collation import collation_fn_restricted_dataset
from utils.dataset import get_dataset


class Pretrainer(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):

        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )

        self.model = MultiHeadMinkUnet(
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            num_heads=None,
        )

        self.label_mapping = label_mapping
        self.label_mapping_inv = label_mapping_inv
        self.unknown_label = unknown_label

        # wCE as loss
        self.criterion = torch.nn.CrossEntropyLoss()
        weights = (
            torch.ones(self.hparams.num_labeled_classes)
            / self.hparams.num_labeled_classes
        )
        self.criterion.weight = weights

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
        optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=self.hparams.train_lr,
            momentum=self.hparams.momentum_for_optim,
            weight_decay=self.hparams.weight_decay_for_optim,
        )

        return optimizer

    def on_train_start(self):
        # Compute/load weights for weighted CE loss
        if not os.path.exists("pret_weights.pt"):
            dataset = get_dataset(self.hparams.dataset)(
                config_file=self.hparams.dataset_config,
                split="train",
                voxel_size=self.hparams.voxel_size,
                label_mapping=self.label_mapping,
            )

            weights = torch.zeros(
                (self.hparams.num_labeled_classes), device=self.device
            )

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=False,
            )

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
                    pbar.update()

            weights += 1
            weights = 1 / weights
            weights = weights / torch.sum(weights)
            self.criterion.weight = weights
            torch.save(weights, "pret_weights.pt")
        else:
            print("Loading pret_weights.pt ...", flush=True)
            weights = torch.load("pret_weights.pt").to(self.device)
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

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
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

    def training_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, _ = data

        # Forward
        coords = coords.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)

        # Must clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)

        mask_lab = mapped_labels != self.unknown_label

        preds = out["logits_lab"]
        preds = preds[mask_lab]

        loss = self.criterion(preds, mapped_labels[mask_lab].long())

        gt_labels = real_labels[mask_lab]
        avail_labels = torch.unique(gt_labels).long()
        pred_labels = torch.argmax(torch.softmax(preds.detach(), dim=1), dim=1)
        # Transform predictions
        for key, value in self.label_mapping_inv.items():
            pred_labels[pred_labels == key] = -value
        pred_labels = -pred_labels

        IoU = jaccard_index(gt_labels, pred_labels, reduction="none")
        IoU = IoU[avail_labels]

        # logging
        results = {
            "train/loss": loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)
        IoU_to_log = {
            f"train/IoU/{self.label_dict[int(avail_labels[i])]}": label_IoU
            for i, label_IoU in enumerate(IoU)
        }
        self.log_dict(IoU_to_log, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, _ = data

        # Forward
        coords = coords.int()

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords)

        # Must clear cache at regular interval
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)

        mask_lab = mapped_labels != self.unknown_label

        preds = out["logits_lab"]
        preds = preds[mask_lab]

        loss = self.criterion(preds, mapped_labels[mask_lab].long())

        gt_labels = real_labels[mask_lab]
        avail_labels = torch.unique(gt_labels).long()
        pred_labels = torch.argmax(torch.softmax(preds.detach(), dim=1), dim=1)
        # Transform predictions
        for key, value in self.label_mapping_inv.items():
            pred_labels[pred_labels == key] = -value
        pred_labels = -pred_labels

        IoU = jaccard_index(gt_labels, pred_labels, reduction="none")
        IoU = IoU[avail_labels]

        # logging
        results = {
            "valid/loss": loss.detach(),
        }
        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)
        IoU_to_log = {
            f"valid/IoU/{self.label_dict[int(avail_labels[i])]}": label_IoU
            for i, label_IoU in enumerate(IoU)
        }
        self.log_dict(IoU_to_log, on_step=False, on_epoch=True, sync_dist=True)

        return
