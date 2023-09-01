import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from modules.Discoverer import Discoverer
from utils import unkn_labels as unk_labels
from utils.callbacks import mIoUEvaluatorCallback

SEED = 1234

parser = ArgumentParser()
parser.add_argument("-s", "--split", type=int, help="split", required=True)
parser.add_argument("--checkpoint", type=str, help="checkpoint to evaluate", required=True)
parser.add_argument("--dataset", choices=["SemanticKITTI", "SemanticPOSS"], default="SemanticPOSS", type=str, help="dataset")
parser.add_argument("--dataset_config", default=None, type=str, help="dataset config file")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="NOPS", type=str, help="wandb project")
parser.add_argument("--entity", default="luigiriz", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--set_deterministic", default=False, action="store_true")


def main(args):

    if args.set_deterministic:
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    run_name = "-".join([f"S{args.split}",
                        "discover_TEST", args.dataset, args.comment])
    wandb_logger = WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    if args.dataset_config is None:
        if args.dataset == "SemanticKITTI":
            args.dataset_config = "config/semkitti_dataset.yaml"
        elif args.dataset == "SemanticPOSS":
            args.dataset_config = "config/semposs_dataset.yaml"
        else:
            raise NameError(f"Dataset {args.dataset} not implemented")

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    unknown_labels = unk_labels.unknown_labels(
        split=args.split, dataset_config=dataset_config
    )

    number_of_unk = len(unknown_labels)

    label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
        unknown_labels, dataset_config["learning_map_inv"].keys()
    )

    args.num_classes = len(label_mapping)
    args.num_unlabeled_classes = number_of_unk
    args.num_labeled_classes = args.num_classes - args.num_unlabeled_classes

    mIoU_callback = mIoUEvaluatorCallback()
    csv_logger = CSVLogger(save_dir=args.log_dir)

    loggers = [wandb_logger, csv_logger] if wandb_logger is not None else [
        csv_logger]

    model = Discoverer.load_from_checkpoint(args.checkpoint,
                                            label_mapping=label_mapping,
                                            label_mapping_inv=label_mapping_inv,
                                            unknown_label=unknown_label,
                                            dataset_config=args.dataset_config)

    trainer = pl.Trainer(
        logger=loggers,
        gpus=-1,
        num_sanity_val_steps=0,
        callbacks=[mIoU_callback],
    )
    trainer.validate(model)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)