import os
import torch
import re
from pytorch_lightning import Callback

class mIoUEvaluatorCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        interesting_metric_regex = re.compile(r'train/IoU/[\S]+_epoch')
        IoU_list = []
        callback_metrics = trainer.callback_metrics
        for key in callback_metrics.keys():
            mo = interesting_metric_regex.search(key)
            if mo is not None:
                IoU_list.append(callback_metrics[key])
        if IoU_list:
            mIoU = torch.mean(torch.stack(IoU_list))
            pl_module.log('train/mIoU', mIoU, rank_zero_only=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        interesting_metric_regex = re.compile(r'valid/IoU/[\S]+')
        IoU_list = []
        callback_metrics = trainer.callback_metrics
        for key in callback_metrics.keys():
            mo = interesting_metric_regex.search(key)
            if mo is not None:
                IoU_list.append(callback_metrics[key])
        if IoU_list:
            mIoU = torch.mean(torch.stack(IoU_list))
            pl_module.log('valid/mIoU', mIoU, rank_zero_only=True)

    def on_test_epoch_end(self, trainer, pl_module):
        interesting_metric_regex = re.compile(r'test/IoU/[\S]+')
        IoU_list = []
        callback_metrics = trainer.callback_metrics
        for key in callback_metrics.keys():
            mo = interesting_metric_regex.search(key)
            if mo is not None:
                IoU_list.append(callback_metrics[key])
        if IoU_list:
            mIoU = torch.mean(torch.stack(IoU_list))
            pl_module.log('test/mIoU', mIoU, rank_zero_only=True)
            
class PretrainCheckpointCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_filename = (
            "-".join(
                [
                    "pretrain",
                    str(pl_module.hparams.split),
                    pl_module.hparams.dataset,
                    pl_module.hparams.comment,
                ]
            )
            + ".ckpt"
        )
        checkpoint_path = os.path.join(
            pl_module.hparams.checkpoint_dir, checkpoint_filename)
        torch.save(pl_module.model.state_dict(), checkpoint_path)