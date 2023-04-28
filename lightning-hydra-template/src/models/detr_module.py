from typing import Any

import torch
from lightning import LightningModule
from src.models.detr.models import build_model
import src.models.detr.util.misc as utils


class DETRModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        args,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net, self.criterion, self.postprocessors = build_model(args)

        self.train_metric_logger = self.make_metric("train")
        self.val_metric_logger = self.make_metric("val")

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        losses = sum(
            loss[k] * self.criterion.weight_dict[k]
            for k in loss.keys()
            if k in self.criterion.weight_dict
        )

        # update and log metrics
        loss_reduced = utils.reduce_dict(loss)
        loss_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_reduced.items()}
        loss_reduced_scaled = {
            k: v * self.criterion.weight_dict[k]
            for k, v in loss_reduced.items()
            if k in self.criterion.weight_dict
        }
        losses_reduced_scaled = sum(loss_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        self.train_metric_logger.update(
            loss=loss_value, **loss_reduced_scaled, **loss_reduced_unscaled
        )
        self.train_metric_logger.update(class_error=loss_reduced["class_error"])

        self.log(
            f"train_loss",
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return losses

    def on_train_epoch_end(self):
        self.train_metric_logger.synchronize_between_processes()
        print("Averaged stats:", self.train_metric_logger)
        self.train_metric_logger = self.make_metric("train")

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        losses = sum(
            loss[k] * self.criterion.weight_dict[k]
            for k in loss.keys()
            if k in self.criterion.weight_dict
        )
        # update and log metrics
        loss_dict_reduced = utils.reduce_dict(loss)
        loss_dict_reduced_scaled = {
            k: v * self.criterion.weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in self.criterion.weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        self.val_metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        self.val_metric_logger.update(class_error=loss_dict_reduced["class_error"])

        self.log(
            f"val_loss",
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        self.val_metric_logger.synchronize_between_processes()
        print("Averaged stats:", self.val_metric_logger)
        self.val_metric_logger = self.make_metric("val")

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        self.train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        return {"optimizer": optimizer}

    def make_metric(self, type):
        if type == "train":
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter(
                "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
            )
            metric_logger.add_meter(
                "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
            )
        elif type == "val":
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter(
                "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
            )
        return metric_logger


if __name__ == "__main__":
    pass
