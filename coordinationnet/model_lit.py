## Copyright (C) 2023 Philipp Benner

import dill
import shutil
import torch
import pytorch_lightning as pl
import os

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

from .model_optimizer import Lamb

## ----------------------------------------------------------------------------

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

## ----------------------------------------------------------------------------


class LitMetricTracker(pl.callbacks.Callback):
    def __init__(self):
        self.val_error_batch = []
        self.val_error = []
        self.train_error_batch = []
        self.train_error = []
        self.test_y = []
        self.test_y_hat = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_error_batch.append(outputs["loss"].item())

    def on_train_epoch_end(self, *args, **kwargs):
        self.train_error.append(torch.mean(torch.tensor(self.train_error_batch)).item())
        self.train_error_batch = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_error_batch.append(outputs["val_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_error.append(torch.mean(torch.tensor(self.val_error_batch)).item())
        self.val_error_batch = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_y.append(outputs["y"].detach().cpu())
        self.test_y_hat.append(outputs["y_hat"].detach().cpu())

    @property
    def test_predictions(self):
        y = torch.cat(self.test_y)
        y_hat = torch.cat(self.test_y_hat)
        return y, y_hat


## ----------------------------------------------------------------------------


class LitProgressBar(pl.callbacks.progress.TQDMProgressBar):
    # Disable validation progress bar
    def on_validation_start(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass


## ----------------------------------------------------------------------------


class LitVerboseOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.param_groups_copy = None
        self._copy_parameters()

    def _copy_parameters(self):
        self.param_groups_copy = []
        for param_group in self.optimizer.param_groups:
            param_group_copy = []
            for parameters in param_group["params"]:
                param_group_copy.append(torch.clone(parameters.data))
            self.param_groups_copy.append(param_group_copy)

    def _print_difference(self):
        delta_min = torch.inf
        delta_max = 0.0
        delta_sum = 0.0
        delta_n = 1.0

        for i, param_group in enumerate(self.optimizer.param_groups):
            for j, parameters in enumerate(param_group["params"]):
                delta = torch.sum(
                    torch.abs(self.param_groups_copy[i][j] - parameters.data)
                ).item()

                if delta_min > delta:
                    delta_min = delta
                if delta_max < delta:
                    delta_max = delta

                delta_sum += delta
                delta_n += 1.0

                print(f"update ({i},{j}): {delta:15.10f}")

        print(f"update max :", delta_max)
        print(f"update min :", delta_min)
        print(f"update mean:", delta_sum / delta_n)
        print()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self, *args, **kwargs):
        return self.optimizer.state_dict(*args, **kwargs)

    def step(self, closure=None):
        self._copy_parameters()
        self.optimizer.step(closure=closure)
        self._print_difference()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state


## ----------------------------------------------------------------------------


class LitDataset(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data,
        val_size=0.2,
        batch_size=32,
        num_workers=2,
        default_root_dir=None,
        load_cached_data=None,
        seed=42,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.val_size = val_size
        self.batch_size = batch_size
        self.data = data
        self.default_root_dir = default_root_dir
        self.load_cached_data = load_cached_data
        self.seed = seed

    @property
    def cache_path(self):
        if self.load_cached_data is None:
            return None

        if self.default_root_dir is not None:
            if not os.path.exists(self.default_root_dir):
                os.makedirs(self.default_root_dir)

            return os.path.join(self.default_root_dir, self.load_cached_data)

        return self.load_cached_data

    # This function is called by lightning trainer class with
    # the corresponding stage option
    def setup(self, stage: Optional[str] = None):
        if self.cache_path is not None:
            with open(self.cache_path, "rb") as f:
                self.data = dill.load(f)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == None:
            # Take a piece of the training data for validation
            self.data_train, self.data_val = torch.utils.data.random_split(
                self.data,
                [1.0 - self.val_size, self.val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == None:
            self.data_test = self.data

        # Assign predict dataset for use in dataloader(s)
        if stage == "predict" or stage == None:
            self.data_predict = self.data

    # Custom method to create a data loader
    @abstractmethod
    def get_dataloader(self, data):
        pass

    # The following functions are called by the trainer class to
    # obtain data loaders
    def train_dataloader(self):
        return self.get_dataloader(self.data_train)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)

    def predict_dataloader(self):
        return self.get_dataloader(self.data_predict)


## ----------------------------------------------------------------------------


class LitModel(pl.LightningModule):
    def __init__(
        self,
        # pytorch model class and loss function
        model,
        loss=torch.nn.L1Loss(),
        # Trainer options
        patience_sd=10,
        patience_es=50,
        max_epochs=1000,
        accelerator="gpu",
        devices=[0],
        plugins=None,
        default_root_dir="checkpoints",
        # Data options
        val_size=0.1,
        batch_size=128,
        num_workers=2,
        # Learning rate
        lr=1e-3,
        lr_groups={},
        # Weight decay
        weight_decay=0.0,
        weight_decay_groups={},
        # Other hyperparameters
        betas=(0.9, 0.95),
        factor=0.8,
        # Optimizer and scheduler selection
        scheduler=None,
        optimizer="AdamW",
        optimizer_verbose=False,
        seed=42,
        **kwargs,
    ):
        super().__init__()

        strategy = "auto"

        if len(devices) > 1:
            strategy = "ddp_find_unused_parameters_true"

        # Save all hyperparameters to `hparams` (e.g. lr)
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_verbose = optimizer_verbose
        self.scheduler = scheduler
        self.model = model(**kwargs)
        self.loss = loss

        self.trainer_options = {
            "patience_sd": patience_sd,
            "patience_es": patience_es,
            "max_epochs": max_epochs,
            "accelerator": accelerator,
            "devices": devices,
            "strategy": strategy,
            "default_root_dir": default_root_dir,
            "plugins": plugins,
        }
        self.data_options = {
            "val_size": val_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "default_root_dir": default_root_dir,
            "seed": seed,
        }
        self._reset_trainer()

    def configure_optimizers(self):
        # Get learning rates
        lr = self.hparams["lr"]
        lr_groups = self.hparams["lr_groups"]
        # Get weight_decay parameters
        weight_decay = self.hparams["weight_decay"]
        weight_decay_groups = self.hparams["weight_decay_groups"]

        # Get parameter groups
        parameter_groups = []
        for name, params in self.model.parameters_grouped().items():
            group = {"params": params}

            if name in lr_groups:
                group["lr"] = lr_groups[name]
            if name in weight_decay_groups:
                group["weight_decay"] = weight_decay_groups[name]

            parameter_groups.append(group)

        # Initialize optimizer
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                parameter_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=self.hparams["betas"],
            )
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=self.hparams["betas"],
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                parameter_groups, lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(
                parameter_groups, lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer == "Lamb":
            optimizer = Lamb(parameter_groups, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        if self.optimizer_verbose:
            optimizer = LitVerboseOptimizer(optimizer)

        # Initialize scheduler
        if self.scheduler is None:
            scheduler = []
        elif self.scheduler == "cycling":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.CyclicLR(
                        optimizer,
                        base_lr=1e-4,
                        max_lr=5e-3,
                        step_size_up=10,
                        cycle_momentum=False,
                        verbose=True,
                    ),
                    "interval": "epoch",
                    "monitor": "val_loss",
                }
            ]
        elif self.scheduler == "plateau":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        patience=self.hparams["patience_sd"],
                        factor=self.hparams["factor"],
                        mode="min",
                        verbose=True,
                    ),
                    "interval": "epoch",
                    "monitor": "train_loss",
                }
            ]
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        return [optimizer], scheduler

    def forward(self, x, **kwargs):
        return self.model.forward(x, **kwargs)

    def training_step(self, batch, batch_index):
        """Train model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        # Call the model
        y_hat = self.model(X_batch)
        loss = self.loss(y_hat, y_batch)
        # Send metrics to progress bar. We also don't want results
        # logged at every step, but let the logger accumulate the
        # results at the end of every epoch
        self.log(
            f"train_loss",
            loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=y_batch.shape[0],
        )
        # Return whatever we might need in callbacks. Lightning automtically minimizes
        # the item called 'loss', which must be present in the returned dictionary
        return {"loss": loss}

    def validation_step(self, batch, batch_index):
        """Validate model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        # Call the model
        y_hat = self.model(X_batch)
        loss = self.loss(y_hat, y_batch)
        # Send metrics to progress bar. We also don't want results
        # logged at every step, but let the logger accumulate the
        # results at the end of every epoch
        self.log(
            "val_loss",
            loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=y_batch.shape[0],
        )
        # Return whatever we might need in callbacks
        return {"val_loss": loss}

    def test_step(self, batch, batch_index):
        """Test model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        # Call the model
        y_hat = self.model(X_batch)
        loss = self.loss(y_hat, y_batch)
        # Log whatever we want to aggregate later
        self.log("test_loss", loss, batch_size=y_batch.shape[0])
        # Return whatever we might need in callbacks
        return {"y": y_batch, "y_hat": y_hat, "test_loss": loss}

    def predict_step(self, batch, batch_index):
        """Prediction on a single batch"""
        return self.model(batch[0])

    def _reset_trainer(self):
        self.trainer_matric_tracker = LitMetricTracker()
        self.trainer_early_stopping = pl.callbacks.EarlyStopping(
            monitor="train_loss", patience=self.trainer_options["patience_es"]
        )
        self.trainer_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1, monitor="val_loss", mode="min"
        )

        # self.trainer is a pre-defined getter/setter in the LightningModule
        self.trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=True,
            max_epochs=self.trainer_options["max_epochs"],
            accelerator=self.trainer_options["accelerator"],
            devices=self.trainer_options["devices"],
            strategy=self.trainer_options["strategy"],
            default_root_dir=self.trainer_options["default_root_dir"],
            callbacks=[
                LitProgressBar(),
                self.trainer_early_stopping,
                self.trainer_checkpoint_callback,
                self.trainer_matric_tracker,
            ],
        )

    def _train(self, data):
        # Train model on train data. The fit method returns just None
        self.trainer.fit(self, data)

        # Get best model
        best_model = self.load_from_checkpoint(
            self.trainer_checkpoint_callback.best_model_path
        )
        # Lightning removes all training related objects before
        # saving the model. Recover all training components
        best_model.trainer = self.trainer
        best_model.trainer_matric_tracker = self.trainer_matric_tracker
        best_model.trainer_early_stopping = self.trainer_early_stopping
        best_model.trainer_checkpoint_callback = self.trainer_checkpoint_callback

        stats = {
            "best_val_error": self.trainer_checkpoint_callback.best_model_score.item(),
            "train_error": self.trainer_matric_tracker.train_error,
            "val_error": self.trainer_matric_tracker.val_error,
        }

        return best_model, stats

    def _test(self, data):
        # Train model on train data. The test method returns accumulated
        # statistics sent to the logger
        stats = self.trainer.test(self, data)

        # There should only be one entry in stats
        assert len(stats) == 1

        # Get targets and predictions
        y, y_hat = self.trainer_matric_tracker.test_predictions

        return y, y_hat, stats[0]

    def _predict(self, data):
        # Train model on train data. The test method returns accumulated
        # statistics sent to the logger
        return torch.cat(self.trainer.predict(self, data))

    def _clone(self):
        model = deepcopy(self)
        model._reset_trainer()

        return model
