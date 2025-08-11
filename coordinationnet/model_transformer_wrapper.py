## Copyright (C) 2023 Philipp Benner

import dill
import torch

from typing import Union
from sklearn.model_selection import KFold

from .model_data import CoordinationFeaturesData
from .model_transformer import ModelCoordinationNet
from .model_transformer_data import (
    BatchedTransformerCoordinationFeaturesData,
    TransformerCoordinationFeaturesLoader,
)
from .model_lit import LitModel, LitDataset

## ----------------------------------------------------------------------------


class LitTransformerCoordinationFeaturesData(LitDataset):
    def __init__(
        self,
        data: CoordinationFeaturesData,
        val_size=0.2,
        batch_size=32,
        num_workers=2,
        default_root_dir=None,
        seed=None,
    ):
        super().__init__(
            data, val_size=val_size, batch_size=batch_size, num_workers=num_workers
        )

    # Custom method to create a data loader
    def get_dataloader(self, data):
        return TransformerCoordinationFeaturesLoader(
            data, batch_size=self.batch_size, num_workers=self.num_workers
        )


## ----------------------------------------------------------------------------


class TransformerCoordinationNet:
    def __init__(self, cache_file=None, **kwargs):
        self.lit_model = LitModel(ModelCoordinationNet, **kwargs)
        self.cache_file = cache_file

        if self.lit_model.global_rank == 0:
            print(f"{self.lit_model.model.model_config}")

            print(
                f"Creating a model with {self.lit_model.model.n_parameters:,} parameters"
            )

    def fit_scaler(self, data: LitTransformerCoordinationFeaturesData):
        y = torch.cat([y_batch for _, y_batch in data.get_dataloader(data.data)])
        self.lit_model.model.scaler_outputs.fit(y)

    def train(
        self,
        data: Union[
            CoordinationFeaturesData, BatchedTransformerCoordinationFeaturesData
        ],
    ):
        data = self.prepare_data(data)
        data = LitTransformerCoordinationFeaturesData(
            data, **self.lit_model.data_options
        )

        # Fit scaler to target values. The scaling of model outputs is done
        # by the model itself
        self.fit_scaler(data)

        self.lit_model, stats = self.lit_model._train(data)

        return stats

    def test(
        self,
        data: Union[
            CoordinationFeaturesData, BatchedTransformerCoordinationFeaturesData
        ],
    ):
        data = self.prepare_data(data)
        data = LitTransformerCoordinationFeaturesData(
            data, **self.lit_model.data_options
        )

        return self.lit_model._test(data)

    def predict(self, data):
        data = self.prepare_data(data)
        data = LitTransformerCoordinationFeaturesData(
            data, **self.lit_model.data_options
        )

        return self.lit_model._predict(data)

    def cross_validation(self, data, n_splits, shuffle=True, random_state=42):
        if n_splits < 2:
            raise ValueError(
                f"k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}"
            )

        data = self.prepare_data(data)

        y_hat = torch.tensor([], dtype=torch.float)
        y = torch.tensor([], dtype=torch.float)

        initial_model = self.lit_model

        for fold, (index_train, index_test) in enumerate(
            KFold(n_splits, shuffle=shuffle, random_state=random_state).split(data)
        ):
            if self.lit_model.global_rank == 0:
                print(f"Training fold {fold + 1}/{n_splits}...")

            data_train = torch.utils.data.Subset(data, index_train)
            data_test = torch.utils.data.Subset(data, index_test)

            # Clone model
            self.lit_model = initial_model._clone()

            # Train model
            best_val_score = self.train(data_train)["best_val_error"]

            # Test model
            test_y, test_y_hat, _ = self.test(data_test)

            # Print score
            if self.lit_model.global_rank == 0:
                print(f"Best validation score: {best_val_score}")

            # Save predictions for model evaluation
            y_hat = torch.cat((y_hat, test_y_hat))
            y = torch.cat((y, test_y))

        # Reset model
        self.lit_model = initial_model

        # Compute final test score
        test_loss = self.lit_model.loss(y_hat, y).item()

        return test_loss, y, y_hat

    def prepare_data(self, data):
        if isinstance(data, torch.utils.data.Subset):
            if not isinstance(data.dataset, BatchedTransformerCoordinationFeaturesData):
                raise ValueError(
                    f"Data Subset must contain dataset of type BatchedCoordinationFeaturesData, but got type {type(data.dataset)}"
                )

        else:
            if not (
                isinstance(data, CoordinationFeaturesData)
                or isinstance(data, BatchedTransformerCoordinationFeaturesData)
            ):
                raise ValueError(
                    f"Data must be given as CoordinationFeaturesData or BatchedCoordinationFeaturesData, but got type {type(data)}"
                )

            if isinstance(data, CoordinationFeaturesData):
                data = BatchedTransformerCoordinationFeaturesData(
                    data,
                    self.lit_model.model.model_config,
                    self.lit_model.data_options["batch_size"],
                    cache_file=self.cache_file,
                )

        return data

    @classmethod
    def load(cls, filename: str) -> "TransformerCoordinationNet":
        with open(filename, "rb") as f:
            model = dill.load(f)

        if not isinstance(model, cls):
            raise ValueError(
                f"file {filename} contains incorrect model class {type(model)}"
            )

        return model

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            dill.dump(self, f)
