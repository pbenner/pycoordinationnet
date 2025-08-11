## Copyright (C) 2023 Philipp Benner

import dill
import torch

from sklearn.model_selection import KFold

from .model_data import CoordinationFeaturesData
from .model_gnn import ModelGraphCoordinationNet
from .model_gnn_data import GraphCoordinationFeaturesLoader, GraphCoordinationData
from .model_lit import LitModel, LitDataset

## ----------------------------------------------------------------------------


class LitGraphCoordinationFeaturesData(LitDataset):
    def __init__(self, data: CoordinationFeaturesData, verbose=False, **kwargs):
        self.data_raw = data

        super().__init__(None, load_cached_data="dataset.dill", **kwargs)

    def prepare_data(self):
        data = GraphCoordinationData(self.data_raw, verbose=True)

        with open(self.cache_path, "wb") as f:
            dill.dump(data, f)

    # Custom method to create a data loader
    def get_dataloader(self, data):
        return GraphCoordinationFeaturesLoader(
            data, batch_size=self.batch_size, num_workers=self.num_workers
        )


## ----------------------------------------------------------------------------


class GraphCoordinationNet:
    def __init__(self, **kwargs):
        self.lit_model = LitModel(ModelGraphCoordinationNet, **kwargs)

        if self.lit_model.global_rank == 0:
            print(f"{self.lit_model.model.model_config}")

            print(
                f"Creating a model with {self.lit_model.model.n_parameters:,} parameters"
            )

    def fit_scaler(self, data: LitGraphCoordinationFeaturesData):
        y = torch.cat([y_batch for _, y_batch in data.data_raw])

        self.lit_model.model.scaler_outputs.fit(y)

    def train(self, data: CoordinationFeaturesData):
        self.lit_model.print(f"{self.lit_model.model.model_config}")

        self.lit_model.print(
            f"Creating a GNN model with {self.lit_model.model.n_parameters:,} parameters"
        )

        data = LitGraphCoordinationFeaturesData(
            data,
            **self.lit_model.data_options,
            verbose=(self.lit_model.global_rank == 0),
        )

        # Fit scaler to target values. The scaling of model outputs is done
        # by the model itself
        self.fit_scaler(data)

        self.lit_model, stats = self.lit_model._train(data)

        return stats

    def test(self, data: CoordinationFeaturesData):
        data = LitGraphCoordinationFeaturesData(
            data,
            **self.lit_model.data_options,
            verbose=(self.lit_model.global_rank == 0),
        )

        return self.lit_model._test(data)

    def predict(self, data: CoordinationFeaturesData):
        data = LitGraphCoordinationFeaturesData(
            data,
            **self.lit_model.data_options,
            verbose=(self.lit_model.global_rank == 0),
        )

        return self.lit_model._predict(data)

    def cross_validation(self, data, n_splits, shuffle=True, seed=None):
        if n_splits < 2:
            raise ValueError(
                f"k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}"
            )

        if seed is None:
            seed = self.lit_model.data_options["seed"]

        y_hat = torch.tensor([], dtype=torch.float)
        y = torch.tensor([], dtype=torch.float)

        initial_model = self.lit_model

        for fold, (index_train, index_test) in enumerate(
            KFold(n_splits, shuffle=shuffle, random_state=seed).split(data)
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

    @classmethod
    def load(cls, filename: str) -> "GraphCoordinationNet":
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
