## Copyright (C) 2023 Philipp Benner

import dill
import torch
import numpy as np

from copy import copy
from typing import Any
from pymatgen.core.structure import Structure

from .features_datatypes import CoordinationFeatures

## ----------------------------------------------------------------------------


class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, X: list[Any], y=None) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    # Called by pytorch DataLoader to collect items that
    # are joined later by collate_fn into a batch
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as f:
            data = dill.load(f)

        if not isinstance(data, cls):
            raise ValueError(
                f"file {filename} contains incorrect data class {type(data)}"
            )

        return data

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            dill.dump(self, f)


## ----------------------------------------------------------------------------


class CoordinationFeaturesData(GenericDataset):
    def __init__(self, X: list[Any], y=None, verbose=False) -> None:
        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            raise ValueError(
                f"X must be of type list or numpy array, but got type {type(X)}"
            )

        if y is None:
            y = len(X) * [None]
        else:
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y)

        for i, item in enumerate(X):
            if isinstance(item, Structure):
                if verbose:
                    print(f"Featurizing structure {i + 1}/{len(X)}")
                X[i] = CoordinationFeatures.from_structure(item, encode=True)
            elif isinstance(item, CoordinationFeatures):
                if not item.encoded:
                    X[i] = item.encode()
            else:
                raise ValueError(
                    f"Items in X must be of type CoordinationFeatures or Structure, but item {i} is of type {type(item)}"
                )

        super().__init__(X, y)


## ----------------------------------------------------------------------------


class Batch:
    # This function is used by the estimator to push
    # data to GPU
    def to(self, device=None):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, "to"):
                result.__setattr__(attr, value.to(device=device))
        return result

    # This function will be called by the pytorch DataLoader
    # after collate_fn has assembled the batch
    def pin_memory(self):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, "pin_memory"):
                result.__setattr__(attr, value.pin_memory())
        return result

    def share_memory_(self):
        result = copy(self)
        for attr, value in result.__dict__.items():
            if hasattr(value, "share_memory_"):
                result.__setattr__(attr, value.share_memory_())
        return result
