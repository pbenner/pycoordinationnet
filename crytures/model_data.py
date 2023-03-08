import numpy as np
import torch

from monty.serialization import loadfn

## ----------------------------------------------------------------------------

class CoordinationFeaturesData(torch.utils.data.Dataset):

    def __init__(self, X, y = None) -> None:

        if y is not None:
            if isinstance(y, torch.Tensor):
                self.y = y
            else:
                self.y = torch.tensor(y)

        self.X = X

        for i, features in enumerate(X):
            if not features.encoded:
                self.X[i] = features.encode()

    def __len__(self):
        return len(self.X)

    # Called by pytorch DataLoader to collect items that
    # are joined later by collate_fn into a batch
    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        else:
            return self.X[index], None