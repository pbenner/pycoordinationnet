## Copyright (C) 2023 Philipp Benner

import math
import numpy as np
import pandas as pd
import torch
import os
import sympy

from functools import lru_cache
from math import pi, sqrt

from .features_coding import NumElements

## ----------------------------------------------------------------------------


class TorchStandardScaler(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Always use requires_grad=False, since we do not want to update
        # parameters during training. However, we must store mean and standard
        # deviations in a Parameter module, so that both get automatically
        # pushed to GPU when required
        self.register_buffer("mean", torch.tensor(dim * [0.0], requires_grad=False))
        self.register_buffer("std", torch.tensor(dim * [1.0], requires_grad=False))

    def fit(self, x):
        self.mean[:] = x.mean(0, keepdim=False)
        self.std[:] = x.std(0, keepdim=False, unbiased=False) + 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


## ----------------------------------------------------------------------------


class ModelDense(torch.nn.Module):
    def __init__(
        self,
        ks,
        skip_connections=True,
        dropout=False,
        layernorm=False,
        batchnorm=False,
        batchnorm_momentum=0.1,
        batchnorm_out=False,
        activation=torch.nn.ELU(),
        activation_out=None,
        seed=None,
    ):
        super().__init__()
        if len(ks) < 2:
            raise ValueError(
                "invalid argument: ks must have at least two values for input and output"
            )
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.activation = activation
        self.activation_out = activation_out
        self.skip_connections = skip_connections
        self.linear = torch.nn.ModuleList([])
        self.linear_skip = torch.nn.ModuleList([])
        self.layernorm = torch.nn.ModuleList([])
        self.batchnorm = torch.nn.ModuleList([])
        self.batchnorm_skip = torch.nn.ModuleList([])
        self.batchnorm_out = None
        self.dropout = torch.nn.ModuleList([])
        for i in range(0, len(ks) - 1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i + 1]))
            if type(dropout) == float:
                self.dropout.append(torch.nn.Dropout(dropout))
        for i in range(0, len(ks) - 2):
            if layernorm:
                self.layernorm.append(torch.nn.LayerNorm(ks[i + 1]))
            if batchnorm:
                self.batchnorm.append(
                    torch.nn.BatchNorm1d(ks[i + 1], momentum=batchnorm_momentum)
                )
            if skip_connections:
                self.linear_skip.append(torch.nn.Linear(ks[i], ks[i + 1], bias=False))
            if batchnorm and skip_connections:
                self.batchnorm_skip.append(
                    torch.nn.BatchNorm1d(ks[i + 1], momentum=batchnorm_momentum)
                )
        # Optional: batch norm for output layer
        if batchnorm_out:
            self.batchnorm_out = torch.nn.BatchNorm1d(
                ks[-1], momentum=batchnorm_momentum
            )

    def block(self, x, i):
        # First, apply dropout
        if len(self.dropout) > 0:
            x = self.dropout[i](x)
        # Apply linear layer
        y = self.linear[i](x)
        # Normalize output
        if len(self.layernorm) > 0:
            y = self.layernorm[i](y)
        if len(self.batchnorm) > 0:
            y = self.batchnorm[i](y)
        # Apply activation
        y = self.activation(y)
        # Apply skip-connections (ResNet)
        if len(self.linear_skip) > 0:
            x = self.linear_skip[i](x)
            if len(self.batchnorm_skip) > 0:
                x = self.batchnorm_skip[i](x)
        if type(self.skip_connections) == int and i % self.skip_connections == 0:
            y = (y + x) / 2.0
        if type(self.skip_connections) == bool and self.skip_connections:
            y = (y + x) / 2.0

        return y

    def block_final(self, x):
        # First, apply dropout
        if len(self.dropout) > 0:
            x = self.dropout[-1](x)
        # Apply final layer if available
        if len(self.linear) >= 1:
            x = self.linear[-1](x)
        # Normalize output
        if self.batchnorm_out is not None:
            x = self.batchnorm_out(x)
        # Apply output activation if available
        if self.activation_out is not None:
            x = self.activation_out(x)

        return x

    def forward(self, x):
        # Apply innear layers
        for i in range(len(self.linear) - 1):
            x = self.block(x, i)
        # Apply final layer if available
        x = self.block_final(x)

        return x


## ----------------------------------------------------------------------------


class RBFLayer(torch.nn.Module):
    def __init__(self, vmin: float, vmax: float, bins: int = 40, gamma: float = None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
        self.gamma = bins / math.fabs(vmax - vmin) if gamma is None else gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gamma * (x.unsqueeze(1) - self.centers) ** 2
        return torch.exp(-x)


## ----------------------------------------------------------------------------


class AngleLayer(torch.nn.Module):
    def __init__(self, edim, layers, **kwargs):
        super().__init__()

        self.dense = ModelDense([edim + 3] + layers + [edim], **kwargs)

    def forward(self, x, x_distances, x_angles):
        x_angles = x_angles / 180 * 2 * torch.pi
        x_angles = torch.cat((torch.sin(x_angles), torch.cos(x_angles)), dim=1)
        return self.dense(torch.cat((x, x_distances, x_angles), dim=1))


## ----------------------------------------------------------------------------


class PaddedEmbedder(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings + 1, embedding_dim, **kwargs)
        self.weight.data[num_embeddings - 1][:] = 0


## ----------------------------------------------------------------------------


class ElementEmbedder(torch.nn.Module):
    def __init__(self, edim, from_pretrained=True, freeze=True):
        super().__init__()
        if from_pretrained:
            currdir = os.path.dirname(os.path.realpath(__file__))
            mat2vec = os.path.join(currdir, "model_layers_mat2vec.csv")
            embedding = pd.read_csv(mat2vec, index_col=0).values
            feat_size = embedding.shape[-1]
            embedding = np.concatenate([embedding, np.zeros((1, feat_size))])
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
            self.embedding = torch.nn.Embedding.from_pretrained(
                embedding, freeze=freeze
            )
            if edim == 200:
                self.linear = torch.nn.Identity()
            else:
                self.linear = torch.nn.Linear(feat_size, edim, bias=False)
        else:
            self.embedding = torch.nn.Embedding(NumElements + 1, edim)
            self.linear = torch.nn.Identity()

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x


## ----------------------------------------------------------------------------


class ZeroPadder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert self.dim >= x.shape[1]

        if self.dim > x.shape[1]:
            z = torch.zeros((x.shape[0], self.dim - x.shape[1]), device=x.device)
            x = torch.cat((x, z), dim=1)

        return x


## ----------------------------------------------------------------------------


class RBFEmbedding(torch.nn.Module):
    def __init__(
        self, vmin: float, vmax: float, bins: int = 40, edim=128, gamma: float = None
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
        self.gamma = bins / math.fabs(vmax - vmin) if gamma is None else gamma
        self.embedding = torch.nn.Embedding(bins, edim)

    #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gamma * (x.unsqueeze(1) - self.centers) ** 2
        x = torch.exp(-x)
        x = x @ self.embedding.weight
        return x


## ----------------------------------------------------------------------------


class SphericalBesselFunction(torch.nn.Module):
    """Calculate the spherical Bessel function based on sympy + pytorch implementations."""

    def __init__(
        self,
        max_l: int = None,
        max_n: int = 5,
        edim: int = 100,
        cutoff: float = 5.0,
        smooth: bool = False,
    ):
        """Args:
        max_l: int, max order (excluding l)
        max_n: int, max number of roots used in each l
        cutoff: float, cutoff radius
        smooth: Whether to smooth the function.
        """
        super().__init__()

        if max_l is None:
            if edim % max_n != 0:
                raise ValueError("edim must be divisible by max_n")

            max_l = int(edim / max_n)

        self.max_l = max_l
        self.max_n = max_n
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.smooth = smooth
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()

        currdir = os.path.dirname(os.path.abspath(__file__))
        self.register_buffer(
            "SPHERICAL_BESSEL_ROOTS",
            torch.tensor(
                np.load(os.path.join(currdir, "model_layers_sbroots.npy")),
                dtype=torch.float,
            ),
        )

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list:
        """Spherical basis functions based on Rayleigh formula. This function
        generates
        symbolic formula.

        Returns: list of symbolic functions
        """
        x = sympy.symbols("x")
        funcs = [
            sympy.expand_func(sympy.functions.special.bessel.jn(i, x))
            for i in range(self.max_l + 1)
        ]
        return [sympy.lambdify(x, func, torch) for func in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> list:
        return self._get_lambda_func(max_n=self.max_n, cutoff=self.cutoff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Args:
            r: torch.Tensor, distance tensor, 1D.

        Returns:
            torch.Tensor: [n, max_n * max_l] spherical Bessel function results
        """
        assert len(r.shape) == 1

        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r):
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r):
        r_c = r.clone()
        r_c[r_c >= self.cutoff] = self.cutoff.to(r_c.dtype)
        r_c[r_c <= 1e-8] = 1e-8
        roots = self.SPHERICAL_BESSEL_ROOTS[: self.max_l, : self.max_n]

        results = []
        factor = sqrt(2.0 / self.cutoff**3)
        for i in range(self.max_l):
            root = roots[i]
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r_c[:, None] * root[None, :] / self.cutoff)
                * factor
                / torch.abs(func_add1(root[None, :]))
            )
        return torch.cat(results, axis=1)

    @staticmethod
    def rbf_j0(r, cutoff: float = 5.0, max_n: int = 3):
        """Spherical Bessel function of order 0, ensuring the function value
        vanishes at cutoff.

        Args:
            r: torch.Tensor pytorch tensors
            cutoff: float, the cutoff radius
            max_n: int max number of basis

        Returns:
            basis function expansion using first spherical Bessel function
        """
        n = (torch.arange(1, max_n + 1)).type(dtype=torch.float)[None, :]
        r = r[:, None]
        return sqrt(2.0 / cutoff) * torch.sin(n * pi / cutoff * r) / r

    @lru_cache(maxsize=128)
    @staticmethod
    def _get_lambda_func(max_n, cutoff: float = 5.0):
        r = sympy.symbols("r")
        en = [i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1) for i in range(max_n)]

        dn = [1.0]
        for i in range(1, max_n):
            dn_value = 1 - en[i] / dn[-1]
            dn.append(dn_value)

        fnr = [
            (-1) ** i
            * sqrt(2.0)
            * pi
            / cutoff**1.5
            * (i + 1)
            * (i + 2)
            / sympy.sqrt(1.0 * (i + 1) ** 2 + (i + 2) ** 2)
            * (
                sympy.sin(r * (i + 1) * pi / cutoff) / (r * (i + 1) * pi / cutoff)
                + sympy.sin(r * (i + 2) * pi / cutoff) / (r * (i + 2) * pi / cutoff)
            )
            for i in range(max_n)
        ]

        gnr = [fnr[0]]
        for i in range(1, max_n):
            gnr_value = (
                1
                / sympy.sqrt(dn[i])
                * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1])
            )
            gnr.append(gnr_value)
        return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]
