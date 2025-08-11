## Copyright (C) 2023 Philipp Benner

import torch

from typing import Union

from torch_geometric.data import Data
from torch_geometric.nn import (
    Sequential,
    GraphConv,
    CGConv,
    HeteroConv,
    ResGatedGraphConv,
    global_mean_pool,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

from .features_coding import NumOxidations, NumGeometries

from .model_layers import (
    TorchStandardScaler,
    ModelDense,
    ElementEmbedder,
    RBFEmbedding,
    PaddedEmbedder,
    SphericalBesselFunction,
)
from .model_gnn_config import GraphCoordinationNetConfig

## ----------------------------------------------------------------------------


class IdConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ) -> torch.Tensor:
        return x


## ----------------------------------------------------------------------------


class ModelGraphCoordinationNet(torch.nn.Module):
    def __init__(
        self,
        # Specify model components
        model_config=GraphCoordinationNetConfig,
        # Options for dense layers
        layers=[512, 128, 64, 1],
        **kwargs,
    ):
        super().__init__()

        # Feature dimensions
        dim_element = model_config["dim_element"]
        dim_oxidation = model_config["dim_oxidation"]
        dim_geometry = model_config["dim_geometry"]
        dim_csm = model_config["dim_csm"]
        dim_distance = model_config["dim_distance"]
        dim_angle = model_config["dim_angle"]

        dim_site = dim_element
        dim_ce = dim_element
        dim_ligand = dim_element

        if model_config["oxidations"]:
            dim_site += dim_oxidation
            dim_ce += dim_oxidation
            dim_ligand += dim_oxidation

        if model_config["distances"]:
            dim_ligand += dim_distance

        if model_config["geometries"]:
            dim_ce += dim_geometry

        if model_config["csms"]:
            dim_ce += dim_csm

        if model_config["angles"]:
            dim_ligand += dim_angle

        # The model config determines which components of the model
        # are active
        self.model_config = model_config
        # Optional scaler of model outputs (predictions)
        self.scaler_outputs = TorchStandardScaler(layers[-1])

        # RBF encoder
        if model_config["rbf_type"] == "Gaussian" or model_config["rbf_type"] is None:
            self.rbf_csm = RBFEmbedding(
                0.0, 1.0, bins=model_config["bins_csm"], edim=dim_csm
            )
            self.rbf_distances_1 = RBFEmbedding(
                0.0, 1.0, bins=model_config["bins_distance"], edim=dim_distance
            )
            self.rbf_distances_2 = RBFEmbedding(
                0.0, 1.0, bins=model_config["bins_distance"], edim=dim_distance
            )
            self.rbf_angles = RBFEmbedding(
                0.0, 1.0, bins=model_config["bins_angle"], edim=dim_angle
            )
        elif model_config["rbf_type"] == "Bessel":
            self.rbf_csm = SphericalBesselFunction(
                max_n=model_config["bins_csm"], edim=dim_csm
            )
            self.rbf_distances_1 = SphericalBesselFunction(
                max_n=model_config["bins_distance"], edim=dim_distance
            )
            self.rbf_distances_2 = SphericalBesselFunction(
                max_n=model_config["bins_distance"], edim=dim_distance
            )
            self.rbf_angles = SphericalBesselFunction(
                max_n=model_config["bins_angle"], edim=dim_angle
            )
        else:
            raise ValueError("Invalid RBF embedder")

        # Embeddings
        self.embedding_element = ElementEmbedder(
            dim_element, from_pretrained=True, freeze=True
        )
        self.embedding_oxidation = torch.nn.Embedding(NumOxidations, dim_oxidation)
        self.embedding_geometry = PaddedEmbedder(NumGeometries, dim_geometry)

        self.activation = torch.nn.ELU(inplace=True)

        sequential_layers = []

        for _ in range(model_config["num_convs"]):
            # Add convolution operation

            if model_config["conv_type"] == "CGConv":
                sequential_layers.append(
                    (
                        HeteroConv(
                            {
                                ("site", "*", "site"): CGConv(
                                    (dim_site, dim_site), add_self_loops=False, dim=0
                                ),
                                ("ligand", "*", "ce"): CGConv(
                                    (dim_ligand, dim_ce),
                                    add_self_loops=True,
                                    dim=dim_distance
                                    if model_config["distances"]
                                    else 0,
                                ),
                                ("ce", "*", "ligand"): CGConv(
                                    (dim_ce, dim_ligand),
                                    add_self_loops=True,
                                    dim=dim_distance
                                    if model_config["distances"]
                                    else 0,
                                ),
                            }
                        ),
                        "x, edge_index, edge_attr -> x",
                    )
                )

            elif model_config["conv_type"] == "ResGatedGraphConv":
                sequential_layers.append(
                    (
                        HeteroConv(
                            {
                                ("site", "*", "site"): ResGatedGraphConv(
                                    (dim_site, dim_site),
                                    dim_site,
                                    add_self_loops=False,
                                    edge_dim=None,
                                ),
                                ("ligand", "*", "ce"): ResGatedGraphConv(
                                    (dim_ligand, dim_ce),
                                    dim_ce,
                                    add_self_loops=True,
                                    edge_dim=dim_distance
                                    if model_config["distances"]
                                    else None,
                                ),
                                ("ce", "*", "ligand"): ResGatedGraphConv(
                                    (dim_ce, dim_ligand),
                                    dim_ligand,
                                    add_self_loops=True,
                                    edge_dim=dim_distance
                                    if model_config["distances"]
                                    else None,
                                ),
                            }
                        ),
                        "x, edge_index, edge_attr -> x",
                    )
                )

            else:
                raise ValueError("Invalid conv_type specified")

            # Add activation function
            sequential_layers.append(
                (
                    lambda x: {
                        k: v if k == "site" else self.activation(v)
                        for k, v in x.items()
                    },
                    "x -> x",
                ),
            )

        sequential_layers += [
            # Mixing layer
            # -------------------------------------------------------------------------------------------
            (
                HeteroConv(
                    {
                        ("ce", "*", "site"): GraphConv(
                            (dim_ce, dim_site),
                            dim_site,
                            add_self_loops=True,
                            bias=False,
                        ),
                    },
                    aggr="mean",
                ),
                "x, edge_index -> x",
            ),
            # Apply activation
            (lambda x: {k: self.activation(v) for k, v in x.items()}, "x -> x"),
            # Output layer
            # -------------------------------------------------------------------------------------------
            # Apply mean pooling
            (
                lambda x, batch: {
                    k: global_mean_pool(v, batch[k]) for k, v in x.items()
                },
                "x, batch -> x",
            ),
            # Extract only site features
            (lambda x: x["site"], "x -> x"),
        ]

        # Core graph network
        self.layers = Sequential("x, edge_index, edge_attr, batch", sequential_layers)

        # Final dense layer
        self.dense = ModelDense([dim_site] + layers, **kwargs)

    def forward(self, x_input):
        device = x_input["site"].x["elements"].device

        n_site = x_input["site"].x["elements"].shape[0]
        n_ce = x_input["ce"].x["elements"].shape[0]
        n_ligand = x_input["ligand"].x["elements"].shape[0]

        x_site = torch.cat(
            (
                self.embedding_element(x_input["site"].x["elements"]),
                self.embedding_oxidation(x_input["site"].x["oxidations"])
                if self.model_config["oxidations"]
                else torch.empty(n_site, 0, device=device),
            ),
            dim=1,
        )

        x_ce = torch.cat(
            (
                self.embedding_element(x_input["ce"].x["elements"]),
                self.embedding_oxidation(x_input["ce"].x["oxidations"])
                if self.model_config["oxidations"]
                else torch.empty(n_ce, 0, device=device),
                self.embedding_geometry(x_input["ce"].x["geometries"])
                if self.model_config["geometries"]
                else torch.empty(n_ce, 0, device=device),
                self.rbf_csm(x_input["ce"].x["csms"])
                if self.model_config["csms"]
                else torch.empty(n_ce, 0, device=device),
            ),
            dim=1,
        )

        x_ligand = torch.cat(
            (
                self.embedding_element(x_input["ligand"].x["elements"]),
                self.embedding_oxidation(x_input["ligand"].x["oxidations"])
                if self.model_config["oxidations"]
                else torch.empty(n_ligand, 0, device=device),
                self.rbf_distances_1(x_input["ligand"].x["distances"])
                if self.model_config["distances"]
                else torch.empty(n_ligand, 0, device=device),
                self.rbf_angles(x_input["ligand"].x["angles"])
                if self.model_config["angles"]
                else torch.empty(n_ligand, 0, device=device),
            ),
            dim=1,
        )

        # Concatenate embeddings to yield a single feature vector per node
        x = {
            "site": x_site,
            "ce": x_ce,
            "ligand": x_ligand,
        }
        edge_attr_dict = (
            {
                ("ligand", "*", "ce"): self.rbf_distances_2(
                    x_input["ligand", "*", "ce"].edge_attr
                ),
                ("ce", "*", "ligand"): self.rbf_distances_2(
                    x_input["ce", "*", "ligand"].edge_attr
                ),
            }
            if self.model_config["distances"]
            else {}
        )
        # Propagate features through graph network
        x = self.layers(x, x_input.edge_index_dict, edge_attr_dict, x_input.batch_dict)
        # Apply final dense layer
        x = self.dense(x)
        # Apply inverse transformation
        x = self.scaler_outputs.inverse_transform(x)

        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_grouped(self):
        return {"all": self.parameters()}
