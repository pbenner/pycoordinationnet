## Copyright (C) 2023 Philipp Benner

from .model_config import ModelConfig

## ----------------------------------------------------------------------------

_graph_config = {
    "num_convs": 2,
    "conv_type": "ResGatedGraphConv",
    "rbf_type": "Gaussian",
    "dim_element": 200,
    "dim_oxidation": 10,
    "dim_geometry": 10,
    "dim_csm": 128,
    "dim_distance": 128,
    "dim_angle": 128,
    "bins_csm": 20,
    "bins_distance": 20,
    "bins_angle": 20,
    "oxidations": True,
    "distances": True,
    "geometries": True,
    "csms": True,
    "angles": True,
}

## ----------------------------------------------------------------------------

GraphCoordinationNetConfig = ModelConfig(_graph_config)
