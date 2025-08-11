## Copyright (C) 2023 Philipp Benner

import os
import pickle

from coordinationnet.model_config import ModelConfig
from coordinationnet.model_gnn_config import _graph_config

## -----------------------------------------------------------------------------

root = os.path.realpath(os.path.dirname(__file__))

## -----------------------------------------------------------------------------


def test_config():
    config = ModelConfig(_graph_config)

    with open("test_config.pkl", "wb") as f:
        pickle.dump(config, f)

    with open("test_config.pkl", "rb") as f:
        config = pickle.load(f)
