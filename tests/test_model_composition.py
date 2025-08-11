## Copyright (C) 2023 Philipp Benner

import torch
import os
import pytest

from monty.serialization import loadfn

from coordinationnet.model_transformer import (
    ModelComposition,
    ModelSiteFeaturesTransformer,
)
from coordinationnet.model_transformer_data import BatchComposition, BatchSiteFeatures

## ----------------------------------------------------------------------------

root = os.path.realpath(os.path.dirname(__file__))

## ----------------------------------------------------------------------------


@pytest.fixture
def features_list():
    data = loadfn(os.path.join(root, "test_features.json.gz"))

    X = [value.encode() for _, value in data.items()]

    return X


## ----------------------------------------------------------------------------


def test_model_composition(features_list):
    b1 = BatchComposition(features_list)
    b2 = BatchSiteFeatures(features_list)

    edim = 4
    m1 = ModelComposition(edim)
    m2 = ModelSiteFeaturesTransformer(
        edim, transformer=False, oxidation=False, csms=False, ligands=False
    )
    m2.embedding_element = m1.embedding_element

    assert not torch.any(m1(b1) - m2(b2, None, None) > 1e-4).item()
