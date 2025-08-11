## Copyright (C) 2023 Philipp Benner

from .model_config import ModelConfig

## ----------------------------------------------------------------------------

_transformer_config = {
    "composition": False,
    "sites": False,
    "sites_oxid": False,
    "sites_ces": False,
    "site_features": False,
    "site_features_ces": False,
    "site_features_oxid": False,
    "site_features_csms": False,
    "site_features_ligands": False,
    "ligands": False,
    "ce_neighbors": False,
}

## ----------------------------------------------------------------------------

TransformerCoordinationNetConfig = ModelConfig(_transformer_config)

DefaultTransformerCoordinationNetConfig = ModelConfig(_transformer_config)(
    site_features=True,
    site_features_ces=True,
    site_features_oxid=True,
    site_features_csms=True,
)
