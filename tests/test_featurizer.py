## Copyright (C) 2023 Sasan Amariamir, Philipp Benner

import numpy as np
import os
import pytest

from pymatgen.core import Structure
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions

from monty.serialization import loadfn

from coordinationnet import CoordinationFeatures, mp_icsd_clean
from coordinationnet.features_featurizer import (
    analyze_environment,
    compute_features_first_degree,
)
from coordinationnet.features_utility import oxide_check

## -----------------------------------------------------------------------------

root = os.path.realpath(os.path.dirname(__file__))

## -----------------------------------------------------------------------------


@pytest.fixture
def testData():
    testData = loadfn(os.path.join(root, "test_data.json.gz"))
    for Tdatum in testData:
        Tdatum["structure"] = Structure.from_dict(Tdatum["structure"])
    return testData


## -----------------------------------------------------------------------------


def test_exper_data_cleaning(testData):
    _, baddata = mp_icsd_clean(testData, reportBadData=True)

    assert all(item in baddata["other_anion_IDs"] for item in ("mp-5634", "mp-788"))
    assert len(baddata["other_anion_IDs"]) == 2
    assert "mp-5634" not in baddata["other_oxidation_IDs"]
    assert any(
        item["material_id"] not in baddata["valence_problem_IDs"] for item in testData
    )
    assert any(
        item["material_id"] not in baddata["bad_structure_IDs"] for item in testData
    )


## -----------------------------------------------------------------------------


@pytest.fixture
def oxide_check_true():
    return loadfn(os.path.join(root, "test_oxide_check.json.gz"))


## -----------------------------------------------------------------------------


def test_oxide_check(oxide_check_true, testData):
    for Tdatum in testData:
        other_anion, other_oxidation, bad_structure, primStruc = oxide_check(
            Tdatum["structure"]
        )
        assert other_anion == oxide_check_true[Tdatum["material_id"]]["other_anion"]
        assert (
            other_oxidation
            == oxide_check_true[Tdatum["material_id"]]["other_oxidation"]
        )
        assert bad_structure == oxide_check_true[Tdatum["material_id"]]["bad_structure"]
        # TODO: There could be permutations or machine percision difference
        assert np.isclose(
            primStruc.volume,
            oxide_check_true[Tdatum["material_id"]]["primStruc"].volume,
        )


## -----------------------------------------------------------------------------


@pytest.fixture
def env_true():
    return loadfn(os.path.join(root, "test_env.json.gz"))


## -----------------------------------------------------------------------------


def test_analyze_env(env_true, testData):
    for Tdatum in testData:
        sc, oxid_states = analyze_environment(
            Tdatum["structure"],
            "simple",
            [AdditionalConditions.ONLY_ANION_CATION_BONDS],
        )
        assert oxid_states == env_true[Tdatum["material_id"]]["oxid_states"]
        assert (
            sc.as_dict()["connectivity_graph"]
            == env_true[Tdatum["material_id"]]["sc"].as_dict()["connectivity_graph"]
        )


## -----------------------------------------------------------------------------


@pytest.fixture
def features_true_list():
    data = loadfn(os.path.join(root, "test_features.json.gz"))
    return data


## -----------------------------------------------------------------------------


def test_features(features_true_list, testData):
    for Tdatum in testData:
        features_true = features_true_list[Tdatum["material_id"]]
        features_test = CoordinationFeatures.from_structure(Tdatum["structure"])

        # Test sites features
        sites_true = features_true.sites
        sites_test = features_test.sites

        assert (np.array(sites_true.elements) == np.array(sites_test.elements)).all()
        assert (
            np.array(sites_true.oxidations) == np.array(sites_test.oxidations)
        ).all()
        assert (np.array(sites_true.ions) == np.array(sites_test.ions)).all()

        # Test distances
        distances_true = {}
        distances_test = {}
        # Fill dictionaries
        for item in features_true.distances:
            key = (item["site"], item["site_to"])
            distances_true[key] = item["distance"]
        for item in features_test.distances:
            key = (item["site"], item["site_to"])
            distances_test[key] = item["distance"]
        # Compare dictionaries
        assert len(distances_true) == len(distances_test)
        for item in features_test.distances:
            key = (item["site"], item["site_to"])
            assert key in distances_true
            if key in distances_true:
                assert (distances_true[key] - item["distance"]) < 1e-4

        # Test ce neighbors
        ce_neighbors_true = {}
        ce_neighbors_test = {}
        # Fill dictionaries
        for item in features_true.ce_neighbors:
            key = (
                (item["site"], item["site_to"])
                + (item["connectivity"],)
                + tuple(item["ligand_indices"])
                + tuple(np.array(item["angles"]).round(2))
            )
            ce_neighbors_true[key] = item
        for item in features_test.ce_neighbors:
            key = (
                (item["site"], item["site_to"])
                + (item["connectivity"],)
                + tuple(item["ligand_indices"])
                + tuple(np.array(item["angles"]).round(2))
            )
            ce_neighbors_test[key] = item
        # Compare dictionaries
        assert len(ce_neighbors_true) == len(ce_neighbors_test)
        for item in features_test.ce_neighbors:
            key = (
                (item["site"], item["site_to"])
                + (item["connectivity"],)
                + tuple(item["ligand_indices"])
                + tuple(np.array(item["angles"]).round(2))
            )
            assert key in ce_neighbors_true
