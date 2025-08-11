## Copyright (C) 2023 Sasan Amariamir, Philipp Benner

import numpy as np
import requests
import json

from monty.serialization import loadfn, dumpfn

from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions

from coordinationnet import CoordinationFeatures
from coordinationnet.features_featurizer import analyze_environment
from coordinationnet.features_utility import oxide_check

## -----------------------------------------------------------------------------

MPID = "Q0tUKnAE52sy7hVO"

## -----------------------------------------------------------------------------

testIDs = [
    "mp-12236",
    "mp-1788",
    "mp-19359",
    "mp-19418",
    "mp-2605",
    "mp-306",
    "mp-4056",
    "mp-4930",
    "mp-5634",
    "mp-560167",
    "mp-5986",
    "mp-6564",
    "mp-7000",
    "mp-788",
    "mp-886",
    "mp-1143",
    "mp-7566",
]

## -----------------------------------------------------------------------------


def get_version():
    response = requests.get(
        "https://www.materialsproject.org/rest/v2/materials/mp-1234/vasp",
        {"API_KEY": MPID},
    )
    response_data = json.loads(response.text)
    return response_data.get("version")


## -----------------------------------------------------------------------------


def get_data():
    testData = []
    with MPRester(MPID) as m:
        for i, testID in enumerate(testIDs):
            testMat = dict()
            if i < 5:
                Tstructure = m.get_structure_by_material_id(
                    testID, conventional_unit_cell=True
                )
            else:
                Tstructure = m.get_structure_by_material_id(
                    testID, conventional_unit_cell=False
                )
            testMat["material_id"] = testID
            testMat["structure"] = Tstructure
            testMat["formula"] = Tstructure.formula
            testData.append(testMat)

    return np.array(testData)


## -----------------------------------------------------------------------------


def load_data():
    testData = loadfn("test_data.json.gz")
    for Tdatum in testData:
        Tdatum["structure"] = Structure.from_dict(Tdatum["structure"])
    return testData


## -----------------------------------------------------------------------------


def groundtruth_oxide_check(testData):
    # This is used to calculate and save the actual results of the tests. Not to be included in the test script.
    oxide_check_dict = dict()
    for Tdatum in testData:
        other_anion, other_oxidation, bad_structure, primStruc = oxide_check(
            Tdatum["structure"]
        )
        oxide_check_dict[Tdatum["material_id"]] = dict(
            (
                ("other_anion", other_anion),
                ("other_oxidation", other_oxidation),
                ("bad_structure", bad_structure),
                ("primStruc", primStruc),
            )
        )

    dumpfn(oxide_check_dict, "test_oxide_check.json.gz")


## -----------------------------------------------------------------------------


def groundtruth_env(testData):
    # This is used to calculate and save the actual results of the tests. Not to be included in the test script.
    analyze_env_dict = dict()
    for Tdatum in testData:
        structure_connectivity, oxid_states = analyze_environment(
            Tdatum["structure"],
            "simple",
            [AdditionalConditions.ONLY_ANION_CATION_BONDS],
        )
        analyze_env_dict[Tdatum["material_id"]] = dict(
            [("oxid_states", oxid_states), ("sc", structure_connectivity)]
        )

    dumpfn(analyze_env_dict, "test_env.json.gz")


## -----------------------------------------------------------------------------


def groundtruth_features(testData):
    # This is used to calculate and save the actual results of the tests. Not to be included in the test script.
    crysFeaturizer_dict = dict()
    for Tdatum in testData:
        crysFeaturizer_dict[Tdatum["material_id"]] = (
            CoordinationFeatures.from_structure(Tdatum["structure"])
        )

    dumpfn(crysFeaturizer_dict, "test_features.json.gz")


## -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Database version: {get_version()}")
    dumpfn(get_data(), "test_data.json.gz")
    testData = load_data()
    groundtruth_oxide_check(testData)
    groundtruth_env(testData)
    groundtruth_features(testData)
