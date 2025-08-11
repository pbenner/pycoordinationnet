## Copyright (C) 2023 Janine George, Sasan Amariamir, Philipp Benner

# This supresses warnings.
import warnings

warnings.filterwarnings("ignore")

## -----------------------------------------------------------------------------

import numpy as np
import requests
import json

from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core.structure import Structure

## -----------------------------------------------------------------------------


def oxide_check(initStruc: Structure) -> tuple[bool, bool, bool, Structure]:
    """
    Checks the oxidation states of the atoms in primary cell. Retruns the primary structure, and
    booleans for whether the structure is bad, whether anions other than Oxygen are present and
    whether the oxidatoin number of Oxygen is anything other tha -2.
    Parameters:
    ----------------
    initStruc : Structure
        The Structure object as it was queried from MP database.
    """

    # In case the conversion to primitive doesn't work, the type remains consistent when function returns
    primStruc = initStruc
    bad_structure = False
    other_anion = False
    other_oxidation = False

    try:
        primStruc = initStruc.get_primitive_structure()

    except ValueError:
        # True indicates there was an error in calculating the primitive structure.
        bad_structure = True
        return other_anion, other_oxidation, bad_structure, primStruc

    # This class implements a maximum a posteriori (MAP) estimation method to
    # determine oxidation states in a structure.
    bv = BVAnalyzer()
    oxid_stats = bv.get_valences(primStruc)

    # Check for any element other than oxygen with a negative oxidation state.
    for i, site in enumerate(primStruc):
        if site.species_string != "O":
            if oxid_stats[i] < 0:
                other_anion = True
                return other_anion, other_oxidation, bad_structure, primStruc
        # Checks for any oxygen with an oxidation state other than '-2'
        else:
            if oxid_stats[i] != -2:
                other_oxidation = True
                return other_anion, other_oxidation, bad_structure, primStruc

    return other_anion, other_oxidation, bad_structure, primStruc


## -----------------------------------------------------------------------------


def mp_icsd_query(
    MPID: str,
    experimental_data=True,
    properties=["formation_energy_per_atom", "e_above_hull"],
) -> str:
    """
    Retrieves experimental (ICSD) crystallographic data through the Materials Project API.
    Currently queries for crystals which contain Oxide anions, are not theoretical, and have at least 2 elements.
    It stores their (pretty) formula, structure object and material_id.
    Parameters:
    ----------------
    MPID : string
        The user ID from Materials Project which allows data query.
    """
    # Request below is just made to print the version of the database and pymatgen
    response = requests.get(
        "https://www.materialsproject.org/rest/v2/materials/mp-1234/vasp",
        {"API_KEY": MPID},
    )
    response_data = json.loads(response.text)
    print(response_data.get("version"))
    # Request above is just made to print the version of the database and pymatgen

    if experimental_data:
        criteria = {
            "icsd_ids": {"$ne": []},  # Allows data with existing 'icsd_ids' tag
            "theoretical": {
                "$ne": experimental_data
            },  # Allows data without the 'theoretical' tag
            "elements": {"$all": ["O"]},  # Allows for crystals with Oxygen present
            "oxide_type": {"$all": ["oxide"]},  # Allows for oxides (e.g. not peroxide)
            "nelements": {"$gte": 2},  # Allows crystals with at least 2 elements
            "nsites": {"$lte": 12},
        }
    else:
        criteria = {
            # We don't need this limit for theoretical data 'icsd_ids': {'$ne': []}, #allows data with existing 'icsd_ids' tag
            "elements": {"$all": ["O"]},  # allows for crystals with Oxygen present
            "oxide_type": {"$all": ["oxide"]},  # allows for oxides (e.g. not peroxide)
            "nelements": {"$gte": 2},  # allows crystals with at least 2 elements
            "nsites": {"$lte": 12},
        }

    with MPRester(api_key=MPID) as mpr:
        data = mpr.query(
            criteria,
            properties=[
                "material_id",
                "exp.tags",
                "icsd_ids",
                "formula",
                "pretty_formula",
                "full_formula",
                "structure",
                "theoretical",
            ]
            + properties,
        )

    # Converts list to array, much faster to work with
    data = np.array(data)

    return data


## -----------------------------------------------------------------------------


def mp_icsd_clean(arrdata, reportBadData: bool = False) -> str:
    """
    Filters undesired data from the stored experimental data.
    Undesired data here include:
    1- structures which cannot be converted to primitive cell.
    2- data the oxidation states of which cannot be analyzed.
    3- Include any anion element which is not Oxygen.
    4- Include Oxygen with any oxidation number other than -2.
    Parameters:
    ----------------
    queried_data_string : string
        The address of the downloaded experimental data.
    Location : string
        The directory in which the cleaned data will be stored.
    reportBadData : bool
        Returns the four lists of undesired data which is removed during this cleaning. Useful for testing.
    """

    print("The initial data length is", len(arrdata))

    other_anion_IDs = []
    other_oxidation_IDs = []
    valence_problem_IDs = []
    bad_structure_IDs = []

    for j, datum in enumerate(arrdata):
        try:
            other_anion, other_oxidation, bad_structure, primStruc = oxide_check(
                initStruc=datum["structure"]
            )

            if other_anion:
                other_anion_IDs.append([j, datum["material_id"]])
            if other_oxidation:
                other_oxidation_IDs.append([j, datum["material_id"]])

            if bad_structure:
                bad_structure_IDs.append([j, datum["material_id"]])
            else:
                datum["structure"] = primStruc

        except ValueError:
            valence_problem_IDs.append([j, datum["material_id"]])

    print(
        "The number of entries with anions other than Oxygen were", len(other_anion_IDs)
    )

    print(
        "The number of entries with different oxidation types were",
        len(other_oxidation_IDs),
    )

    print(
        "The number of entries where valence/oxidation could not be analyzed were",
        len(valence_problem_IDs),
    )

    print(
        "The number of entries where the primitive structure could not be calculated were",
        len(bad_structure_IDs),
    )

    anion_ind = [i[0] for i in other_anion_IDs]
    oxid_ind = [i[0] for i in other_oxidation_IDs]
    valence_ind = [i[0] for i in valence_problem_IDs]
    structure_ind = [i[0] for i in bad_structure_IDs]

    arrdata = np.delete(arrdata, [*anion_ind, *oxid_ind, *valence_ind, *structure_ind])

    print("The length of data after removing undesired entries is", len(arrdata))

    if not reportBadData:
        return arrdata
    else:
        baddata = {
            "other_anion_IDs": [i[1] for i in other_anion_IDs],
            "other_oxidation_IDs": [i[1] for i in other_oxidation_IDs],
            "valence_problem_IDs": [i[1] for i in valence_problem_IDs],
            "bad_structure_IDs": [i[1] for i in bad_structure_IDs],
        }

        return arrdata, baddata
