## Copyright (C) 2023 Janine George, Sasan Amariamir, Philipp Benner

# This supresses warnings.
import warnings

warnings.filterwarnings("ignore")

## -----------------------------------------------------------------------------

import sys
import os

from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import (
    ConnectivityFinder,
)
from pymatgen.analysis.chemenv.connectivity.structure_connectivity import (
    StructureConnectivity,
)
from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.util.coord import get_angle

## -----------------------------------------------------------------------------


def analyze_environment(
    structure: Structure, env_strategy: str, additional_conditions
) -> tuple[StructureConnectivity, list[int]]:
    """
    Analyzes the coordination environments and returns the StructureConnectivity object for the crystal and the list of oxidation states.
    First, BVAnalyzer() calculates the oxidation states. Then, the LocalGeometryFinder() computes the structure_environment object,
    from which the LightStructureEnvironment (LSE) is derived. Finally, The ConnectivityFinder() builds the StructureConnectivity (SE) based on LSE.
    At the end only the SE is returned, as it includes the LSE object as an attribute.

    Args:
        struc (Structure):
            crystal Structure object from pymatgen
        mystrategy (string):
                The simple or combined strategy for calculating the coordination environments
    """
    if env_strategy == "simple":
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
    else:
        strategy = env_strategy

    # The BVAnalyzer class implements a maximum a posteriori (MAP) estimation method to determine oxidation states in a structure.
    # TODO:
    # Extend similar to emmet:
    # https://github.com/materialsproject/emmet/blob/1a185027d017475e6112164df50428a0b06406c8/emmet-core/emmet/core/oxidation_states.py#L71
    bv = BVAnalyzer()
    oxid_states = bv.get_valences(structure)

    # Backup current stdout
    old_stdout = sys.stdout
    # Avoid printing to the console
    sys.stdout = open(os.devnull, "w")
    # Print a long stroy every time it is initiated
    lgf = LocalGeometryFinder()
    # Reset old stdout
    sys.stdout = old_stdout

    lgf.setup_structure(structure=structure)

    # Get the StructureEnvironments
    se = lgf.compute_structure_environments(
        only_cations=True,
        valences=oxid_states,
        additional_conditions=additional_conditions,
    )

    # Get LightStructureEnvironments
    lse = LightStructureEnvironments.from_structure_environments(
        strategy=strategy, structure_environments=se
    )

    # Get StructureConnectivity object
    cf = ConnectivityFinder()
    sc = cf.get_structure_connectivity(light_structure_environments=lse)

    return sc, oxid_states


## -----------------------------------------------------------------------------


def compute_features_first_degree(
    structure_connectivity: StructureConnectivity,
    oxidation_list: list[int],
    result: "CoordinationFeatures",
) -> "CoordinationFeatures":
    """
    Calculates the desired primary features (related to the atom and nearest neighbors) based on SC object,
    returns them as a dictionary. These features are stored for each atom, under their structure index.
    Features Include: Oxidation number, type of ion, element, coordination for all atoms.
    Cation specific features are the local(coordination) env and nearest neighbor elements & distances.

    Args:
        structure_connectivity (StructureConnectivity):
            The connectivity structure of the material
        oxidation_list (list[int]):
            A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.

    Returns:
        A dictionary with first degree features
    """

    structure = structure_connectivity.light_structure_environments.structure
    # Take lightStructureEnvironment Obj from StructureConnecivity Obj
    lse = structure_connectivity.light_structure_environments
    # Take coordination/local environments from lightStructureEnvironment Obj
    ce_list = lse.coordination_environments

    for atomIndex, atom in enumerate(lse.neighbors_sets):
        if atom == None:
            result.sites.add_item(
                atomIndex,
                oxidation_list[atomIndex],
                "anion",
                structure[atomIndex].species_string,
                structure[atomIndex].coords,
            )
            # Skip further featurization. We're not analyzing envs with anions
            continue

        result.sites.add_item(
            atomIndex,
            oxidation_list[atomIndex],
            "cation",
            structure[atomIndex].species_string,
            structure[atomIndex].coords,
        )
        result.ces.add_item(atomIndex, ce_list[atomIndex])

        for nb in atom[0].neighb_sites_and_indices:
            # Pymatgen bug-fix (PeriodicNeighbor cannot be serialized, need to convert to PeriodicSite)
            # (fixed with 0eb1e3d72fd894b7ba39a5129fbd8b18aedf4b46)
            # site     = PeriodicSite.from_dict(nb['site'].as_dict())
            site = nb["site"]
            distance = site.distance_from_point(structure[atomIndex].coords)
            result.distances.add_item(atomIndex, nb["index"], distance)

    return result


## -----------------------------------------------------------------------------


def compute_features_nnn(
    structure_connectivity: StructureConnectivity, result: "CoordinationFeatures"
) -> "CoordinationFeatures":
    """
    Calculates the desired NNN (next nearest neighbors) features based on SC object,
    and adds them to a dictionary (of primary features). These features are stored
    for each atom, under their structure index. NNN features Include: Polhedral neighbor
    elements, distances, connectivity angles & types.

    Args:
        structure_connectivity (StructureConnectivity):
            The connectivity structure of the material
        structure_data (dict):
            A dictionary containing primary features of the crystal. The NNN features
            will be added under the same atom index.

    Returns:
        A dictionary with next nearest neighbor features added to the structure_data
        object
    """

    structure = structure_connectivity.light_structure_environments.structure
    nodes = structure_connectivity.environment_subgraph().nodes()

    # Loop over all sites in the structure
    for node in nodes:
        for edge in structure_connectivity.environment_subgraph().edges(
            node, data=True
        ):
            # Get site indices for which the distance is computed
            if node.isite == edge[2]["start"]:
                site = edge[2]["start"]
                site_to = edge[2]["end"]
            else:
                site = edge[2]["end"]
                site_to = edge[2]["start"]

            # Compute distance
            distance = structure[edge[2]["start"]].distance(
                structure[edge[2]["end"]], edge[2]["delta"]
            )

            # Angles calculation
            ligands = edge[2]["ligands"]

            # Determine the type of connectivity from the number of ligands
            if len(ligands) == 0:
                connectivity = "isolated"
            elif len(ligands) == 1:
                connectivity = "corner"
            elif len(ligands) == 2:
                connectivity = "edge"
            else:
                connectivity = "face"

            angles = []
            ligand_indices = []
            # For each ligand compute the angle to another coordination environment (central atom)
            for ligand in ligands:
                # The ligand item contains a path one central atom (cation) to another central atom
                # along a single ligand (anions). The `start` item always points to a central atom,
                # while the end will be the ligand.

                # We consider two connecting atoms of the ligand. Get the coordinates of all three
                # sites
                pos0 = structure[ligand[1]["start"]].frac_coords
                pos1 = structure[ligand[1]["end"]].frac_coords + ligand[1]["delta"]
                pos2 = structure[ligand[2]["start"]].frac_coords
                pos3 = structure[ligand[2]["end"]].frac_coords + ligand[2]["delta"]

                cart_pos0 = structure.lattice.get_cartesian_coords(pos0)
                cart_pos1 = structure.lattice.get_cartesian_coords(pos1)
                cart_pos2 = structure.lattice.get_cartesian_coords(pos2)
                cart_pos3 = structure.lattice.get_cartesian_coords(pos3)

                # Measure the angle at the ligand
                angle = get_angle(
                    cart_pos0 - cart_pos1, cart_pos2 - cart_pos3, units="degrees"
                )

                if ligand[1]["start"] == node.isite:
                    assert site == ligand[1]["start"]
                    assert site_to == ligand[2]["start"]
                elif ligand[2]["start"] == node.isite:
                    assert site == ligand[2]["start"]
                    assert site_to == ligand[1]["start"]
                else:
                    raise ValueError(f"Ligand is not connected to center atom")

                assert ligand[0] == ligand[1]["end"]
                assert ligand[0] == ligand[2]["end"]

                angles.append(angle)
                ligand_indices.append(ligand[0])

            result.ce_neighbors.add_item(
                site, site_to, distance, connectivity, ligand_indices, angles
            )

    return result
