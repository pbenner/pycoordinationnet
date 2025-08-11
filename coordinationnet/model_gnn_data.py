## Copyright (C) 2023 Philipp Benner

import math
import torch

from tqdm import tqdm

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import HeteroData
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.loader import DataLoader as GraphDataLoader

from .features_coding import NumOxidations, NumGeometries
from .features_datatypes import CoordinationFeatures

from .model_data import GenericDataset, Batch

## ----------------------------------------------------------------------------


def code_csms(csms) -> list[float]:
    # According to specs, CSM is a value between 0 and 100
    return torch.tensor(csms, dtype=torch.float) / 100


def code_distance(distance: float, l: int) -> torch.Tensor:
    # Sites `from` and `to` get distance assigned, all ligands
    # get inf
    x = torch.tensor(l * [distance], dtype=torch.float) / 8.0
    return x


def code_angles(angles: list[float]) -> torch.Tensor:
    # Ligands get angle information, sites `from` and `to`
    # get inf
    x = torch.tensor(angles, dtype=torch.float) / 180
    return x


def get_distance(features, site, site_to):
    for item in features.distances:
        if item["site"] == site and item["site_to"] == site_to:
            return item["distance"]
        if item["site_to"] == site and item["site"] == site_to:
            return item["distance"]
    raise RuntimeError("Distance not available")


## ----------------------------------------------------------------------------


class GraphCoordinationData(GenericDataset):
    def __init__(self, dataset, verbose=False) -> None:
        X = [item[0] for item in dataset]
        y = [item[1] for item in dataset]

        X = self.__compute_graphs__(X, verbose=verbose)

        super().__init__(X, y)

    @classmethod
    def __compute_graph_sites__(
        cls, features: CoordinationFeatures, data: HeteroData
    ) -> None:
        # Number of nodes in this graph
        nsites = len(features.sites.elements)
        # Explicitly set the number of nodes
        data["site"].num_nodes = nsites
        # Initialize graph with isolated nodes for each site
        data["site"].x = {
            "elements": torch.tensor(features.sites.elements, dtype=torch.long),
            "oxidations": torch.tensor(features.sites.oxidations, dtype=torch.long),
        }
        # Global edge index, initialize with self-connections for
        # isolated nodes
        data["site", "*", "site"].edge_index = torch.tensor(
            [[i for i in range(nsites)], [i for i in range(nsites)]], dtype=torch.long
        )

    @classmethod
    def __compute_graph_ce_pairs__(
        cls, features: CoordinationFeatures, data: HeteroData
    ) -> None:
        # Number of nodes in this graph
        nsites = len(features.sites.elements)
        # Get CE symbols and CSMs
        site_ces = nsites * [NumGeometries]
        site_csm = nsites * [math.inf]
        # Each site may have multiple CEs, but in almost all cases a site fits only one CE.
        # Some sites (anions) do not have any site information, where we use the value
        # `NumGeometries`
        for ce in features.ces:
            # Get site index
            j = ce["site"]
            # Consider only the first CE symbol
            site_ces[j] = ce["ce_symbols"][0]
            site_csm[j] = ce["csms"][0]
        # Initial node features
        x_ce = {
            "elements": torch.tensor([], dtype=torch.long),
            "oxidations": torch.tensor([], dtype=torch.long),
            "geometries": torch.tensor([], dtype=torch.long),
            "csms": torch.tensor([], dtype=torch.float),
        }
        x_ligand = {
            "elements": torch.tensor([], dtype=torch.long),
            "oxidations": torch.tensor([], dtype=torch.long),
            "distances": torch.tensor([], dtype=torch.float),
            "angles": torch.tensor([], dtype=torch.float),
        }
        # Edges
        edge_index_1 = [[], []]
        edge_index_2 = [[], []]
        edge_index_3 = [[], []]
        # Edge features
        edge_attr = []
        # Global node index i
        i1 = 0
        i2 = 0
        # Construct CE graphs
        for nb in features.ce_neighbors:
            l = len(nb["ligand_indices"])
            if l > 0:
                # Get site indices
                idx_ce = [nb["site"], nb["site_to"]]
                idx_ligand = nb["ligand_indices"]
                # Construct CE features
                x_ce["elements"] = torch.cat(
                    (
                        x_ce["elements"],
                        torch.tensor(
                            [features.sites.elements[site] for site in idx_ce],
                            dtype=torch.long,
                        ),
                    )
                )
                x_ce["oxidations"] = torch.cat(
                    (
                        x_ce["oxidations"],
                        torch.tensor(
                            [features.sites.oxidations[site] for site in idx_ce],
                            dtype=torch.long,
                        ),
                    )
                )
                x_ce["geometries"] = torch.cat(
                    (
                        x_ce["geometries"],
                        torch.tensor(
                            [site_ces[site] for site in idx_ce], dtype=torch.long
                        ),
                    )
                )
                x_ce["csms"] = torch.cat(
                    (x_ce["csms"], code_csms([site_csm[site] for site in idx_ce]))
                )
                # Construct ligand features
                x_ligand["elements"] = torch.cat(
                    (
                        x_ligand["elements"],
                        torch.tensor(
                            [features.sites.elements[site] for site in idx_ligand],
                            dtype=torch.long,
                        ),
                    )
                )
                x_ligand["oxidations"] = torch.cat(
                    (
                        x_ligand["oxidations"],
                        torch.tensor(
                            [features.sites.oxidations[site] for site in idx_ligand],
                            dtype=torch.long,
                        ),
                    )
                )
                x_ligand["distances"] = torch.cat(
                    (x_ligand["distances"], code_distance(nb["distance"], l))
                )
                x_ligand["angles"] = torch.cat(
                    (x_ligand["angles"], code_angles(nb["angles"]))
                )

                for j, k in enumerate(nb["ligand_indices"]):
                    # From ligand     ; To CE
                    edge_index_1[0].append(i2 + j)
                    edge_index_1[1].append(i1 + 0)
                    edge_index_1[0].append(i2 + j)
                    edge_index_1[1].append(i1 + 1)
                    # From CE         ; To ligand
                    edge_index_2[0].append(i1 + 0)
                    edge_index_2[1].append(i2 + j)
                    edge_index_2[0].append(i1 + 1)
                    edge_index_2[1].append(i2 + j)
                    # ligand-CE features
                    edge_attr.append(get_distance(features, idx_ce[0], k))
                    edge_attr.append(get_distance(features, idx_ce[1], k))

                # Connect CE nodes to site nodes
                edge_index_3[0].append(i1 + 0)
                edge_index_3[1].append(nb["site"])
                edge_index_3[0].append(i1 + 1)
                edge_index_3[1].append(nb["site_to"])

                i1 += 2
                i2 += len(nb["ligand_indices"])

        # Explicitly set the number of nodes
        data["ce"].num_nodes = i1
        data["ligand"].num_nodes = i2
        # Assign features
        data["ce"].x = x_ce
        data["ligand"].x = x_ligand
        # Assign edges
        data["ligand", "*", "ce"].edge_index = torch.tensor(
            edge_index_1, dtype=torch.long
        )
        data["ce", "*", "ligand"].edge_index = torch.tensor(
            edge_index_2, dtype=torch.long
        )
        # Assign edge features
        data["ligand", "*", "ce"].edge_attr = torch.tensor(edge_attr)
        data["ce", "*", "ligand"].edge_attr = torch.tensor(edge_attr)
        # Connect CE nodes to site nodes
        data["ce", "*", "site"].edge_index = torch.tensor(
            edge_index_3, dtype=torch.long
        )

    @classmethod
    def __compute_graph__(cls, features: CoordinationFeatures) -> GraphData:
        # Initialize empty heterogeneous graph
        data = HeteroData()
        # Add individual sites
        cls.__compute_graph_sites__(features, data)
        # Add CE pairs
        cls.__compute_graph_ce_pairs__(features, data)

        return data

    @classmethod
    def __compute_graphs__(
        cls, cofe_list: list[CoordinationFeatures], verbose=False
    ) -> list[GraphData]:
        r = len(cofe_list) * [None]
        for i, features in tqdm(
            enumerate(cofe_list),
            desc="Computing graphs",
            disable=(not verbose),
            total=len(cofe_list),
        ):
            r[i] = cls.__compute_graph__(features)

        return r


## ----------------------------------------------------------------------------


class GraphCoordinationFeaturesLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs) -> None:
        if "collate_fn" in kwargs:
            raise TypeError(
                f"{self.__class__}.__init__() got an unexpected keyword argument 'collate_fn'"
            )

        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]

        return GraphBatch.from_data_list(x), torch.utils.data.default_collate(y)
