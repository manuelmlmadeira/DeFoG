import copy
import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
from concurrent.futures import ThreadPoolExecutor, TimeoutError


from datasets.dataset_utils import load_pickle, save_pickle, to_list
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

import multiprocessing
import time
import threading


class TLSGraphDataset(InMemoryDataset):
    def __init__(
        self,
        stage,
        root,
        target_prop=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        self.stage = stage
        # self.file_idx is used by the init of super class
        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ["train.pkl", "val.pkl", "test.pkl"]

    @property
    def split_file_name(self):
        return ["train.pkl", "val.pkl", "test.pkl"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        low_tls_urls = {
            "train": "https://github.com/manuelmlmadeira/ConStruct/raw/main/data/low_tls_200/raw/train.pkl",
            "val": "https://github.com/manuelmlmadeira/ConStruct/raw/main/data/low_tls_200/raw/val.pkl",
            "test": "https://github.com/manuelmlmadeira/ConStruct/raw/main/data/low_tls_200/raw/test.pkl",
        }
        high_tls_urls = {
            "train": "https://github.com/manuelmlmadeira/ConStruct/raw/main/data/high_tls_200/raw/train.pkl",
            "val": "https://github.com/manuelmlmadeira/ConStruct/raw/main/data/high_tls_200/raw/val.pkl",
            "test": "https://github.com/manuelmlmadeira/ConStruct/raw/main/data/high_tls_200/raw/test.pkl",
        }

        for i, key in enumerate(["train", "val", "test"]):
            low_tls_file_path = download_url(
                low_tls_urls[key], osp.join(self.raw_dir, f"low_tls")
            )
            low_tls_data = load_pickle(low_tls_file_path)
            high_tls_file_path = download_url(
                high_tls_urls[key], osp.join(self.raw_dir, f"high_tls")
            )
            high_tls_data = load_pickle(high_tls_file_path)
            # Merge datasets
            all_data = low_tls_data + high_tls_data
            save_pickle(all_data, self.raw_paths[i])
            print(f"Created {key} dataset (len: {len(all_data)})")

    def process(self):
        raw_dataset = load_pickle(os.path.join(self.raw_dir, f"{self.stage}.pkl"))

        data_list = []
        for idx, graph in enumerate(raw_dataset):
            cell_graph = CellGraph(graph)

            # Check that all cell graphs are either low or high TLS (no ambiguous cell graphs)
            if not (cell_graph.has_low_TLS() or cell_graph.has_high_TLS()):
                raise ValueError(
                    f"Cell graph {idx} is ambiguous in TLS content: does not have low (k1 < 0.05) or high TLS (k2 > 0.05)"
                )

            data = cell_graph.to_torch_geometric()
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data.x = F.one_hot(data.x, num_classes=9).float()
            data.edge_attr = F.one_hot(data.edge_attr, num_classes=2).float()
            data.y = torch.tensor(
                [cell_graph.tls_features[f"k_{a}"] for a in range(6)]
            ).unsqueeze(0)
            data.idx = idx
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class SelectK2Transform:
    def __call__(self, data):
        data.y = (data.y[..., 2] > 0.05).float().unsqueeze(0)
        return data


class TLSDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir

        # For conditional generation
        is_conditional = cfg.general.conditional
        target = cfg.general.target
        if is_conditional:
            if target == "k2":
                transform = SelectK2Transform()
            else:
                raise ValueError(f"Unknown target {target}")

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            "train": TLSGraphDataset(
                transform=transform,
                stage="train",
                root=root_path,
            ),
            "val": TLSGraphDataset(
                transform=transform,
                stage="val",
                root=root_path,
            ),
            "test": TLSGraphDataset(
                transform=transform,
                stage="test",
                root=root_path,
            ),
        }

        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        super().__init__(cfg, datasets)


class TLSInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


PHENOTYPE_DECODER = [
    "B",
    "T",
    "Epithelial",
    "Fibroblast",
    "Myofibroblast",
    "CD38+ Lymphocyte",
    "Macrophages/Granulocytes",
    "Marker",
    "Endothelial",
]

PHENOTYPE_ENCODER = {v: k for k, v in enumerate(PHENOTYPE_DECODER)}


class CellGraph(nx.Graph):
    def __init__(self, graph):
        # nx specific
        super().__init__()
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))

        self.tls_features = self.compute_tls_features()

    def has_low_TLS(self):
        return self.tls_features["k_1"] < 0.05

    def has_high_TLS(self):
        return 0.05 < self.tls_features["k_2"]

    def to_label(self):
        if self.has_low_TLS():
            return 0
        elif self.has_high_TLS():
            return 1
        else:
            return -1

    def to_torch_geometric(self):
        n = self.number_of_nodes()
        n_nodes = n * torch.ones(1, dtype=torch.long)
        phenotypes = [self.nodes[node].get("phenotype") for node in self.nodes()]
        encoded_phenotypes = [PHENOTYPE_ENCODER[phenotype] for phenotype in phenotypes]
        X = torch.tensor(encoded_phenotypes, dtype=torch.long)
        torch_adj = torch.Tensor(
            nx.to_numpy_array(self)
        )  # by default follows same order as G.nodes()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(torch_adj)
        edge_attr = torch.ones(edge_index.shape[-1], dtype=torch.long)  # no edge types

        data = torch_geometric.data.Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_nodes=n_nodes,
        )
        return data

    @classmethod
    def from_torch_geometric(cls, data):
        nx_graph = torch_geometric.utils.to_networkx(
            data, node_attrs=["x"], to_undirected=True
        )

        # collapse the one-hot encoding of the phenotypes
        for node in nx_graph.nodes():
            ohe_encoded_phenotype = nx_graph.nodes[node].pop(
                "x"
            )  # also deletes the key "x"
            enc_pheno = torch.argmax(torch.tensor(ohe_encoded_phenotype)).item()
            nx_graph.nodes[node]["phenotype"] = PHENOTYPE_DECODER[enc_pheno]

        cell_graph = cls(nx_graph)
        return cell_graph

    @classmethod
    def from_dense_graph(cls, dense_graph):

        node_features = (
            dense_graph[0].astype(int)
            if isinstance(dense_graph[0], np.ndarray)
            else dense_graph[0].cpu().numpy().astype(int)
        )
        adj = (
            dense_graph[1].astype(int)
            if isinstance(dense_graph[1], np.ndarray)
            else dense_graph[1].cpu().numpy().astype(int)
        )
        n_nodes = np.sum(node_features != -1)
        node_features = node_features[:n_nodes]
        adj = adj[:n_nodes, :n_nodes]

        nx_graph = nx.from_numpy_array(adj)
        # Delete weight on edges
        for _, _, data in nx_graph.edges(data=True):
            del data["weight"]

        for node_idx in nx_graph.nodes():
            encoded_phenotype = node_features[node_idx]
            phenotype = PHENOTYPE_DECODER[encoded_phenotype]
            nx_graph.nodes[node_idx]["phenotype"] = phenotype

        cell_graph = cls(nx_graph)
        return cell_graph

    # def to_placeholder(self, dataset_infos):
    #     tg_graph = self.to_torch_geometric()
    #     placeholder = to_dense(tg_graph, dataset_infos)
    #     placeholder = placeholder.collapse(collapse_charges=None)
    #     print("is placeholder implemented in flow matching??")
    #     breakpoint()
    #     return placeholder

    # def is_isomorphic(self, other_cg):
    #     return nx.is_isomorphic(
    #         self,
    #         other_cg,
    #         node_match=lambda x, y: x["phenotype"] == y["phenotype"],
    #     )

    def is_isomorphic(self, other_cg):
        import networkx.algorithms.isomorphism as iso

        # Create a copy of the graphs with numerical phenotypes
        self_numerical = CellGraph(self)  # Create a copy of the graph
        other_numerical = CellGraph(other_cg)  # Create a copy of the other graph

        # Manually relabel the "phenotype" node features
        for node, data in self_numerical.nodes(data=True):
            if data["phenotype"] in PHENOTYPE_ENCODER:
                self_numerical.nodes[node]["phenotype"] = PHENOTYPE_ENCODER[
                    data["phenotype"]
                ]

        for node, data in other_numerical.nodes(data=True):
            if data["phenotype"] in PHENOTYPE_ENCODER:
                other_numerical.nodes[node]["phenotype"] = PHENOTYPE_ENCODER[
                    data["phenotype"]
                ]
        node_matching = iso.categorical_node_match("phenotype", None)

        return iso.is_isomorphic(
            self_numerical, other_numerical, node_match=node_matching
        )

    # def is_isomorphic(self, other_cg, timeout=1):
    #     def check_isomorphism():
    #         print("start check_isomorphism")
    #         time.sleep(10)  # Simulate a long-running task
    #         return nx.is_isomorphic(
    #             self,
    #             other_cg,
    #             node_match=lambda x, y: x["phenotype"] == y["phenotype"],
    #         )

    #     def check_isomorphism_numerical():
    #         import networkx.algorithms.isomorphism as iso

    #         # Create a copy of the graphs with numerical phenotypes
    #         self_numerical = CellGraph(self)  # Create a copy of the graph
    #         other_numerical = CellGraph(other_cg)  # Create a copy of the other graph

    #         # Manually relabel the "phenotype" node features
    #         for node, data in self_numerical.nodes(data=True):
    #             if data["phenotype"] in PHENOTYPE_ENCODER:
    #                 self_numerical.nodes[node]["phenotype"] = PHENOTYPE_ENCODER[data["phenotype"]]

    #         for node, data in other_numerical.nodes(data=True):
    #             if data["phenotype"] in PHENOTYPE_ENCODER:
    #                 other_numerical.nodes[node]["phenotype"] = PHENOTYPE_ENCODER[data["phenotype"]]
    #         time.sleep(10)  # Simulate a long-running task
    #         node_matching = iso.categorical_node_match("phenotype", None)
    #         return iso.is_isomorphic(self_numerical, other_numerical, node_match=node_matching)

    #     def run_with_timeout(func, timeout):
    #         result = [None]  # Use a list to allow access in the inner function
    #         stop_event = threading.Event()  # Create an event for signaling thread termination

    #         def target():
    #             # Check periodically if the stop_event has been set
    #             while not stop_event.is_set():
    #                 try:
    #                     result[0] = func()  # Set the result in the list
    #                     break  # Exit the loop if function completes
    #                 except Exception as e:
    #                     # You can handle any exceptions or set stop_event here if needed
    #                     result[0] = None
    #                     break

    #         thread = threading.Thread(target=target)
    #         thread.start()
    #         thread.join(timeout)

    #         if thread.is_alive():
    #             print("Timeout exceeded, signaling the thread to stop.")
    #             stop_event.set()  # Signal the thread to stop

    #             # Optionally, wait a bit for it to clean up
    #             thread.join(5)  # Allow some grace time for the thread to finish after signaling

    #             if thread.is_alive():
    #                 print("Thread did not stop after signaling.")
    #             return None

    #         return result[0]

    #     # First try the original isomorphism check
    #     result = run_with_timeout(check_isomorphism, timeout)
    #     if result is not None:
    #         return result

    #     # If timeout happens, try numerical version
    #     print(f"Timeout after {timeout} seconds in original isomorphism test. Trying numerical version...")
    #     result_numerical = run_with_timeout(check_isomorphism_numerical, timeout)
    #     if result_numerical is not None:
    #         return result_numerical

    #     print(f"Timeout again in numerical isomorphism test. Returning True as fallback.")
    #     return True

    # with ThreadPoolExecutor() as executor:
    #     # First try the original isomorphism check
    #     future = executor.submit(check_isomorphism)
    #     try:
    #         result = future.result(timeout=timeout)
    #         return result  # Return result if successful
    #     except TimeoutError:
    #         print(
    #             f"Timeout after {timeout} seconds in original isomorphism test. Trying numerical version..."
    #         )

    #         # Now try the numerical isomorphism check
    #         future_numerical = executor.submit(check_isomorphism_numerical)
    #         try:
    #             result_numerical = future_numerical.result(timeout=timeout)
    #             return result_numerical  # Return result if successful
    #         except TimeoutError:
    #             print(
    #                 f"Timeout again after {timeout} seconds in numerical isomorphism test. Returning False as fallback."
    #             )
    #             return True  # Return True if both checks fail, get a lower bound on actual performance...

    # def get_phenotypes_list(self):
    #     return [self.nodes[node]["phenotype"] for node in self.nodes()]

    # def get_cell_phenotype_from_idx(self, idx):
    #     return self.nodes[idx]["phenotype"]

    @property
    def map_phenotype_to_color(self):
        return {
            "B": "c",
            "T": "b",
            "Epithelial": "k",
            "Fibroblast": "#C4A484",  # light brown
            "Myofibroblast": "g",
            "CD38+ Lymphocyte": "#FEE12B",  # yellow (towards gold)
            "Macrophages/Granulocytes": "C3",  # "r",
            "Marker": "0.9",  # grey
            "Endothelial": "0.75",  # grey
        }

    def set_pos(self, pos=None):
        if self.get_pos():
            raise ValueError("Positions already set")
        elif pos is None:
            positions = nx.spring_layout(self)
        else:
            positions = pos
        nx.set_node_attributes(self, positions, "pos")

    def get_pos(self):
        return nx.get_node_attributes(self, "pos")

    def plot_graph(
        self,
        has_legend=True,
        node_size=50,
        save_path=None,
        black_border=True,
        no_edges=False,
        fontsize=12,
        verbose=True,
        time=None,
    ):
        node_colors = [
            self.map_phenotype_to_color[phenotype]
            for phenotype in nx.get_node_attributes(self, "phenotype").values()
        ]

        # Positons
        if not self.get_pos():
            self.set_pos()
        positions = self.get_pos()

        # Plot graph
        plt.figure()
        nx.draw_networkx_nodes(
            self,
            # with_labels=False,
            pos=positions,
            node_size=node_size,
            node_color=node_colors,
            # font_weight=font_weight,
            edgecolors="k" if black_border else None,
        )
        if not no_edges:
            nx.draw_networkx_edges(
                self,
                pos=positions,
                edge_color="k",
                width=1,
            )

        # Create a legend
        if has_legend:
            legend_labels = list(self.map_phenotype_to_color.keys())
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=label,
                )
                for label, color in self.map_phenotype_to_color.items()
            ]
            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),
                ncol=len(legend_labels) / 3,
            )

        if time is not None:
            plt.text(
                0.5,
                0.05,  # place below the graph
                f"t = {time:.2f}",
                ha="center",
                va="center",
                transform=plt.gcf().transFigure,
                fontsize=16,
            )

        plt.axis("off")  # delete black square around

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            if verbose:
                print(f"Saved graph plot to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_graph_with_tls_edges(
        self,
        save_path=None,
        has_legend=True,
        fontsize=12,
    ):
        blur_color = "0.9"
        node_colors = {
            node_idx: (
                self.map_phenotype_to_color[phenotype]
                if phenotype in ["B", "T"]
                else "w"
            )
            for node_idx, phenotype in nx.get_node_attributes(self, "phenotype").items()
        }

        nodes_positions = nx.get_node_attributes(self, "pos")
        if not nodes_positions:
            nodes_positions = nx.spring_layout(
                self, seed=0
            )  # needed for reproducibility
        for color in set(node_colors.values()):
            nodes_with_color = [
                node_idx
                for node_idx, node_color in node_colors.items()
                if node_color == color
            ]
            nx.draw_networkx_nodes(
                self,
                pos=nodes_positions,
                nodelist=nodes_with_color,
                node_color=color,
                edgecolors=blur_color if color == "w" else "k",
                node_size=100,
            )

        edge_types = []
        map_edge_type_to_color = {
            "ignore": blur_color,
            "alpha": "k",
            "gamma_0": "#1f77b4",
            "gamma_1": "#ff7f0e",
            "gamma_2": "#2ca02c",
            "gamma_3": "#d62728",
            "gamma_4": "#9467bd",
            "gamma_5": "#8c564b",
        }

        for edge in self.edges:
            edge_types.append(self.classify_TLS_edge(edge))
        edge_colors = [map_edge_type_to_color[edge_type] for edge_type in edge_types]
        edge_widths = [2 if edge_type != "ignore" else 0.5 for edge_type in edge_types]
        h_edges = nx.draw_networkx_edges(
            self,
            pos=nodes_positions,
            edge_color=edge_colors,
            width=edge_widths,
        )

        # # Legend
        from matplotlib.lines import Line2D

        def make_line(clr, **kwargs):
            return Line2D([], [], color=clr, linewidth=2, **kwargs)

        labels = []
        proxies = []
        for edge_type in [
            "alpha",
            "gamma_0",
            "gamma_1",
            "gamma_2",
            "gamma_3",
            "gamma_4",
            "gamma_5",
        ]:
            # get edge with that edge type
            try:
                edge_idx = edge_types.index(edge_type)
                edge_color = edge_colors[edge_idx]
                labels.append("$\\" + edge_type + "$")
                proxies.append(make_line(edge_color))
            except:
                continue  # that type does not exists in this graph

        plt.rcParams.update(
            {
                "text.latex.preamble": r"\renewcommand{\seriesdefault}{b}\boldmath",  # bold text and math in latex
                "text.usetex": True,
                "font.family": "Computer Modern Roman",
                "font.size": fontsize,
            }
        )

        plt.legend(
            handles=proxies,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=4,
        )

        plt.axis("off")  # delete square around

        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved graph plot to {save_path}")
        plt.close()

    def classify_TLS_edge(self, edge):
        allowed_cell_types = ["B", "T"]

        start_node, end_node = edge
        start_phenotype = self.nodes[start_node]["phenotype"]
        end_phenotype = self.nodes[end_node]["phenotype"]

        if (
            start_phenotype not in allowed_cell_types
            or end_phenotype not in allowed_cell_types
        ):
            edge_type = "ignore"

        elif start_phenotype == end_phenotype:
            edge_type = "alpha"
        else:
            b_cell = start_node if start_phenotype == "B" else end_node
            num_of_b_neighbors = len(
                [
                    node
                    for node in self.neighbors(b_cell)
                    if self.nodes[node]["phenotype"] == "B"
                ]
            )
            edge_type = f"gamma_{num_of_b_neighbors}"

        return edge_type

    def compute_tls_features(
        self,
        a_max: int = 5,
        min_num_gamma_edges: int = 0,
        verbose: bool = True,
    ):
        """Compute TLS feature metric from https://arxiv.org/pdf/2310.06661.pdf.

        Args:
            graph (nx.Graph): Graph to compute the TLS features. Should have a "phenotype" attribute with some nodes labeled as "B" and "T".
            a_max (int, optional): Maximum a of k(a) to consider. Defaults to 5.
            min_num_gamma_edges (int, optional): Minimum number of gamma edges to consider the

        Returns:
            dict: Dictionary with the TLO features.
        """

        if self.is_directed():
            raise ValueError("Graph should be undirected.")

        graph_phenotypes = nx.get_node_attributes(self, "phenotype")
        nodes_to_remove = [
            node
            for node, phenotype in graph_phenotypes.items()
            if phenotype != "B" and phenotype != "T"
        ]
        bt_subgraph = copy.deepcopy(self)
        bt_subgraph.remove_nodes_from(nodes_to_remove)
        total_num_edges = bt_subgraph.number_of_edges()

        # Get alpha and gamma edges count
        num_edge_types_idxs = self.get_edges_idxs_by_tlo_type(a_max)

        # Compute TLO features
        denominator = total_num_edges - num_edge_types_idxs["alpha"]
        # If the number of gamma edges are too few, then the feature estimation
        # is unreliable.
        if denominator < min_num_gamma_edges:
            tlo_dict = {f"k_{a}": None for a in range(a_max + 1)}
            if verbose:
                print("WARNING: too few gamma edges. TLO feature set to -1.")
        else:
            tlo_dict = {}
            if denominator == 0:
                tlo_dict.update({f"k_{a}": 0.0 for a in range(a_max + 1)})
            else:
                k = 1.0
                for a in range(a_max + 1):
                    k -= num_edge_types_idxs[f"gamma_{a}"] / denominator
                    # Due to precision errors, sometimes the k computed is negative. To avoid
                    # this, we clip it to 0 in those cases. The rounding is performed to 4
                    # decimal cases, as there is no meaning on the remaining decimal cases.
                    tlo_dict.update({f"k_{a}": max(0.0, round(k, 4))})

        return tlo_dict

    def get_edges_idxs_by_tlo_type(self, a_max: int):
        """Provided a graph, count

        Args:
            graph (nx.Graph): _description_
            a_max (int): _description_

        Returns:
            _type_: _description_
        """

        num_edges_by_type = {f"gamma_{a}": 0 for a in range(a_max + 1)}
        num_edges_by_type["alpha"] = 0

        for edge in self.edges:
            edge_type = self.classify_TLS_edge(edge)
            if edge_type in num_edges_by_type:
                num_edges_by_type[edge_type] += 1

        return num_edges_by_type
