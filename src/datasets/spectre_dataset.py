import os
import pathlib
import pickle as pkl
import zipfile

from networkx import to_numpy_array

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class SpectreGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.sbm_file = "sbm_200.pt"
        self.planar_file = "planar_64_200.pt"
        self.comm20_file = "community_12_21_100.pt"
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_graphs = len(self.data.n_nodes)

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        return [self.split + ".pt"]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/sbm.pkl"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/planar.pkl"
        elif self.dataset_name == "tree":
            raw_url = "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/tree.pkl"
        elif self.dataset_name == "comm20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        elif self.dataset_name == "ego":
            raw_url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
        elif self.dataset_name == "imdb":
            raw_url = "https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip"
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name in ["tree", "sbm", "planar"]:
            with open(file_path, "rb") as f:
                dataset = pkl.load(f)
            train_data = dataset["train"]
            val_data = dataset["val"]
            test_data = dataset["test"]

            train_data = [
                torch.Tensor(to_numpy_array(graph)).fill_diagonal_(0)
                for graph in train_data
            ]
            val_data = [
                torch.Tensor(to_numpy_array(graph)).fill_diagonal_(0)
                for graph in val_data
            ]
            test_data = [
                torch.Tensor(to_numpy_array(graph)).fill_diagonal_(0)
                for graph in test_data
            ]
        else:
            if self.dataset_name == "ego":
                networks = pkl.load(open(file_path, "rb"))
                adjs = [
                    torch.Tensor(to_numpy_array(network)).fill_diagonal_(0)
                    for network in networks
                ]
            elif self.dataset_name == "imdb":
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(file_path))

                # Step 1: Read edge_index from file
                index_path = os.path.join(
                    os.path.dirname(file_path), "IMDB-BINARY", "IMDB-BINARY_A.txt"
                )
                edge_index = []
                with open(index_path, "r") as file:
                    for line in file:
                        int1, int2 = map(int, line.strip().split(","))
                        edge_index.append([int1, int2])
                edge_index = torch.tensor(edge_index).t().contiguous() - 1

                # Step 2: Read graph_indicator from file
                index_path = os.path.join(
                    os.path.dirname(file_path),
                    "IMDB-BINARY",
                    "IMDB-BINARY_graph_indicator.txt",
                )
                graph_indicator = []
                with open(index_path, "r") as file:
                    for line in file:
                        num = int(line.strip())
                        graph_indicator.append(num)
                graph_indicator = torch.tensor(graph_indicator) - 1

                # Step 3: Create individual graphs based on graph_indicator
                num_graphs = graph_indicator.max().item() + 1
                adjs = []
                for i in range(num_graphs):
                    node_mask = graph_indicator == i
                    n_node = node_mask.sum().item()
                    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                    edges = edge_index[:, edge_mask]
                    ptr = torch.where(node_mask)[0][0]
                    edges -= ptr
                    adj = torch.zeros(n_node, n_node)
                    adj[edges[0], edges[1]] = 1
                    adj[edges[1], edges[0]] = 1
                    adjs.append(adj)
            else:
                (
                    adjs,
                    eigvals,
                    eigvecs,
                    n_nodes,
                    max_eigval,
                    min_eigval,
                    same_sample,
                    n_max,
                ) = torch.load(file_path)

            g_cpu = torch.Generator().manual_seed(1234)
            self.num_graphs = 200
            if self.dataset_name in ["ego", "protein"]:
                self.num_graphs = len(adjs)
            elif self.dataset_name == "imdb":
                self.num_graphs = graph_indicator.max().item() + 1

            if self.dataset_name == "ego":
                test_len = int(round(self.num_graphs * 0.2))
                train_len = int(round(self.num_graphs * 0.8))
                val_len = int(round(self.num_graphs * 0.2))
                indices = torch.randperm(self.num_graphs, generator=g_cpu)
                print(
                    f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}"
                )
                train_indices = indices[:train_len]
                val_indices = indices[:val_len]
                test_indices = indices[train_len:]
            else:
                test_len = int(round(self.num_graphs * 0.2))
                train_len = int(round((self.num_graphs - test_len) * 0.8))
                val_len = self.num_graphs - train_len - test_len
                indices = torch.randperm(self.num_graphs, generator=g_cpu)
                print(
                    f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}"
                )
                train_indices = indices[:train_len]
                val_indices = indices[train_len : train_len + val_len]
                test_indices = indices[train_len + val_len :]

            train_data = []
            val_data = []
            test_data = []

            for i, adj in enumerate(adjs):
                if i in train_indices:
                    train_data.append(adj)
                if i in val_indices:
                    val_data.append(adj)
                if i in test_indices:
                    test_data.append(adj)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {"train": 0, "val": 1, "test": 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            "train": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name, split="train", root=root_path
            ),
            "val": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name, split="val", root=root_path
            ),
            "test": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name, split="test", root=root_path
            ),
        }

        train_len = len(datasets["train"].data.n_nodes)
        val_len = len(datasets["val"].data.n_nodes)
        test_len = len(datasets["test"].data.n_nodes)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.dataset_name = datamodule.inner.dataset_name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
