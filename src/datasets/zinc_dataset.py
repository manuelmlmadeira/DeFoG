import os
import os.path as osp
import pathlib
from typing import Any, Sequence, Callable, List, Optional

import torch
import torch.nn.functional as F
# from torch_geometric.io import fs

import pickle
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return torch.zeros((1, 0), dtype=torch.float)
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return data.y[..., 3].unsqueeze(1)
        data.y = data.y[..., 3].unsqueeze(1)
        return data


class SelectHOMOTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return data.y[..., 5].unsqueeze(1)
        data.y = data.y[..., 5].unsqueeze(1)
        return data


class SelectBothTransform:
    def __call__(self, data, return_y=False):
        if return_y:
            return torch.hstack([data.y[..., 3], data.y[..., 5]]).unsqueeze(0)
        data.y = torch.hstack([data.y[..., 3], data.y[..., 5]]).unsqueeze(0)
        return data


class ZINCDataset(InMemoryDataset):
    csv_url = "https://raw.githubusercontent.com/harryjo97/GruM/master/GruM_2D/data/zinc250k.csv"
    val_index_url = "https://raw.githubusercontent.com/harryjo97/GruM/master/GruM_2D/data/valid_idx_zinc250k.json"
    
    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')
    
    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = 'train',
        remove_h: bool = True,
        aromatic: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ) -> None:
        self.subset = subset
        assert split in ['train', 'val', 'test']
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')

        if split == "train":
            self.file_idx = 0
        elif split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        download_url(self.csv_url, self.root)
        download_url(self.val_index_url, self.root)

        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def save_smiles(self):
        # Load the full dataset from CSV.
        smiles_df = pd.read_csv(osp.join(self.root, 'zinc250k.csv'))

        # Load the valid indices from JSON.
        valid_indices = pd.read_json(osp.join(self.root, 'valid_idx_zinc250k.json'))[0].tolist()

        # Loop over the splits: train, val, and test.
        for split in ['train', 'val', 'test']:
            # For 'val' and 'test', use the valid indices.
            if split in ['val', 'test']:
                indices = valid_indices
            else:
                # For training, use all indices that are not in valid_indices.
                indices = [i for i in range(len(smiles_df)) if i not in valid_indices]

            # Extract SMILES strings for the given indices.
            split_smiles = [smiles_df.iloc[idx]['smiles'] for idx in range(len(indices))]

            # Build the output CSV file name and path.
            smiles_file_name = f"{split}_smiles_origin.csv"
            smiles_path = osp.join(self.root, smiles_file_name)  # or use a specific directory if needed
            
            # save the SMILES strings to a CSV file.
            pd.DataFrame({'smiles': split_smiles}).to_csv(smiles_path, index=False)

    def process(self):
        smiles = pd.read_csv(osp.join(self.root, 'zinc250k.csv'))
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}

        for split in ['train', 'val', 'test']:
            val_indices = pd.read_json(osp.join(self.root, 'valid_idx_zinc250k.json'))[0].tolist()
            if split == 'val' or split == 'test':
                indices = val_indices
            else:
                indices = []
                for i in range(len(smiles)):
                    if i not in val_indices:
                        indices.append(i)
        
            print(f'Processing {split} dataset with {len(indices)} molecules')

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                smile = smiles.iloc[idx]['smiles']

                mol = Chem.MolFromSmiles(smile, sanitize=False)

                # Remove hydrogen atoms if they are present
                mol = Chem.RemoveHs(mol)

                # Kekulize the molecule: convert aromatic bonds to alternating single and double bonds
                # The 'clearAromaticFlags=True' option ensures that aromatic flags are removed after kekulization
                Chem.Kekulize(mol, clearAromaticFlags=True)

                N = mol.GetNumAtoms()

                type_idx = []
                for atom in mol.GetAtoms():
                    type_idx.append(types[atom.GetSymbol()])

                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [bonds[bond.GetBondType()] + 1]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                y = torch.zeros(size=(1, 0), dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))


class ZINCDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic

        target = getattr(cfg.general, "target", None)
        regressor = getattr(cfg.general, "conditional", None)
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = SelectBothTransform()
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            "train": ZINCDataset(
                split="train",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                transform=transform,
            ),
            "val": ZINCDataset(
                split="val",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                transform=transform,
            ),
            "test": ZINCDataset(
                split="test",
                root=root_path,
                remove_h=cfg.dataset.remove_h,
                aromatic=cfg.dataset.aromatic,
                transform=transform,
            ),
        }
        self.test_labels = transform(datasets["test"].data, return_y=True)
        

        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        super().__init__(cfg, datasets)


class ZINCinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.aromatic = cfg.dataset.aromatic
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )
        self.compute_fcd = cfg.dataset.compute_fcd

        self.name = "zinc"
        if self.remove_h:
            # Atom encoder/decoder
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
            self.atom_decoder = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

            self.num_atom_types = 9

            # Valencies based on chemistry rules for ZINC atoms
            self.valencies = [4, 3, 2, 1, 5, 6, 1, 1, 1]

            # Approximate atomic weights (standard values)
            self.atom_weights = {
                0: 12,   # C
                1: 14,   # N
                2: 16,   # O
                3: 19,   # F
                4: 30,   # P
                5: 32.,   # S
                6: 35.5,   # Cl
                7: 78,   # Br
                8: 127   # I
            }

            self.max_n_nodes = 38  # typical upper bound from ZINC250k
            self.max_weight = 500  # standard upper bound for ZINC molecules

            self.n_nodes = torch.tensor(
                [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                        1.3359e-05, 2.2265e-05, 5.7889e-05, 2.9835e-04, 7.9263e-04, 2.9123e-03,
                        4.6890e-03, 7.1515e-03, 1.1275e-02, 1.7117e-02, 2.5360e-02, 3.5014e-02,
                        4.6707e-02, 5.8178e-02, 7.0829e-02, 8.1472e-02, 7.4922e-02, 8.4384e-02,
                        9.3099e-02, 9.1451e-02, 7.7175e-02, 6.3397e-02, 4.0331e-02, 3.1131e-02,
                        2.4394e-02, 1.9237e-02, 1.5029e-02, 1.0362e-02, 6.9155e-03, 4.1190e-03,
                        1.5942e-03, 5.6108e-04, 8.9060e-06
                ]
            )
            self.node_types = torch.tensor([
                            7.3678e-01, 1.2211e-01, 9.9746e-02,
                            1.3745e-02, 2.4428e-05, 1.7806e-02,
                            7.4231e-03, 2.2057e-03, 1.5522e-04])
            self.edge_types = torch.tensor([9.0658e-01, 6.9411e-02, 2.3771e-02, 2.3480e-04])
            self.valency_distribution = torch.tensor(
                [0.0000e+00, 1.1364e-01, 3.0431e-01, 3.5063e-01, 2.2655e-01, 2.2697e-05,
                    4.8356e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
            )

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt("n_counts.txt", self.n_nodes.numpy())
            self.node_types = datamodule.node_types()  # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt("atom_types.txt", self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt("edge_types.txt", self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes, zinc=True)
            print("Distribution of the valencies", valencies)
            np.savetxt("valencies.txt", valencies.numpy())
            self.valency_distribution = valencies
            assert False


def get_smiles(cfg, datamodule, dataset_infos, evaluate_datasets=False):

    return {
        "train": get_loader_smiles(
            cfg,
            datamodule.train_dataloader(),
            dataset_infos,
            "train",
            evaluate_dataset=evaluate_datasets,
        ),
        "val": get_loader_smiles(
            cfg,
            datamodule.val_dataloader(),
            dataset_infos,
            "val",
            evaluate_dataset=evaluate_datasets,
        ),
        "test": get_loader_smiles(
            cfg,
            datamodule.test_dataloader(),
            dataset_infos,
            "test",
            evaluate_dataset=evaluate_datasets,
        ),
    }


def get_loader_smiles(
    cfg,
    dataloader,
    dataset_infos,
    split_key,
    evaluate_dataset=False,
):
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = (
        f"{split_key}_smiles_no_h.npy" if remove_h else f"{split_key}_smiles_h.npy"
    )
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print(f"Dataset {split_key} smiles were found.")
        smiles = np.load(smiles_path).tolist()
    else:
        print(f"Computing dataset {split_key} smiles...")
        smiles = compute_zinc_smiles(atom_decoder, dataloader, remove_h)
        np.save(smiles_path, np.array(smiles))

    if evaluate_dataset:
        # Convert loader to molecules
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
        all_molecules = []
        for i, data in enumerate(dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        # load train smiles
        train_smiles_file_name = (
            f"train_smiles_no_h.npy" if remove_h else f"train_smiles_h.npy"
        )
        train_smiles_path = os.path.join(root_dir, datadir, train_smiles_file_name)
        train_smiles = np.load(train_smiles_path)
        # get evaluation and output
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        print(metrics[0])

    return smiles


def compute_zinc_smiles(atom_decoder, train_dataloader, remove_h):
    """
    :param dataset_name: zinc or zinc_second_half
    :return:
    """
    print(f"\tConverting ZINC dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print(
                "\tConverting ZINC dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles
