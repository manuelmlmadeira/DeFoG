import math
import numpy as np
import torch
import re
import wandb
import pandas as pd
from tqdm import tqdm
from torchmetrics import MeanSquaredError, MeanAbsoluteError

try:
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
    from rdkit.Chem.rdForceFieldHelpers import (
        MMFFHasAllMoleculeParams,
        MMFFOptimizeMolecule,
    )

    print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn

    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"

try:
    import psi4
except ModuleNotFoundError:
    print("PSI4 not found")

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, train_smiles=None, args=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.args = args

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles
        self.cond_val = MeanAbsoluteError()

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        all_smiles_without_test = []

        for graph in tqdm(
            generated, desc="Generated molecules validity check progress"
        ):
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            all_smiles_without_test.append(mol2smilesWithNoSanitize(mol))
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        with open(r"final_smiles_all.txt", "w") as fp:
            for smiles in all_smiles_without_test:
                # write each item on a new line
                fp.write("%s\n" % smiles)
            print("All smiles saved")

        print(all_smiles_without_test)
        df = pd.DataFrame(all_smiles_without_test, columns=["SMILES"])
        df.to_csv("final_smiles_all.csv", index=False)
        print("All SMILES saved to CSV")

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in tqdm(unique, desc="Unique molecules novelty check progress"):
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in tqdm(
            generated, desc="Generated molecules relaxed validity check progress"
        ):
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def cond_sample_metric(self, samples, input_properties):
        mols_dipoles = []
        mols_homo = []
        self.num_valid_molecules = 0
        self.num_total = 0

        # Hardware side settings (CPU thread number and memory settings used for calculation)
        psi4.set_num_threads(nthread=4)
        psi4.set_memory("5GB")
        psi4.core.set_output_file("psi4_output.dat", False)
        # input_properties = torch.cat(input_properties).unsqueeze(-1)
        # print(input_properties)
        true_properties = []

        for i, sample in enumerate(samples):
            atom_types, edge_types = sample
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
            try:
                Chem.SanitizeMol(mol)
            except:
                print("invalid chemistry")
                continue

            # Coarse 3D structure optimization by generating 3D structure from SMILES
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1
            try:
                EmbedMolecule(mol, params)
            except Chem.rdchem.AtomValenceException:
                print("invalid chemistry")
                continue

            # Structural optimization with MMFF (Merck Molecular Force Field)
            try:
                s = MMFFOptimizeMolecule(mol)
                print(s)
            except:
                print("Bad conformer ID")
                continue

            try:
                conf = mol.GetConformer()
            except:
                print("Can not get conformer")
                continue

            # Convert to a format that can be input to Psi4.
            # Set charge and spin multiplicity (below is charge 0, spin multiplicity 1)

            # Get the formal charge
            fc = "FormalCharge"
            mol_FormalCharge = (
                int(mol.GetProp(fc)) if mol.HasProp(fc) else Chem.GetFormalCharge(mol)
            )

            sm = "SpinMultiplicity"
            if mol.HasProp(sm):
                mol_spin_multiplicity = int(mol.GetProp(sm))
            else:
                # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
                NumRadicalElectrons = 0
                for Atom in mol.GetAtoms():
                    NumRadicalElectrons += Atom.GetNumRadicalElectrons()
                TotalElectronicSpin = NumRadicalElectrons / 2
                SpinMultiplicity = 2 * TotalElectronicSpin + 1
                mol_spin_multiplicity = int(SpinMultiplicity)

            mol_input = "%s %s" % (mol_FormalCharge, mol_spin_multiplicity)
            print(mol_input)
            # mol_input = "0 1"

            # Describe the coordinates of each atom in XYZ format
            for atom in mol.GetAtoms():
                mol_input += (
                    "\n "
                    + atom.GetSymbol()
                    + " "
                    + str(conf.GetAtomPosition(atom.GetIdx()).x)
                    + " "
                    + str(conf.GetAtomPosition(atom.GetIdx()).y)
                    + " "
                    + str(conf.GetAtomPosition(atom.GetIdx()).z)
                )

            try:
                molecule = psi4.geometry(mol_input)
            except:
                print("Can not calculate psi4 geometry")
                continue

            # Convert to a format that can be input to pyscf
            # Set calculation method (functional) and basis set
            level = "b3lyp/6-31G*"

            # Calculation method (functional), example of basis set
            # theory = ['hf', 'b3lyp']
            # basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

            # Perform structural optimization calculations
            print("Psi4 calculation starts!!!")
            print("input properties", input_properties[i])
            # energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
            try:
                energy, wave_function = psi4.energy(
                    level, molecule=molecule, return_wfn=True
                )
                true_properties.append(input_properties[i])
            except psi4.driver.SCFConvergenceError:
                print("Psi4 did not converge")
                continue

            print("Chemistry information check!!!")

            if self.args.general.target in ["mu", "both"]:
                dip_x, dip_y, dip_z = (
                    wave_function.variable("SCF DIPOLE")[0],
                    wave_function.variable("SCF DIPOLE")[1],
                    wave_function.variable("SCF DIPOLE")[2],
                )
                dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                print("Dipole moment", dipole_moment)
                mols_dipoles.append(dipole_moment)

            if self.args.general.target in ["homo", "both"]:
                # Compute HOMO (Unit: au= Hartree）
                LUMO_idx = wave_function.nalpha()
                HOMO_idx = LUMO_idx - 1
                homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]

                # convert unit from a.u. to ev
                # homo = homo * 27.211324570273
                print("HOMO", homo)
                mols_homo.append(homo)

        true_properties = torch.cat(true_properties).unsqueeze(-1)

        num_valid_molecules = max(len(mols_dipoles), len(mols_homo))
        print("Number of valid samples", num_valid_molecules)
        self.num_valid_molecules += num_valid_molecules
        self.num_total += len(samples)

        mols_dipoles = torch.FloatTensor(mols_dipoles)
        mols_homo = torch.FloatTensor(mols_homo)

        if self.args.general.target == "mu":
            mae = self.cond_val(
                mols_dipoles.unsqueeze(1),
                # input_properties.repeat(len(mols_dipoles), 1).cpu(),
                true_properties,
            )

        elif self.args.general.target == "homo":
            mae = self.cond_val(
                mols_homo.unsqueeze(1),
                # input_properties.repeat(len(mols_homo), 1).cpu()
                true_properties,
            )

        elif self.args.general.target == "both":
            properties = torch.hstack(
                (mols_dipoles.unsqueeze(1), mols_homo.unsqueeze(1))
            )
            mae = self.cond_val(
                properties,
                # input_properties.repeat(len(mols_dipoles), 1).cpu()
                true_properties,
            )

        print("Conditional generation metric:")
        print(f"MAE: {mae}")
        # wandb.log(
        #     {
        #         "val_epoch/conditional generation mae": mae,
        #         "Valid molecules": num_valid_molecules,
        #     }
        # )

        return mae, self.num_valid_molecules / self.num_total

    def evaluate(self, generated, input_properties=None, test=False):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        valid, validity, num_components, all_smiles = self.compute_validity(generated)

        if test:
            with open(r"final_smiles.txt", "w") as fp:
                for smiles in all_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print("All smiles saved")

            print(all_smiles)
            df = pd.DataFrame(all_smiles, columns=["SMILES"])
            df.to_csv("final_smiles.csv", index=False)
            print("All SMILES saved to CSV")

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}"
        )

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        print(
            f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%"
        )

        cond_mae = cond_val = -1.0
        if input_properties is not None:
            cond_mae, cond_val = self.cond_sample_metric(generated, input_properties)

        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(
                f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(
                    f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
                )
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return (
            [validity, relaxed_validity, uniqueness, novelty],
            unique,
            dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            all_smiles,
            [cond_mae, cond_val],
        )


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def mol2smilesWithNoSanitize(mol):
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0  # set edges in virtual state to non-bonded
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
    return mol


def build_molecule_with_partial_charges(
    atom_types, edge_types, atom_decoder, verbose=False
):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0  # set edges in virtual state to non-bonded
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [
            (s, len(s)) for s in sm.split(".")
        ]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


if __name__ == "__main__":
    smiles_mol = "C1CCC1"
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)

use_rdkit = True


def check_stability(
    atom_types, edge_types, dataset_info, debug=False, atom_decoder=None
):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype="int")

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type], atom_n_bond)
            )
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(
    molecule_list, train_smiles, dataset_info, labels, args=None, test=False
):
    """molecule_list: (dict)"""

    if not dataset_info.remove_h:
        print(f"Analyzing molecule stability...")

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in tqdm(
            enumerate(molecule_list), desc="Stability computation progress"
        ):
            atom_types, edge_types = mol

            validity_results = check_stability(atom_types, edge_types, dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
        if wandb.run:
            wandb.log(validity_dict)
    else:
        validity_dict = {"mol_stable": -1, "atm_stable": -1}

    metrics = BasicMolecularMetrics(dataset_info, train_smiles, args)
    rdkit_metrics = metrics.evaluate(molecule_list, labels, test)
    all_smiles = rdkit_metrics[-2]

    nc = rdkit_metrics[-3]
    dic = {
        "Validity": rdkit_metrics[0][0],
        "Relaxed Validity": rdkit_metrics[0][1],
        "Uniqueness": rdkit_metrics[0][2],
        "Novelty": rdkit_metrics[0][3],
        "nc_max": nc["nc_max"],
        "nc_mu": nc["nc_mu"],
        "cond_mae": rdkit_metrics[-1][0],
        "cond_val": rdkit_metrics[-1][1],
    }
    if wandb.run:
        dic = dic
        wandb.log(dic)

    return validity_dict, rdkit_metrics, all_smiles, dic
