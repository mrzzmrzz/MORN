import random
import re
from typing import List

import numpy as np
import rdkit
import rdkit.Chem as Chem
import torch
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


from utils.trans_dict import _i2a, _a2i, _pair_list


def get_tokenizer_re(atoms):
    return re.compile("(" + "|".join(atoms) + r"|\%\d\d|.)")


def smiles_separator(smiles, pair_list=_pair_list):
    """
    :param pair_list: the two-atom list to recognize
    :param smiles: str, the initial smiles string
    :return: list, the smiles list after seperator
                    [recognize the atom and the descriptor]
    """
    if pair_list:
        reg = get_tokenizer_re(pair_list)
    else:
        reg = get_tokenizer_re(_pair_list)

    smiles_list = reg.split(smiles)[1::2]
    return smiles_list


def encode_smiles(smiles: List[str], pad_size: int):
    # make the smiles string into a vector list
    """
    :param smiles: list, the smiles list after separator
    :param pad_size: int, the length of this vector list
    :return: list, the vector list representing the smiles
    """
    cur_smiles_len = len(smiles)
    res = [_a2i[s] for s in smiles]
    if cur_smiles_len > pad_size:
        return res[:pad_size]
    else:
        return res + [0] * (pad_size - cur_smiles_len)


def smiles_to_tensor(smiles: str, pad_size=50):
    # change smiles[text] into tensor format
    smiles_tensor = smiles_separator(smiles)
    smiles_tensor = encode_smiles(smiles_tensor, pad_size)
    smiles_tensor = torch.tensor(smiles_tensor).long()
    return smiles_tensor


def encode_smiles_full_length(smiles: List[str]):
    # make the smiles string into a vector list
    """
    :param smiles: list, the smiles list after separator
    :return: list, the vector list representing the smiles
    """
    smiles_tensor = smiles_separator(smiles)
    res = [_a2i[s] for s in smiles]
    return res, len(res)


def encode_multi_smiles_list(smiles_list):
    vec_list = []
    len_list = []
    for smiles in smiles_list:
        smiles_norm = smiles_separator(smiles)
        vec, length = encode_smiles_full_length(smiles_norm)
        vec_list.append(vec)
        len_list.append(length)
    return vec_list, len_list


def smiles_list_to_matrix(smiles_list: List[str]):
    tmp_list = torch.stack([smiles_to_tensor(smiles) for smiles in smiles_list])
    return tmp_list


def get_pos_smiles_matrix(smiles_list: List[str]):
    tmp_list = [get_pos_smiles(smiles) for smiles in smiles_list]
    return smiles_list_to_matrix(tmp_list)


def get_mol_list(smiles_list):
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    return mol_list


def get_pos_smiles(smiles: str):
    """
    :param smiles: the initial smiles string, type: str
    :return: a list containing positive smiles related to the input smiles, type: list[str]
    """
    # you can choose to use the tryEmbedding to get better result
    opts = StereoEnumerationOptions(tryEmbedding=False)
    mol = Chem.MolFromSmiles(smiles)

    # produce the isomers based the rdkit package
    isomers = tuple(EnumerateStereoisomers(mol, opts))

    # randomly choose one positive sample
    pos_smiles = Chem.MolToSmiles(random.choice(isomers))

    return pos_smiles


def get_neg_smiles(pad_size):
    # randomly produce the negative smiles
    num_pad = random.randint(0, 10)
    num_fill = pad_size - num_pad
    avail_smiles_symbol = [key for key in _i2a.keys()]
    neg_smiles = np.random.choice(avail_smiles_symbol, num_fill)
    neg_smiles = np.concatenate([neg_smiles, [0] * num_pad], axis=0)
    return neg_smiles


def to_onehot(val, cat):
    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c:
            vec[i] = 1
    if np.sum(vec) == 0:
        print("* exception: missing category", val)
    assert np.sum(vec) == 1
    return vec


def get_ring_info(atom: Chem.Atom):
    # get the atom's ring info
    if atom.IsInRing():
        ring_info = get_ring_size(atom)
        ring_info = to_onehot(ring_info, [3, 4, 5, 6, 7, 8, 9])
        return np.concatenate([[1], ring_info], axis=0)
    else:
        return [0] * 8


def get_ring_size(atom: Chem.Atom):
    # get the ring size of one target atom
    # if the ring size is bigger than 10
    # return the predefined marco size
    macro_size = 9
    ring_size = [3, 4, 5, 6, 7, 8]
    for r_size in ring_size:
        if atom.IsInRingSize(r_size):
            return r_size
    return macro_size


def hybridization2vec(hybrid_type):
    if hybrid_type == rdkit.Chem.rdchem.HybridizationType.SP:
        res = [0, 0, 0, 1]
    elif hybrid_type == rdkit.Chem.rdchem.HybridizationType.SP2:
        res = [0, 0, 1, 0]
    elif hybrid_type == rdkit.Chem.rdchem.HybridizationType.SP3:
        res = [0, 1, 0, 0]
    else:
        res = [1, 0, 0, 0]
    return res


def get_edge_index(mol, use_Aromatic=False):
    row = []
    col = []
    # if not use_Aromatic:
    #     Chem.Kekulize(mol)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        cur_row, cur_col = [i, j], [j, i]
        typ = bond.GetBondType()
        if typ == rdkit.Chem.rdchem.BondType.DOUBLE:
            cur_row += [i, j]
            cur_col += [j, i]
        elif typ == rdkit.Chem.rdchem.BondType.TRIPLE:
            cur_row += [i, j] * 2
            cur_col += [j, i] * 2
        else:
            cur_row += [i, j] * 3
            cur_col += [j, i] * 3

        row += cur_row
        col += cur_col
    edge_index = [row, col]
    return torch.tensor(edge_index, dtype=torch.long)


def get_atom_features_onehot(atom: Chem.Atom, max_atom_idx=53):
    # TODO: remind to update the feature
    atom_charge = to_onehot(atom.GetFormalCharge(), [-1, 1, 0])[:2]
    atom_H = to_onehot(atom.GetNumExplicitHs(), [1, 2, 3, 0])[:3]
    atom_ring = get_ring_info(atom)
    atom_hybrid = hybridization2vec(atom.GetHybridization())
    # atom_mass
    # atom_atomic_num
    # v3 = np.zeros(max_atom_idx)
    # v3[atom.GetAtomicNum() - 1] = 1
    return np.concatenate([atom_charge, atom_H, atom_ring, atom_hybrid], axis=0)


def get_mol_feat_onehot(mol):
    atom_feat = np.array(
        [np.array(get_atom_features_onehot(cur_atom)) for cur_atom in mol.GetAtoms()]
    )
    return torch.tensor(atom_feat, dtype=torch.float32)


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    node_feat = get_mol_feat_onehot(mol)
    edge_index = get_edge_index(mol)
    return node_feat, edge_index


if __name__ == "__main__":
    smiles = "FC(F)Oc1ccc(cc1)[C@@]1(N=C(N)N(C)C1=O)c1cc(ccc1)CCCC#N"
    mol = Chem.MolFromSmiles(smiles)
    # feat = get_mol_feat_onehot(mol)
    edge_index = get_edge_index(mol, use_Aromatic=False)
    print(edge_index)
