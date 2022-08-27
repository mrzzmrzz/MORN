# Tools for compound features.
# Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
# Adapted From PaddlePaddle Helix

#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle
import re
from copyreg import pickle
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import subprocess

# import cairosvg
# import cv2
import random
from tqdm import tqdm

from utils.compound_constants import DAY_LIGHT_FG_SMARTS_LIST
from utils.trans_dict import _a2i, _pair_list


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.

    Args:
        mol: rdkit mol object.
        n_iter(int): number of iterations. Default 12.

    Returns:
        list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(
        mol, nIter=n_iter, throwOnParamFailure=True
    )
    partial_charges = [float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.

    Returns:
        inchi.
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(
            AllChem.MolFromSmiles(smiles), isomericSmiles=False
        )
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None:
            if "." in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.

    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split(".")
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.

    Args:
        mol_list(list): a list of rdkit mol object.

    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def rdchem_enum_to_list(values):
    """
    values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """
    tbd
    """
    return list(
        map(
            len,
            [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names],
        )
    )


def get_bond_feature_dims(list_acquired_feature_names):
    """tbd"""
    list_bond_feat_dim = list(
        map(
            len,
            [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names],
        )
    )
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit
    """

    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ["misc"],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "misc"],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "misc"],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "num_radical_e": [0, 1, 2, 3, 4, "misc"],
        "atom_is_in_ring": [0, 1],
        "valence_out_shell": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size3": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size4": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size5": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size6": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size7": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size8": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    }

    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        "bond_stereo": rdchem_enum_to_list(rdchem.BondStereo.values),
        "is_conjugated": [0, 1],
    }
    # float features
    # van_der_waals_radis: 范德华半径
    # partial charge: 部分电子
    # mass: 质量
    atom_float_names = ["van_der_waals_radis", "partial_charge", "mass"]

    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    # functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [
        Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list
    ]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    # atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == "atomic_num":
            return atom.GetAtomicNum()
        elif name == "chiral_tag":
            return atom.GetChiralTag()
        elif name == "degree":
            return atom.GetDegree()
        elif name == "explicit_valence":
            return atom.GetExplicitValence()
        elif name == "formal_charge":
            return atom.GetFormalCharge()
        elif name == "hybridization":
            return atom.GetHybridization()
        elif name == "implicit_valence":
            return atom.GetImplicitValence()
        elif name == "is_aromatic":
            return int(atom.GetIsAromatic())
        elif name == "mass":
            return int(atom.GetMass())
        elif name == "total_numHs":
            return atom.GetTotalNumHs()
        elif name == "num_radical_e":
            return atom.GetNumRadicalElectrons()
        elif name == "atom_is_in_ring":
            return int(atom.IsInRing())
        elif name == "valence_out_shell":
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        # atom： 原子， rdkit.Chem.Atom 类型
        # name: 属性名称，例如'atomic_num'， 'chiral_tag'等
        # 返回：在该属性的列表中的位置，否则返回len(list)-1
        assert name in CompoundKit.atom_vocab_dict, (
            "%s not found in atom_vocab_dict" % name
        )
        return safe_index(
            CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name)
        )

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        # name: 属性名称，例如'atomic_num'， 'chiral_tag'等
        # 返回：该属性的列表中的长度
        assert name in CompoundKit.atom_vocab_dict, (
            "%s not found in atom_vocab_dict" % name
        )
        return len(CompoundKit.atom_vocab_dict[name])

    # bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == "bond_dir":
            return bond.GetBondDir()
        elif name == "bond_type":
            return bond.GetBondType()
        elif name == "is_in_ring":
            return int(bond.IsInRing())
        elif name == "is_conjugated":
            return int(bond.GetIsConjugated())
        elif name == "bond_stereo":
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, (
            "%s not found in bond_vocab_dict" % name
        )
        return safe_index(
            CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name)
        )

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, (
            "%s not found in bond_vocab_dict" % name
        )
        return len(CompoundKit.bond_vocab_dict[name])

    # fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    # functional groups
    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        # 对于分子中的每一个原子，返回该原子所处环的信息
        # 每个原子对应一个列表，列表从3环开始：3环，4环，5环，6环，7环，8环
        # 每个位置的数值表示，该原子所处环的个数
        # [0, 0, 1, 1, 0, 0]表示，该原子参与了一个五环和一个六环的组成
        # [0, 0, 0, 2, 0, 0]表示，该原子参与了两个六元环的组成
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)

            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """tbd"""
        atom_names = {
            "atomic_num": safe_index(
                CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()
            ),
            "chiral_tag": safe_index(
                CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()
            ),
            "degree": safe_index(
                CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()
            ),
            "explicit_valence": safe_index(
                CompoundKit.atom_vocab_dict["explicit_valence"],
                atom.GetExplicitValence(),
            ),
            "formal_charge": safe_index(
                CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()
            ),
            "hybridization": safe_index(
                CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()
            ),
            "implicit_valence": safe_index(
                CompoundKit.atom_vocab_dict["implicit_valence"],
                atom.GetImplicitValence(),
            ),
            "is_aromatic": safe_index(
                CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())
            ),
            "total_numHs": safe_index(
                CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()
            ),
            "num_radical_e": safe_index(
                CompoundKit.atom_vocab_dict["num_radical_e"],
                atom.GetNumRadicalElectrons(),
            ),
            "atom_is_in_ring": safe_index(
                CompoundKit.atom_vocab_dict["atom_is_in_ring"], int(atom.IsInRing())
            ),
            "valence_out_shell": safe_index(
                CompoundKit.atom_vocab_dict["valence_out_shell"],
                CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum()),
            ),
            "van_der_waals_radis": CompoundKit.period_table.GetRvdw(
                atom.GetAtomicNum()
            ),
            "partial_charge": CompoundKit.check_partial_charge(atom),
            "mass": atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        # 'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]["in_num_ring_with_size3"] = safe_index(
                CompoundKit.atom_vocab_dict["in_num_ring_with_size3"], ring_list[i][0]
            )
            atom_features_dicts[i]["in_num_ring_with_size4"] = safe_index(
                CompoundKit.atom_vocab_dict["in_num_ring_with_size4"], ring_list[i][1]
            )
            atom_features_dicts[i]["in_num_ring_with_size5"] = safe_index(
                CompoundKit.atom_vocab_dict["in_num_ring_with_size5"], ring_list[i][2]
            )
            atom_features_dicts[i]["in_num_ring_with_size6"] = safe_index(
                CompoundKit.atom_vocab_dict["in_num_ring_with_size6"], ring_list[i][3]
            )
            atom_features_dicts[i]["in_num_ring_with_size7"] = safe_index(
                CompoundKit.atom_vocab_dict["in_num_ring_with_size7"], ring_list[i][4]
            )
            atom_features_dicts[i]["in_num_ring_with_size8"] = safe_index(
                CompoundKit.atom_vocab_dict["in_num_ring_with_size8"], ring_list[i][5]
            )

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp("_GasteigerCharge")

        # TODO: needs to understand
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float("inf"):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""

    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)

            # MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            # 获得边的长度
            bond_lengths.append(
                np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i])
            )
        bond_lengths = np.array(bond_lengths, "float32")
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type="HT"):
        """get superedge angles"""

        # 获得两条边之间的夹角

        def _get_vec(atom_poses, edge):
            # 向量长度
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            # 获得两个向量之间的夹角
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            # target atom idx
            tar_edge = edges[tar_edge_i]
            if dir_type == "HT":
                # head, tail -> HT
                # "终点-起点" 相同，作为一组边
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == "HH":
                # "终点-终点" 相同，做为一组边
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                # H -> H or H -> T
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], "int64")
            bond_angles = np.zeros(
                [
                    0,
                ],
                "float32",
            )
        else:
            super_edges = np.array(super_edges, "int64")
            bond_angles = np.array(bond_angles, "float32")

        return super_edges, bond_angles, bond_angle_dirs


class CompoundImageKit:
    @staticmethod
    def save_comp_imgs_from_smiles(smiles, index, pre_path):
        IMG_SIZE = 200
        mol = Chem.MolFromSmiles(smiles)
        DrawingOptions.atomLabelFontSize = 55
        DrawingOptions.dotsPerAngstrom = 100
        DrawingOptions.bondLineWidth = 1.5
        Draw.MolToFile(
            mol,
            os.path.join(pre_path, "{}.svg".format(index)),
            size=(IMG_SIZE, IMG_SIZE),
        )
        cairosvg.svg2png(
            url=os.path.join(pre_path, "{}.svg".format(index)),
            write_to=os.path.join(pre_path, "{}.png".format(index)),
        )
        subprocess.call(["rm", os.path.join(pre_path, "{}.svg".format(index))])
        img_arr = cv2.imread(os.path.join(pre_path, "{}.png".format(index)))
        if random.random() >= 0.50:
            angle = random.randint(0, 359)
            rows, cols, channel = img_arr.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_arr = cv2.warpAffine(
                img_arr,
                rotation_matrix,
                (cols, rows),
                cv2.INTER_LINEAR,
                borderValue=(255, 255, 255),
            )  # cv2.BORDER_CONSTANT, 255)

        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.astype(np.float32).transpose(2, 0, 1)
        return torch.tensor(img_arr)

    @staticmethod
    def process_to_image_scaffold(
        smiles_list, index: int, dataset="bace", mode="train"
    ):
        image_feat_list = []
        pre_path = os.path.join(os.getcwd(), "media/IMG", dataset, mode, str(index))
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        for i, smiles in enumerate(tqdm(smiles_list)):
            image_feat = CompoundImageKit.save_comp_imgs_from_smiles(
                smiles, i, pre_path
            )
            image_feat_list.append(image_feat)
        return image_feat_list

    @staticmethod
    def uniformRandomRotation():
        # QR decomposition
        q, r = np.linalg.qr(np.random.normal(size=(3, 3)))
        M = np.dot(q, np.diag(np.sign(np.diag(r))))
        if np.linalg.det(M) < 0:  # Fixing the flipping
            M[:, 0] = -M[:, 0]  # det(M)=1
        return M

    @staticmethod
    def get_mol_center_pos(atom_poses):
        return atom_poses.mean(axis=0).astype(np.float32)

    @staticmethod
    def rotate(coords, rotMat, center=(0, 0, 0)):
        """
        Rotate a selection of atoms by a given rotation around a center
        """
        new_coords = coords - center
        return np.dot(new_coords, np.transpose(rotMat)) + center

    @staticmethod
    def generate_image(smile, index, pre_path):
        mol = Chem.MolFromSmiles(smile)
        atom_poses_list = []

        optimized_mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol)
        atom_poses = np.array(atom_poses)

        # get the molecular's postion center
        # which we will get the mean pos according the axis = 0
        lig_center = CompoundImageKit.get_mol_center_pos(atom_poses)
        n_atoms = len(atom_poses)

        atom_poses_list.append(atom_poses)
        for p in range(n_atoms - 1):
            # Get the Rotation Matrix
            rrot = CompoundImageKit.uniformRandomRotation()
            new_atom_poses = CompoundImageKit.rotate(
                atom_poses, rrot, center=lig_center
            )
            atom_poses_list.append(new_atom_poses)

        atom_poses_arr = np.array(atom_poses_list)
        path = CompoundImageKit.saveImageFromCoords(atom_poses_arr, index, pre_path)
        img = plt.imread(path)

        # normalization [0,255] -> [0,1]
        img = img / 255.0
        img = img.astype(np.float32).transpose(2, 0, 1)
        return torch.tensor(img)

    @staticmethod
    def saveImageFromCoords(coords, index, pre_path):
        xmax = np.max(coords)
        xmin = np.min(coords)
        ymax = 255
        ymin = 0
        # min-max uniform -> [0,255]
        aa = (ymax - ymin) * (coords - xmin) / ((xmax - xmin) + ymin)
        piexl = np.round(aa)
        image = piexl.astype(int)

        plt.axis("off")
        fig = plt.gcf()

        # dpi = 300
        # output = 60*60 pixels 0.608
        fig.set_size_inches(0.608 / 3, 0.608 / 3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(image)

        path = pre_path + "/" + str(index) + ".jpg"
        fig.savefig(path, format="jpg", transparent=True, dpi=990, pad_inches=0)
        return path

    @staticmethod
    def preprocess_to_image(smiles_list, index: int, dataset="bace", mode="train"):
        image_feat_list = []
        pre_path = (
            os.getcwd() + "/media/3DImage/" + dataset + "/" + mode + "/" + str(index)
        )
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        for i, smiles in enumerate(tqdm(smiles_list)):
            image_feat = CompoundImageKit.generate_image(smiles, i, pre_path)
            image_feat_list.append(image_feat)
        return image_feat_list


class CompoundSeqKit:
    @staticmethod
    def get_tokenizer_re(atoms):
        return re.compile("(" + "|".join(atoms) + r"|\%\d\d|.)")

    @staticmethod
    def smiles_separator(smiles, pair_list=_pair_list):
        """
        :param pair_list: the two-atom list to recognize
        :param smiles: str, the initial smiles string
        :return: list, the smiles list after seperator
                        [recognize the atom and the descriptor]
        """
        if pair_list:
            reg = CompoundSeqKit.get_tokenizer_re(pair_list)
        else:
            reg = CompoundSeqKit.get_tokenizer_re(_pair_list)

        smiles_list = reg.split(smiles)[1::2]
        return smiles_list

    @staticmethod
    def encode_single_smiles(smiles: str, max_len: int):
        # make the smiles string into a vector list
        """
        :param smiles: list, the smiles list after separator
        :param max_len: int, the length of returned seq
        :return: list, the vector list representing the smiles
        """
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles))
        seperator_smiles = CompoundSeqKit.smiles_separator(smiles)
        cur_smiles_len = len(seperator_smiles)
        res = [_a2i[s] for s in seperator_smiles]
        res = res + [0] * (max_len - cur_smiles_len)
        res = torch.tensor(res).long()
        return res, cur_smiles_len

    @staticmethod
    def encode_multi_smiles(smiles_list: str, max_len: int):
        seq_feat_list = []
        for i, smiles in enumerate(tqdm(smiles_list)):
            seq_feat, lens = CompoundSeqKit.encode_single_smiles(smiles, max_len)
            seq_feat_list.append((seq_feat, lens))
        return seq_feat_list


def smiles_to_graph_data(smiles, **kwargs):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = mol_to_graph_data(mol)
    return data


def mol_to_graph_data(mol):
    """
    mol_to_graph_data

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = (
        list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    )
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    data = {}

    # atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    # bond and bond features
    for name in bond_id_names:
        data[name] = []
    data["edges"] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data["edges"] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    # self loop
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data["edges"] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = (
            get_bond_feature_dims([name])[0] - 1
        )  # self loop: value = len - 1
        data[name] += [bond_feature_id] * N

    # make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], "int64")
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], "float32")
    for name in bond_id_names:
        data[name] = np.array(data[name], "int64")
    data["edges"] = np.array(data["edges"], "int64")

    # morgan fingerprint
    data["morgan_fp"] = np.array(CompoundKit.get_morgan_fingerprint(mol), "int64")
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data["maccs_fp"] = np.array(CompoundKit.get_maccs_fingerprint(mol), "int64")
    data["daylight_fg_counts"] = np.array(
        CompoundKit.get_daylight_functional_group_counts(mol), "int64"
    )
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data["atom_pos"] = np.array(atom_poses, "float32")
    data["bond_length"] = Compound3DKit.get_bond_lengths(
        data["edges"], data["atom_pos"]
    )
    (
        BondAngleGraph_edges,
        bond_angles,
        bond_angle_dirs,
    ) = Compound3DKit.get_superedge_angles(data["edges"], data["atom_pos"])
    data["BondAngleGraph_edges"] = BondAngleGraph_edges
    data["bond_angle"] = np.array(bond_angles, "float32")
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type="HT")


def mol_to_geognn_graph_data_raw3d(mol):
    """tbd"""
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type="HT")


def atom_attr(data: dict, atom_attr_list=None):
    if atom_attr_list is None:
        atom_attr_list = [
            "atomic_num",
            "formal_charge",
            "degree",
            "chiral_tag",
            "total_numHs",
            "is_aromatic",
            "hybridization",
        ]
    atom_attr_search_list = []
    for attr in atom_attr_list:
        atom_attr_search_list.append(data.get(attr, None))
    return torch.tensor(np.array(atom_attr_search_list), dtype=torch.int32).T


def edge_attr(data: dict, edge_attr_list=None):
    if edge_attr_list is None:
        edge_attr_list = ["bond_dir", "bond_type", "is_in_ring", "bond_length"]
    edge_attr_search_list = []
    for attr in edge_attr_list:
        edge_attr_search_list.append(data.get(attr, None))
    return torch.tensor(np.array(edge_attr_search_list), dtype=torch.float64).T


if __name__ == "__main__":
    import pandas as pd

    bbbp_smiles_df = pd.read_csv(
        "/home/lab/mrz/Proj/Chem/CLFU/dataset/bbbp/raw/bbbp.csv"
    )
    bbbp_smiles_list = bbbp_smiles_df["smiles"]
    CompoundImageKit.preprocess_to_image(bbbp_smiles_list, "bbbp")
