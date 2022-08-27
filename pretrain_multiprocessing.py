import multiprocessing
from pickle import dump
from math import ceil

from dataset_load import *

from utils.compound_tools import *

len_dict = {
    "bace": 194,
    "bbbp": 234,
    "hiv": 495,
    "clintox": 254,
    "tox21": 267,
    "sider": 982,
    "toxcast": 240,
    "muv": 84,
    "iron": 211,
}


def run_geo_for_smiles(start, end, mol_list):
    cur_mol_list = mol_list[start: end]
    raw_data_list = []
    for mol in cur_mol_list:
        raw_data = mol_to_geognn_graph_data_MMFF3d(mol)
        raw_data_list.append(raw_data)
    return raw_data_list


def run_seq_for_smiles(start, end, smiles_list, max_len):
    cur_smiles_list = smiles_list[start: end]
    return CompoundSeqKit.encode_multi_smiles(cur_smiles_list, max_len)


def preprocess_graph_feat(input_path,
                          seed=42,
                          process_num=20,
                          dataset="bace",
                          task="graph"):
    cur_path = os.getcwd()

    if dataset == "bace":
        trn_df, val_df, test_df = load_bace_dataset(
            input_path, seed, random=False)
    elif dataset == "bbbp":
        trn_df, val_df, test_df = load_bbbp_dataset(
            input_path, seed, random=False)
    elif dataset == "hiv":
        trn_df, val_df, test_df = load_hiv_dataset(
            input_path, seed, random=False)
    elif dataset == "clintox":
        trn_df, val_df, test_df = load_clintox_dataset(input_path, seed)
    elif dataset == "tox21":
        trn_df, val_df, test_df = load_tox21_dataset(input_path, seed)
    elif dataset == "sider":
        trn_df, val_df, test_df = load_sider_dataset(input_path, seed)
    elif dataset == "toxcast":
        trn_df, val_df, test_df = load_toxcast_dataset(input_path, seed)
    elif dataset == "muv":
        trn_df, val_df, test_df = load_muv_dataset(input_path, seed)
    elif dataset == "iron":
        trn_df, val_df, test_df = load_iron_dataset(input_path, seed)
    else:
        raise ValueError("Unsupported Dataset!")

    print("dataset loaded!")
    path_prefix_name = ["train", "valid", "test"]

    cur_path = os.getcwd()
    saved_graph_feat_path = os.path.join(
        cur_path, "molattrs/saved_graph_feat/{}".format(dataset))
    saved_seq_feat_path = os.path.join(
        cur_path, "molattrs/saved_seq_feat/{}".format(dataset))

    check_save_pre_path = [saved_graph_feat_path, saved_seq_feat_path]
    for path in check_save_pre_path:
        if not os.path.exists(path):
            os.mkdir(path)

    for name in path_prefix_name:
        cur_graph_feat_path = os.path.join(
            saved_graph_feat_path, "{}".format(name))
        cur_seq_feat_path = os.path.join(
            saved_seq_feat_path, "{}".format(name))

        cur_path_check = [cur_graph_feat_path, cur_seq_feat_path]

        for path in cur_path_check:
            if not os.path.exists(path):
                os.mkdir(path)

    for i, dataframe in enumerate([trn_df, val_df, test_df]):
        smiles_list = list(dataframe["smiles"])
        smiles_len = len(smiles_list)
        print("{} :{}".format(path_prefix_name[i], smiles_len))

        rdkit_mol_list = [AllChem.MolFromSmiles(
            smiles) for smiles in smiles_list]
        block_num = ceil(smiles_len / process_num)
        res = []

        pool = multiprocessing.Pool(processes=process_num)
        for j in tqdm(range(block_num)):
            start_index = j * process_num
            if j != block_num - 1:
                end_index = (j + 1) * process_num
            else:
                end_index = smiles_len
            print(start_index, end_index)

            # Here is for different tasks
            if task == "graph":
                res.append(pool.apply_async(
                    run_geo_for_smiles, args=(start_index, end_index, rdkit_mol_list)))
            if task == "seq":
                max_len = len_dict.get(dataset, 100)
                res.append(pool.apply_async(
                    run_seq_for_smiles, args=(start_index, end_index, smiles_list, max_len)))

        pool.close()
        pool.join()
        print("All Subprocesses Have Been Done!")

        processed_data_list = []
        for r in res:
            processed_data_list += r.get()
        cur_prefix_name = os.path.join(dataset, path_prefix_name[i])

        save_path = None
        if task == "graph":
            save_path = os.path.join(
                cur_path,
                "molattrs/saved_graph_feat", cur_prefix_name, "graph_feat_{}.pt".format(seed))
        if task == "seq":
            save_path = os.path.join(
                cur_path,
                "molattrs/saved_seq_feat", cur_prefix_name, "seq_feat_{}.pt".format(seed))

        with open(save_path, "wb") as f:
            dump(processed_data_list, f)
        print(save_path, "Have been saved!")


def main_process(input_path: str, process_number: int, seed_num: int, dataset: str):
    preprocess_graph_feat(input_path,
                          task="graph",
                          dataset=dataset,
                          process_num=process_number,
                          seed=seed_num)
    preprocess_graph_feat(input_path,
                          task="seq",
                          dataset=dataset,
                          process_num=process_number,
                          seed=seed_num)

    print("#" * 100, "multiprocess preprocessing has done!", "#" * 100, sep="\n")
