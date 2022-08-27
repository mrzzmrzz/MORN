dataset = "iron"
dataset_seed = 100
cuda_device_num = 1
num_tasks = None

# here is the num_tasks para
if dataset in ["bace", "bbbp", "hiv", "iron"]:
    num_tasks = 1
elif dataset == "clintox":
    num_tasks = 2
elif dataset == "tox21":
    num_tasks = 12
elif dataset == "toxcast":
    num_tasks = 617
elif dataset == "sider":
    num_tasks = 27
elif dataset == "muv":
    num_tasks = 17
else:
    num_tasks = -1

# here is the max_len para
smiles_max_len_dict = {
    "bace": 194,
    "bbbp": 234,
    "hiv": 495,
    "clintox": 254,
    "tox21": 267,
    "sider": 982,
    "muv": 84,
    "iron": 211,
}

max_len = smiles_max_len_dict.get(dataset, 500)

###########################################################
#                   Layers Settings                       #
###########################################################

gnn_cls_config = {
    "gnn_config": {
        "embed_dim": 64,
        "dropout": 0.5,
        "layer_num": 5,
        "readout": "mean",
        "atom_names": [
            "atomic_num",
            "formal_charge",
            "degree",
            "chiral_tag",
            "total_numHs",
            "is_aromatic",
            "hybridization",
        ],
        "bond_names": ["bond_dir", "bond_type", "is_in_ring"],
        "bond_float_names": ["bond_length"],
        "bond_angle_float_names": ["bond_angle"],
    },
    "classifier_config": {
        "layer_num": 2,
        "input_dim": 64,
        "hidden_dim": 128,
        "output_dim": num_tasks,
        "dropout_rate": 0.2,
    },
}

seq_cls_config = {
    "seq_config": {
        "embed_dim": 32,
        "hid_dim": 128,
        "out_dim": 64,
        "num_layer": 2,
        "max_len": max_len,
    },
    "classifier_config": {
        "layer_num": 2,
        "input_dim": 64,
        "hidden_dim": 64,
        "output_dim": num_tasks,
        "dropout_rate": 0.2,
    },
}

clfu_config = {
    "dim_in": 64,
    "dim_hid": 128,
    "mlp_config": {
        "layer_num": 2,
        # after the dim_hid
        "input_dim": 128,
        "hidden_dim": 256,
        "output_dim": num_tasks,
        "dropout_rate": 0.2,
    },
}

###########################################################
#                   Trainer Settings                      #
###########################################################

trainer_gnn_config = {
    "seed": dataset_seed,
    "cuda_device": cuda_device_num,
    "batch_size": 64,
    "lr": 3e-4,
    "epoch": 300,
    "dataset": dataset,
    "weight_decay": 1e-5,
    "model_type": "gnn",
    "early_stop_patience": 30,
    "comments": "train gnn",
    "save_id": 100,
}

trainer_seq_config = {
    "seed": dataset_seed,
    "cuda_device": cuda_device_num,
    "batch_size": 64,
    "lr": 4e-5,  # 5e-5
    "epoch": 300,
    "dataset": dataset,
    "weight_decay": 1e-3,
    "model_type": "seq",
    "early_stop_patience": 30,
    "comments": "train seq",
    "save_id": 100,
}

trainer_clfu_config = {
    "seed": dataset_seed,
    "cuda_device": cuda_device_num,
    "batch_size": 32,
    "lr": 1e-4,
    "epoch": 1000,
    "dataset": dataset,
    "weight_decay": 1e-5,
    "model_type": "CLFU",
    "early_stop_patience": 30,
    "de_novo_train_graph": False,
    "de_novo_train_seq": False,
    "comments": "late fusion",
    "save_id": 1,
}
