from models.pipeline_model import CLFUModel
from data_preprocessing import RawMoleculeDataset
from trainer import Trainer
from pipeline_trainer import LateFusion
from models.pretrain_model import GNNPreModel
from models.pretrain_model import SeqPreModel
import os
from config import (
    gnn_cls_config,
    seq_cls_config,
    clfu_config,
    trainer_clfu_config,
    trainer_gnn_config,
    trainer_seq_config,
)

cur_path = os.getcwd()
root_path = cur_path + "/dataset/{}/".format(trainer_clfu_config["dataset"])

train_dataset = RawMoleculeDataset(
    root=root_path,
    seed=trainer_clfu_config["seed"],
    mode="train",
    dataset=trainer_clfu_config["dataset"],
)

valid_dataset = RawMoleculeDataset(
    root=root_path,
    seed=trainer_clfu_config["seed"],
    mode="valid",
    dataset=trainer_clfu_config["dataset"],
)

test_dataset = RawMoleculeDataset(
    root=root_path,
    seed=trainer_clfu_config["seed"],
    mode="test",
    dataset=trainer_clfu_config["dataset"],
)

print("DataSet Loaded! Current DataSet is {}".format(
    trainer_clfu_config["dataset"]))


def grid_search():
    eval_dict = {}
    lr_list = [
        1e-2,
        8e-3,
        6e-3,
        4e-3,
        2e-3,
        1e-3,
        8e-4,
        6e-4,
        4e-4,
        2e-4,
        1e-4,
        8e-5,
        4e-5,
        2e-5,
        1e-5,
    ]

    alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    cur_config = trainer_clfu_config
    cur_config["save_id"] = trainer_clfu_config["save_id"] * 100

    for lr in lr_list:
        cur_config["lr"] = lr
        for alpha in alpha_list:
            print("save_id: {}".format(cur_config["save_id"]))
            model = CLFUModel(clfu_config)

            gnn_cls_model = GNNPreModel(gnn_cls_config)
            trainer_graph = Trainer(gnn_cls_model, trainer_gnn_config)

            seq_cls_model = SeqPreModel(seq_cls_config)
            trainer_seq = Trainer(seq_cls_model, trainer_seq_config)

            trainer = LateFusion(
                trainer_graph,
                trainer_seq,
                model,
                cur_config,
                train_dataset,
                valid_dataset,
                test_dataset,
            )

            trainer.set_alpha(alpha)
            print("cur lr is: {}, cur alpha is:{}".format(lr, alpha))
            trainer.train()
            auc_mean, auc_std = trainer.eval_model()
            print(
                "cur lr is: {}, cur alpha is:{}, auc_mean:{}, auc_std:{}".format(
                    lr, alpha, auc_mean, auc_std
                )
            )

            eval_dict[cur_config["save_id"]] = (lr, alpha, auc_mean, auc_std)
            cur_config["save_id"] = cur_config["save_id"] + 1

    for key in eval_dict.keys():
        print(key, eval_dict[key])


def train_gnn_model():
    eval_dict = {}
    cur_gnn_config = trainer_gnn_config
    cur_gnn_config["save_id"] = cur_gnn_config["save_id"] * 100

    lr_list = [
        1e-2,
        8e-3,
        6e-3,
        4e-3,
        2e-3,
        1e-3,
        8e-4,
        6e-4,
        4e-4,
        2e-4,
        1e-4,
        8e-5,
        6e-5,
        4e-5,
        2e-5,
        1e-5,
    ]

    for cur_lr in lr_list:
        gnn_cls_model = GNNPreModel(gnn_cls_config)
        cur_gnn_config["lr"] = cur_lr
        print("cur lr is {:.6f}".format(cur_lr))
        trainer = Trainer(
            gnn_cls_model, cur_gnn_config, train_dataset, valid_dataset, test_dataset
        )
        trainer.train()
        auc_mean, auc_std = trainer.eval_model()
        print(
            "cur lr: {} ; auc mean: {}; auc std: {}".format(
                cur_lr, auc_mean, auc_std)
        )
        eval_dict[cur_gnn_config["save_id"]] = (cur_lr, auc_mean, auc_std)
        cur_gnn_config["save_id"] = cur_gnn_config["save_id"] + 1

    for key in eval_dict.keys():
        print(key, eval_dict[key])


def train_seq_model():
    eval_dict = {}
    cur_seq_config = trainer_seq_config
    cur_seq_config["save_id"] = cur_seq_config["save_id"] * 100

    lr_list = [
        1e-2,
        8e-3,
        6e-3,
        4e-3,
        2e-3,
        1e-3,
        8e-4,
        6e-4,
        4e-4,
        2e-4,
        1e-4,
        8e-5,
        6e-5,
        4e-5,
        2e-5,
        1e-5,
    ]

    for cur_lr in lr_list:
        seq_cls_model = SeqPreModel(seq_cls_config)
        cur_seq_config["lr"] = cur_lr
        print("cur lr is {:.6f}".format(cur_lr))
        trainer = Trainer(
            seq_cls_model, cur_seq_config, train_dataset, valid_dataset, test_dataset
        )
        trainer.train()

        auc_mean, auc_std = trainer.eval_model()
        print("cur lr: {} auc mean: {} auc std: {}".format(
            cur_lr, auc_mean, auc_std))
        eval_dict[cur_seq_config["save_id"]] = (cur_lr, auc_mean, auc_std)
        cur_seq_config["save_id"] = cur_seq_config["save_id"] + 1

    for key in eval_dict.keys():
        print(key, eval_dict[key])


if __name__ == "__main__":
    # step 1: train gnn model, and select the best model to use in step 3
    # you should comment the other steps code before run step 1
    # train_gnn_model()

    # step 2: train seq model, and select the best model to use in step 3
    # you should comment the other steps code before run step 2
    # train_seq_model()

    # step 3: train late fusion model, use the pretrained gnn and seq model in step 1 and step 2
    # fill the best ckpt id in the config file for fusion model
    # you should comment the other steps code before run step 3
    grid_search()
