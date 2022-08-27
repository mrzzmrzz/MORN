import os
import os.path as osp
import warnings
import numpy as np
import pandas as pd
import torch
import shutil
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def seed_set(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def evaluation(pred, label):
    accuracy = accuracy_score(label, pred)
    macro_precision = precision_score(label, pred)
    macro_recall = recall_score(label, pred)
    macro_f1 = f1_score(label, pred, average="macro")
    micro_f1 = f1_score(label, pred, average="micro")
    return accuracy, macro_precision, macro_recall, macro_f1, micro_f1


def evaluation_auc(pred, label):
    fpr, tpr, _ = roc_curve(label, pred[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr


class Trainer:
    def __init__(self, model, config, train_data=None, valid_data=None, test_data=None):

        self.config = config
        self.dataset = config["dataset"]
        self.epoch = config["epoch"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.seed = config["seed"]
        self.early_stop_patience = config["early_stop_patience"]
        self.model_type = config["model_type"]
        self.save_id = self.config["save_id"]

        print(self.config)

        self.device = torch.device(
            "cuda:{}".format(self.config["cuda_device"])
            if torch.cuda.is_available() and self.config["cuda_device"] >= 0
            else "cpu"
        )

        self.model = model.to(self.device)

        # dataloader
        if train_data:
            self.train_dataloader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                follow_batch=["x_a", "x_b"],
                shuffle=True,
            )
        if valid_data:
            self.valid_dataloader = DataLoader(
                valid_data,
                batch_size=self.batch_size,
                follow_batch=["x_a", "x_b"],
                shuffle=True,
            )
        if test_data:
            self.test_dataloader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                follow_batch=["x_a", "x_b"],
                shuffle=True,
            )

        # train criterion and optimizer
        # self.criterion = nn.CrossEntropyLoss()

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=[25, 50, 80, 120, 160, 220, 300], gamma=0.7)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.8
        )

        seed_set(self.seed)
        print("*" * 100, "Random Seed Has Been Set!!!", "*" * 100, end="\n")

        # add TensorBoard Support
        self.writer = SummaryWriter(
            comment="{}_EP{}_BS{}\
            _LR{}_WD{}_SD{}_ID{}_{}".format(
                self.dataset,
                self.epoch,
                self.batch_size,
                self.lr,
                self.weight_decay,
                self.seed,
                self.config["save_id"],
                self.config["comments"],
            )
        )

        # train records files
        self.record = {
            "trn_record": [],
            "valid_record": [],
            "valid_loss": [],
            "valid_auc": [],
            "best_ckpt": "",
        }

        # related file save path
        self.abs_file_dir = osp.dirname(__file__)
        self.record_save_path = osp.join(
            self.abs_file_dir,
            "record",
            "record{}_ID_{}_{}.csv".format(
                self.dataset, self.config["save_id"], self.model_type
            ),
        )

        self.ckpt_save_dir = osp.join(
            self.abs_file_dir,
            "ckpt",
            "ckpt_{}_ID_{}_{}".format(
                self.dataset, self.config["save_id"], self.model_type
            ),
        )

        self.saved_best_ckpt_path = osp.join(self.ckpt_save_dir, "best")

        self.eval_save_path = osp.join(
            self.abs_file_dir,
            "record",
            "eval",
            "data_{}_ID_{}_{}.csv".format(
                self.dataset, self.config["save_id"], self.model_type
            ),
        )

        if not osp.exists(self.ckpt_save_dir):
            os.mkdir(self.ckpt_save_dir)
        if not osp.exists(self.saved_best_ckpt_path):
            os.mkdir(self.saved_best_ckpt_path)

    def train_iterations(self):
        # switch to the train mode
        self.model.train()
        losses = []
        for i, batch in enumerate(self.train_dataloader):
            batch = batch.to(self.device)
            output = self.model(batch)
            label = batch.label.view(output.shape).type(torch.float64)

            # Whether y is non-null or not.
            is_valid = label**2 > 0

            # Loss matrix
            loss_mat = self.criterion(output.double(), (label + 1) / 2)

            # Loss matrix after removing null target
            loss_mat = torch.where(
                is_valid,
                loss_mat,
                torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
            )

            self.optimizer.zero_grad()
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if i % 5 == 0:
                print(
                    "\t batch:{} cur_loss:{} ave_loss:{}".format(
                        i, loss.item(), np.array(losses).mean()
                    )
                )

        # return this batch's mean loss
        trn_loss = np.array(losses).mean()
        self.lr_scheduler.step()
        return trn_loss

    def valid_iterations(self, epoch, mode="valid", verbose=True):
        # switch to the eval mode
        self.model.eval()
        if mode == "test":
            dataloader = self.test_dataloader
        elif mode == "valid":
            dataloader = self.valid_dataloader
        else:
            raise ValueError("Wrong Mode")

        outputs = []
        labels = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                output = self.model(batch)
                label = batch.label.view(output.shape).type(torch.float64)
                outputs.append(output)
                labels.append(label)

            labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs, dim=0)

            # Non-Nan labels
            batch_is_valid = labels**2 > 0

            # Loss matrix
            loss_mat = self.criterion(outputs, (labels + 1) / 2)

            # Loss matrix after removing null target
            loss_mat = torch.where(
                batch_is_valid,
                loss_mat,
                torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
            )

            loss = torch.sum(loss_mat) / torch.sum(batch_is_valid)
            loss = loss.item()

        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        roc_list = []
        # print("labels size {}".format(labels.shape))
        for i in range(labels.shape[1]):
            if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == -1) > 0:
                is_valid = labels[:, i] ** 2 > 0
                roc_list.append(
                    roc_auc_score((labels[is_valid, i] + 1) / 2, outputs[is_valid, i])
                )

        if len(roc_list) < labels.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" % (1 - float(len(roc_list)) / labels.shape[1]))

        auc_score = sum(roc_list) / len(roc_list)  # y_true.shape[1]
        if verbose:
            print("Epoch: {} loss: {:.4f} auc:{:.4f}".format(epoch, loss, auc_score))
        return [epoch, loss, auc_score]

    def train(self, early_stop: bool = True):
        early_stop_cnt = 0
        for epoch in range(self.config["epoch"]):
            print("#" * 100)
            trn_loss = self.train_iterations()
            valid_record = self.valid_iterations(epoch)
            valid_loss = valid_record[1]
            valid_auc = valid_record[2]
            self.record["trn_record"].append([epoch, trn_loss])
            self.record["valid_record"].append([epoch, valid_record])

            # two valid screen metric
            self.record["valid_loss"].append(valid_loss)
            self.record["valid_auc"].append(valid_auc)

            # WriterSummary
            self.writer.add_scalar(
                tag="train/lr",
                scalar_value=self.lr_scheduler.get_last_lr()[0],
                global_step=epoch,
            )
            self.writer.add_scalar(
                tag="train/loss", scalar_value=trn_loss, global_step=epoch
            )
            self.writer.add_scalar(
                tag="test/loss", scalar_value=valid_loss, global_step=epoch
            )
            self.writer.add_scalar(
                tag="test/AUC", scalar_value=valid_auc, global_step=epoch
            )

            # show the best auc up to now
            auc_max = max(self.record["valid_auc"])
            auc_max_index = self.record["valid_auc"].index(auc_max)
            print("Best Auc :{:.5f} At Epoch {}".format(auc_max, auc_max_index))

            # early stop settings
            if early_stop:
                if (
                    valid_auc == auc_max
                    or valid_loss == np.array(self.record["valid_loss"]).mean()
                ):
                    self.save_model_and_record(epoch, trn_loss, valid_loss)
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                if 0 < self.early_stop_patience < early_stop_cnt:
                    print("#" * 80, "Early Stop", "#" * 80)
                    break
            if epoch == self.config["epoch"] - 1:
                self.save_model_and_record(epoch, trn_loss, valid_loss, final_save=True)

        print("The best ckpt is {}".format(self.record["best_ckpt"]))
        cur_best_ckpt_path = osp.join(self.ckpt_save_dir, self.record["best_ckpt"])
        # print(cur_best_ckpt_path)
        shutil.move(cur_best_ckpt_path, self.saved_best_ckpt_path)

    def save_model_and_record(self, epoch, trn_loss, valid_loss, final_save=False):
        if final_save:
            self.save_loss_record()
            print("Train Completely! Model Saved!")
            file_name = "Final_{}_{}_{:.3f}_{:.3f}.ckpt".format(
                self.model_type, epoch, trn_loss, valid_loss
            )
        else:
            file_name = "{}_{}_{:.3f}_{:.3f}.ckpt".format(
                self.model_type, epoch, trn_loss, valid_loss
            )
            print("Best Model Has Been Saved At Epoch {}".format(epoch))
            self.record["best_ckpt"] = file_name

        with open(osp.join(self.ckpt_save_dir, file_name), "wb") as f:
            torch.save(
                {
                    "config": self.config,
                    "record": self.record,
                    "model_state_dict": self.model.state_dict(),
                },
                f,
            )

        # print("model saved at epoch {}".format(epoch))

    def save_loss_record(self):
        trn_record = pd.DataFrame(
            data=self.record["trn_record"], columns=["epoch", "trn_loss"]
        )
        valid_record = pd.DataFrame(
            data=self.record["valid_record"], columns=["epoch", "valid_loss"]
        )
        res = pd.DataFrame(
            {
                "Epoch": trn_record["epoch"],
                "training Loss": trn_record["trn_loss"],
                "validation Loss": valid_record["valid_loss"],
            }
        )
        res.to_csv(self.record_save_path)

    def load_ckpt(self, ckpt_path):
        print("ckpt loading: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        self.config = ckpt["config"]
        self.record = ckpt["record"]
        self.model.load_state_dict(ckpt["model_state_dict"])

    def load_best_ckpt(self):
        print("best ckpt is :{}".format(self.record["best_ckpt"]))
        best_ckpt = osp.join(self.saved_best_ckpt_path, self.record["best_ckpt"])
        self.load_ckpt(best_ckpt)

    def eval_model(self):
        self.load_best_ckpt()
        eval_dict = {"epoch": [], "loss": [], "auc": []}

        for i in range(100):
            ret = self.valid_iterations(i, mode="test", verbose=True)
            eval_dict["epoch"].append(ret[0])
            eval_dict["loss"].append(ret[1])
            eval_dict["auc"].append(ret[2])

        res_dataframe = pd.DataFrame(
            {
                "epoch": eval_dict["epoch"],
                "loss": eval_dict["loss"],
                "auc": eval_dict["auc"],
            }
        )

        res_dataframe.to_csv(self.eval_save_path)
        auc_list = np.array(list(res_dataframe["auc"]))
        auc_mean = auc_list.mean()
        auc_std = auc_list.std()
        return auc_mean, auc_std
