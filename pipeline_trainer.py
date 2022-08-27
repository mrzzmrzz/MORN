import os
import os.path as osp
import torch
from trainer import Trainer
import numpy as np
from sklearn.metrics import roc_auc_score


class LateFusion(Trainer):
    def __init__(
        self,
        trainer_graph: Trainer,
        trainer_seq: Trainer,
        model,
        config,
        train_data,
        valid_data,
        test_data,
    ):
        super(LateFusion, self).__init__(
            model, config, train_data, valid_data, test_data
        )

        self.alpha = None
        de_novo_train_graph = config["de_novo_train_graph"]
        de_novo_train_seq = config["de_novo_train_seq"]
        ckpt_path = "ckpt"

        print("load the best encoder for graph and seq")
        if de_novo_train_graph:
            trainer_graph.train()
            trainer_graph.load_best_ckpt()
            trainer_graph.eval_model()
        else:
            ckpt_graph_dir = "ckpt_{}_ID_{}_{}".format(
                self.dataset, trainer_graph.config["save_id"], trainer_graph.model_type
            )

            saved_ckpt_graph_path = osp.join(ckpt_path, ckpt_graph_dir, "best")
            best_ckpt_file = os.listdir(saved_ckpt_graph_path)[0]
            best_ckpt_file_path = osp.join(
                saved_ckpt_graph_path, best_ckpt_file)
            trainer_graph.load_ckpt(best_ckpt_file_path)
            print("#" * 50, "GNN Model Has Been Loaded!", "#" * 50, end="\n")
        self.graph_encoder = trainer_graph.model.gnnModel

        if de_novo_train_seq:
            trainer_seq.train()
            trainer_seq.load_best_ckpt()
            trainer_seq.eval_model()
        else:
            ckpt_seq_dir = "ckpt_{}_ID_{}_{}".format(
                self.dataset, trainer_seq.config["save_id"], trainer_seq.model_type
            )

            saved_ckpt_seq_path = osp.join(ckpt_path, ckpt_seq_dir, "best")
            best_ckpt_file = os.listdir(saved_ckpt_seq_path)[0]
            best_ckpt_file_path = osp.join(saved_ckpt_seq_path, best_ckpt_file)
            trainer_seq.load_ckpt(best_ckpt_file_path)
            print("#" * 50, "Seq Model Has Been Loaded!", "#" * 50, end="\n")
        self.seq_encoder = trainer_seq.model.seqModel

    def set_alpha(self, alpha):
        self.alpha = alpha

    def train_iterations(self):
        # switch to the train mode
        self.model.train()

        losses = []
        for i, batch in enumerate(self.train_dataloader):
            batch = batch.to(self.device)
            graph_repr = self.graph_encoder(batch).detach().clone()
            seq_repr = self.seq_encoder(
                batch.seq_feat, batch.seq_len).detach().clone()

            sim_loss, output = self.model.forward(graph_repr, seq_repr)
            label = batch.label.view(output.shape).type(torch.float64)

            is_valid = label**2 > 0

            cls_loss_mat = self.criterion(output, (label + 1) / 2)

            cls_loss_mat = torch.where(
                is_valid,
                cls_loss_mat,
                torch.zeros(cls_loss_mat.shape)
                .to(cls_loss_mat.device)
                .to(cls_loss_mat.dtype),
            )
            cls_loss = torch.sum(cls_loss_mat) / torch.sum(is_valid)

            self.optimizer.zero_grad()
            loss = self.alpha * sim_loss + cls_loss
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            if i % 5 == 0:
                print(
                    "batch:{} cur_loss:{} ave_loss:{} sim_loss:{}, cls_loss:{}".format(
                        i,
                        loss.item(),
                        np.array(losses).mean(),
                        sim_loss.item(),
                        cls_loss.item(),
                    )
                )

        trn_loss = np.array(losses).mean()
        self.lr_scheduler.step()
        return trn_loss

    def valid_iterations(self, epoch, mode="valid", verbose=True, binary_class=True):
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
        sim_losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                graph_repr = self.graph_encoder(batch).detach().clone()
                seq_repr = (
                    self.seq_encoder(
                        batch.seq_feat, batch.seq_len).detach().clone()
                )

                sim_loss, output = self.model(graph_repr, seq_repr)
                label = batch.label.view(output.shape).type(torch.float64)
                sim_losses.append(sim_loss.item())
                outputs.append(output)
                labels.append(label.to(self.device))

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
                torch.zeros(loss_mat.shape).to(
                    loss_mat.device).to(loss_mat.dtype),
            )
            loss_cls = torch.sum(loss_mat) / torch.sum(batch_is_valid)

            ave_sim_loss = np.array(sim_losses).mean()
            loss = self.alpha * ave_sim_loss + loss_cls.item()

        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        roc_list = []
        for i in range(labels.shape[1]):
            if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == -1) > 0:
                is_valid = labels[:, i] ** 2 > 0
                roc_list.append(
                    roc_auc_score(
                        (labels[is_valid, i] + 1) / 2, outputs[is_valid, i])
                )

        if len(roc_list) < labels.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %
                  (1 - float(len(roc_list)) / labels.shape[1]))

        auc_score = sum(roc_list) / len(roc_list)  # y_true.shape[1]
        if verbose:
            print(
                "Epoch: {} total_loss: {:.4f}, cls_loss: {:.4f}, sim_loss: {:.4f},auc:{:.4f}".format(
                    epoch, loss, loss_cls, ave_sim_loss, auc_score
                )
            )

        return [epoch, loss, auc_score]
