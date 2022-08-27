from ast import main
from pretrain_multiprocessing import main_process
from torch_geometric.data import InMemoryDataset
from utils.compound_tools import *
from data import MultiGraph
from pickle import load
from dataset_load import *


class RawMoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        seed,
        mode="train",
        dataset="bace",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        self.dataset = dataset
        self.root = root
        self.seed = seed
        self.mode = mode
        self.weight = 0
        super(RawMoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.transform, self.pre_transform, self.pre_filter = (
            transform,
            pre_transform,
            pre_filter,
        )

        if self.mode == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == "valid":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ["{}.csv".format(self.dataset)]

    @property
    def processed_file_names(self):
        return [
            "{}_train_{}.pt".format(self.dataset, self.seed),
            "{}_val_{}.pt".format(self.dataset, self.seed),
            "{}_test_{}.pt".format(self.dataset, self.seed),
        ]

    def download(self):
        pass

    def process(self):
        path_prefix_name = ["train", "valid", "test"]

        if self.dataset == "bace":
            trn_df, valid_df, test_df = load_bace_dataset(
                self.raw_paths[0], self.seed, random=False
            )
            tasks = ["Class"]
        elif self.dataset == "bbbp":
            trn_df, valid_df, test_df = load_bbbp_dataset(
                self.raw_paths[0], self.seed, random=False
            )
            tasks = ["p_np"]
        elif self.dataset == "hiv":
            trn_df, valid_df, test_df = load_hiv_dataset(
                self.raw_paths[0], self.seed, random=False
            )
            tasks = ["HIV_active"]
        elif self.dataset == "clintox":
            trn_df, valid_df, test_df = load_clintox_dataset(
                self.raw_paths[0], self.seed
            )
            tasks = ["FDA_APPROVED", "CT_TOX"]
        elif self.dataset == "tox21":
            trn_df, valid_df, test_df = load_tox21_dataset(
                self.raw_paths[0], self.seed)
            tasks = [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ]
        elif self.dataset == "sider":
            trn_df, valid_df, test_df = load_sider_dataset(
                self.raw_paths[0], self.seed)
            tasks = [
                "SIDER1",
                "SIDER2",
                "SIDER3",
                "SIDER4",
                "SIDER5",
                "SIDER6",
                "SIDER7",
                "SIDER8",
                "SIDER9",
                "SIDER10",
                "SIDER11",
                "SIDER12",
                "SIDER13",
                "SIDER14",
                "SIDER15",
                "SIDER16",
                "SIDER17",
                "SIDER18",
                "SIDER19",
                "SIDER20",
                "SIDER21",
                "SIDER22",
                "SIDER23",
                "SIDER24",
                "SIDER25",
                "SIDER26",
                "SIDER27",
            ]
        elif self.dataset == "toxcast":
            trn_df, valid_df, test_df = load_toxcast_dataset(
                self.raw_paths[0], self.seed
            )
            tasks = list(trn_df.columns)[2:]
        elif self.dataset == "iron":
            trn_df, valid_df, test_df = load_iron_dataset(
                self.raw_paths[0], self.seed)
            tasks = ["iron_reactive"]
        elif self.dataset == "muv":
            trn_df, valid_df, test_df = load_muv_dataset(
                self.raw_paths[0], self.seed)
            tasks = [
                "MUV-466",
                "MUV-548",
                "MUV-600",
                "MUV-644",
                "MUV-652",
                "MUV-689",
                "MUV-692",
                "MUV-712",
                "MUV-713",
                "MUV-733",
                "MUV-737",
                "MUV-810",
                "MUV-832",
                "MUV-846",
                "MUV-852",
                "MUV-858",
                "MUV-859",
            ]

        else:
            raise ValueError("Not Supported Dataset")

        for i, dataframe in enumerate([trn_df, valid_df, test_df]):
            data_list = []
            label_list = dataframe[tasks].to_numpy()

            cur_graph_feat_path = os.path.join(
                os.getcwd(),
                "molattrs/saved_graph_feat",
                self.dataset,
                path_prefix_name[i],
                "graph_feat_{}.pt".format(self.seed),
            )

            cur_seq_feat_path = os.path.join(
                os.getcwd(),
                "molattrs/saved_seq_feat",
                self.dataset,
                path_prefix_name[i],
                "seq_feat_{}.pt".format(self.seed),
            )

            if not (
                os.path.exists(cur_graph_feat_path)
                and os.path.exists(cur_seq_feat_path)
            ):
                main_process(
                    self.raw_paths[0],
                    process_number=30,
                    seed_num=self.seed,
                    dataset=self.dataset,
                )

            with open(cur_graph_feat_path, "rb") as f:
                cur_raw_data_list = load(f)
            with open(cur_seq_feat_path, "rb") as f:
                cur_seq_data_list = load(f)

            for j in tqdm(range(len(label_list))):
                # transform the Smiles data into the graph data
                # raw_data = mol_to_geognn_graph_data_MMFF3d(rdkit_mol_list[j])

                raw_data = cur_raw_data_list[j]

                # created for the atom bond graph
                _node_attr_a_b_g = atom_attr(raw_data)
                _edge_index_a_b_g = torch.tensor(raw_data["edges"]).T
                _edge_attr_a_b_g = edge_attr(raw_data)

                # created for the bond angle graph
                # _node_attr_b_a_g = torch.ones([_node_attr_a_b_g.shape[0], 1])
                _node_attr_b_a_g = torch.ones([_edge_index_a_b_g.shape[1], 1])

                # just for the placeholder
                _edge_index_b_a_g = torch.tensor(
                    raw_data["BondAngleGraph_edges"]).T
                _edge_attr_b_a_g = edge_attr(
                    raw_data, edge_attr_list=["bond_angle"])

                # get the other data attr for pic and seq tasks
                _seq_feat, _seq_len = cur_seq_data_list[j]

                # the true label of this data

                _y = torch.tensor(label_list[j], dtype=torch.float64)

                data = MultiGraph(
                    x_a=_node_attr_a_b_g,
                    edge_index_a=_edge_index_a_b_g,
                    edge_attr_a=_edge_attr_a_b_g,
                    x_b=_node_attr_b_a_g,
                    edge_index_b=_edge_index_b_a_g,
                    edge_attr_b=_edge_attr_b_a_g,
                    seq_feat=_seq_feat,
                    seq_len=_seq_len,
                    label=_y,
                )

                data_list.append(data)

            # begin to save the split dataset
            data, slices = self.collate(data_list)

            torch.save((data, slices), self.processed_paths[i])


if __name__ == "__main__":
    cur_path = os.getcwd()
    dataset = "iron"
    root_path = cur_path + "/dataset/{}/".format(dataset)
    dataset = RawMoleculeDataset(
        root=root_path, seed=100, mode="train", dataset=dataset)
    print(dataset[0])
    print(len(dataset))
