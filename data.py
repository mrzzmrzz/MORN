from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, NodeType, EdgeType
from typing import Any


class MultiGraph(Data):
    def __init__(
        self,
        x_a: OptTensor = None,
        edge_index_a: OptTensor = None,
        edge_attr_a: OptTensor = None,
        x_b: OptTensor = None,
        edge_index_b: OptTensor = None,
        edge_attr_b: OptTensor = None,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.x_a = x_a
        self.edge_index_a = edge_index_a
        self.edge_attr_a = edge_attr_a

        self.x_b = x_b
        self.edge_index_b = edge_index_b
        self.edge_attr_b = edge_attr_b

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_a":
            return self.x_a.size(0)
        if key == "edge_index_b":
            return self.x_b.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "seq_feat":
            return None
        if key == "img_feat":
            return None
        if key == "struct_feat":
            return None
        else:
            return super(MultiGraph, self).__cat_dim__(key, value, *args, **kwargs)
