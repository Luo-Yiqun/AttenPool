"""
@ Yiqun Luo (luo2@andrew.cmu.edu)

Implementation of MEGNet class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from layers import SoftPlus2, MEGNetBlock, TransformerBlock
from torch_geometric.nn import Set2Set

POOLING_LIST = ("global_mean_pool", "global_add_pool", "global_max_pool", "TransformerBlock", "Set2Set")


class MEGNet(torch.nn.Module):
    """MatErials Graph Network

    A model for predicting crystal and molecular properties using GraphNetworks.

    Example
    -------
    >>> import numpy as np
    >>> from torch_geometric.data import Batch
    >>> from deepchem.feat import GraphData
    >>> n_nodes, n_node_features = 5, 10
    >>> n_edges, n_edge_attrs = 5, 2
    >>> n_global_features = 4
    >>> node_features = np.random.randn(n_nodes, n_node_features)
    >>> edge_attrs = np.random.randn(n_edges, n_edge_attrs)
    >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    >>> global_features = np.random.randn(1, n_global_features)
    >>> graph = GraphData(node_features, edge_index, edge_attrs, global_features=global_features)
    >>> batch = Batch()
    >>> batch = batch.from_data_list([graph.to_pyg_graph()])
    >>> model = MEGNet(n_node_features=n_node_features, n_edge_features=n_edge_attrs, n_global_features=n_global_features)
    >>> pred = model(batch)

    Note
    ----
    This class requires torch-geometric to be installed.
    """

    def __init__(self,
                 n_node_features: int = 32,
                 n_edge_features: int = 32,
                 n_global_features: int = 32,
                 n1 : int = 64,
                 n2 : int = 32,
                 n3 : int = 16,
                 n_blocks: int = 2,
                 is_undirected: bool = False,
                 residual_connection: bool = True,
                 pooling: str = "TransformerBlock",
                 mode: str = 'regression',
                 n_classes: int = 2,
                 n_tasks: int = 1):
        """

        Parameters
        ----------
        n_node_features: int
            Number of features in a node
        n_edge_features: int
            Number of features in a edge
        n_global_features: int
            Number of global features
        n1: int
            Number of hidden neurons for all nodes, edge, and global features
            inside a block
        n2: int
            Number of features for all nodes, edges, and global features
            between blocks
        n3: int
            Number of neurons in the last hidden layer
        n_blocks: int
            Number of GraphNetworks block to use in update
        is_undirected: bool, optional (default True)
            True when the graph is undirected graph , otherwise False
            Not sure how MEGNet is implementing this, but the undirected nature is already considered when constructing the dataset
        residual_connection: bool, optional (default True)
            If True, the layer uses a residual connection during training
        n_tasks: int, default 1
            The number of tasks
        pooling: str
            The pooling method, must be one of ("global_mean_pool", "global_add_pool", "global_max_pool", "TransformerBlock", "Set2Set")
        mode: str, default 'regression'
            The model type - classification or regression
        n_classes: int, default 2
            The number of classes to predict (used only in classification mode).
        """
        super(MEGNet, self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_global_features = n_global_features
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n_tasks = n_tasks
        self.pooling = pooling
        self.mode = mode
        self.n_classes = n_classes

        self.ff_edge = nn.Sequential(
            nn.Linear(self.n_edge_features, n1),
            SoftPlus2(),
            nn.Linear(n1, n2),
            SoftPlus2()
        )
        self.ff_node = nn.Sequential(
            nn.Linear(self.n_node_features, n1),
            SoftPlus2(),
            nn.Linear(n1, n2),
            SoftPlus2()
        )
        self.ff_global = nn.Sequential(
            nn.Linear(self.n_global_features, n1),
            SoftPlus2(),
            nn.Linear(n1, n2),
            SoftPlus2()
        )

        self.megnet_blocks = nn.ModuleList()
        self.n_blocks = n_blocks
        for i in range(n_blocks):
            self.megnet_blocks.append(
                MEGNetBlock(n1, n2,
                   is_undirected=is_undirected,
                   residual_connection=residual_connection))


        assert self.pooling in ("global_mean_pool", "global_add_pool", "global_max_pool", "TransformerBlock", "Set2Set")
        if self.pooling == "TransformerBlock":
            self.pooling_nodes = TransformerBlock(n_2 = self.n2, num_heads = 4)
            self.pooling_edges = TransformerBlock(n_2 = self.n2, num_heads = 4)
            in_size = 3 * n2
        elif self.pooling == "Set2Set":
            self.set2set_nodes = Set2Set(in_channels = n2, processing_steps = 3, num_layers = 1)
            self.set2set_edges = Set2Set(in_channels = n2, processing_steps = 3, num_layers = 1)
            in_size = 5 * n2
        else:
            self.poolings = getattr(torch_geometric.nn, self.pooling)
            in_size = 3 * n2

        self.dense = nn.Sequential(
            nn.Linear(in_size, n2),
            SoftPlus2(),
            nn.Linear(n2, n3),
            SoftPlus2()
            )

        if self.mode == 'regression':
            self.out = nn.Linear(n3, n_tasks)
        elif self.mode == 'classification':
            self.out = nn.Linear(n3, n_tasks * n_classes)

    def forward(self, pyg_batch):
        """
        Parameters
        ----------
        pyg_batch: torch_geometric.data.Batch
            A pytorch-geometric batch of graphs where node attributes are stores
            as pyg_batch['x'], edge_index in pyg_batch['edge_index'], edge features
            in pyg_batch['edge_attr'], global features in pyg_batch['global_features']

        Returns
        -------
        torch.Tensor: Predictions for the graph
        """
        node_features = pyg_batch['x']
        edge_index, edge_features = pyg_batch['edge_index'], pyg_batch[
            'edge_attr']
        global_features = pyg_batch['global_features']
        batch = pyg_batch['batch']

        edge_features = self.ff_edge(edge_features)
        node_features = self.ff_node(node_features)
        global_features = self.ff_global(global_features)

        for i in range(self.n_blocks):
            node_features, edge_features, global_features = self.megnet_blocks[
                i](node_features, edge_index, edge_features, global_features,
                   batch)

        if self.pooling == "TransformerBlock":
            node_out = self.pooling_nodes(global_features, node_features, batch)
            edge_out = self.pooling_edges(global_features, edge_features, batch)
            out = torch.cat([node_out, edge_out, global_features], axis = 1)
        elif self.pooling == "Set2Set":
            node_features = self.set2set_nodes(node_features, batch)
            edge_features = self.set2set_edges(edge_features, batch[edge_index[0]])
            out = torch.cat([node_features, edge_features, global_features], axis = 1)
        else:
            node_out = self.poolings(node_features, batch)
            edge_out = self.poolings(edge_features, batch[edge_index[0]])
            out = torch.cat([node_out, edge_out, global_features], axis = 1)
            
        out = self.out(self.dense(out))

        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits = out.view(-1, self.n_classes)
                softmax_dim = 1
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
                softmax_dim = 2
            proba = F.softmax(logits, dim=softmax_dim)
            return proba, logits
        elif self.mode == 'regression':
            return out


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data
    n_nodes, n_node_features = 7, 10
    n_edges, n_edge_attrs = 5, 2
    n_global_features = 4

    data_list = []
    for _ in range(2):
        node_features = torch.randn(n_nodes, n_node_features)
        edge_attrs = torch.randn(n_edges, n_edge_attrs)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
        global_features = torch.randn(1, n_global_features)
        graph = Data(x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attrs,
                global_features=global_features
                )
        data_list.append(graph)
    
    batch = Batch()
    batch = batch.from_data_list(data_list)
    model = MEGNet(n_node_features=n_node_features, n_edge_features=n_edge_attrs, n_global_features=n_global_features)
    pred = model(batch)
    print(pred)