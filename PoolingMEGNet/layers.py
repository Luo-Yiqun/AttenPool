# Yiqun Luo (luo2@andrew.cmu.edu)

import torch
from torch import nn
from torch import Tensor
from typing import Any, Tuple, Optional
from torch_geometric.utils import scatter
import math


class SoftPlus2(nn.Module):
    """
    Activation function used in MEGNet

    SoftPlus2(x) = ln((e^x + 1)/2), which is a modified version of the softplus
    activation function to make function to be 0 at x = 0
    """
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x) - torch.log(torch.tensor(2))


class MEGNetBlock(nn.Module):
    """
    MEGNet block

    A MEGNet block takes a graph as input and returns an updated graph
    as output. The output graph has same structure as input graph but it
    has updated node features, edge features and global state features.

    Here, the input and output features are n2 for all nodes, edges, and global
    features, and the hidden layer dimensions are all n1, as used in the MEGNet
    github

    Parameters
    ----------
    n1: int
        hidden neuron numbers
    n2: int
        input/output features numbers
    is_undirected: bool, optional (default True)
        Directed or undirected graph
    residual_connection: bool, optional (default True)
        If True, the layer uses a residual connection during training

    Example
    -------
    >>> import torch
    >>> n_nodes = 5
    >>> n_edges = 5
    >>> n_features = 32
    >>> node_features = torch.randn(n_nodes, n_features)
    >>> edge_features = torch.randn(n_edges, n_features)
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
    >>> global_features = torch.randn(1, n_features)
    >>> gn = MEGNetBlock()
    >>> node_features, edge_features, global_features = gn(node_features, edge_index, edge_features, global_features)
    """

    def __init__(
            self,
            n1 : int = 64,
            n2 : int = 32,
            is_undirected : bool = True,
            residual_connection : bool = True
    ):
        super().__init__()
        self.is_undirected = is_undirected
        self.residual_connection = residual_connection
        self.edge_dense = nn.Sequential(
            nn.Linear(n2, n1),
            SoftPlus2(),
            nn.Linear(n1, n2),
            SoftPlus2()
        )
        self.node_dense = nn.Sequential(
            nn.Linear(n2, n1),
            SoftPlus2(),
            nn.Linear(n1, n2),
            SoftPlus2()
        )
        self.global_dense = nn.Sequential(
            nn.Linear(n2, n1),
            SoftPlus2(),
            nn.Linear(n1, n2),
            SoftPlus2()
        )
        self.edge_MEGNet = nn.Linear(4 * n2, n2)
        self.node_MEGNet = nn.Linear(3 * n2, n2)
        self.global_MEGNet = nn.Linear(3 * n2, n2)

    def reset_parameters(self) -> None:
        for i in range(0, len(self.edge_dense)):
            self.edge_dense[i].reset_parameters()
        for i in range(0, len(self.node_dense)):
            self.node_dense[i].reset_parameters()
        for i in range(0, len(self.global_dense)):
            self.global_dense[i].reset_parameters()
        self.edge_MEGNet.reset_parameters()
        self.node_MEGNet.reset_parameters()
        self.global_MEGNet.reset_parameters()
        return
        
    def _update_edge_features(self, node_features, edge_index, edge_features,
                              global_features, batch):
        src_index, dst_index = edge_index
        out = torch.cat((
            node_features[src_index], 
            node_features[dst_index], 
            edge_features, 
            global_features[batch]
            ), dim=1)
        return self.edge_MEGNet(out)

    def _update_node_features(self, node_features, edge_index, edge_features,
                              global_features, batch):

        src_index, dst_index = edge_index
        # Compute mean edge features for each node by dst_index (each node
        # receives information from edges which have that node as its destination,
        # hence the computation uses dst_index to aggregate information)
        edge_features_mean_by_node = scatter(src=edge_features,
                                             index=dst_index,
                                             dim=0,
                                             dim_size = len(node_features),
                                             reduce='mean')
        out = torch.cat(
            (node_features, edge_features_mean_by_node, global_features[batch]),
            dim=1)
        return self.node_MEGNet(out)

    def _update_global_features(self, node_features, edge_features,
                                global_features, node_batch_map,
                                edge_batch_map):
        edge_features_mean = scatter(src=edge_features,
                                     index=edge_batch_map,
                                     dim=0,
                                     reduce='mean')
        node_features_mean = scatter(src=node_features,
                                     index=node_batch_map,
                                     dim=0,
                                     reduce='mean')
        out = torch.cat(
            (edge_features_mean, node_features_mean, global_features), dim=1)
        return self.global_MEGNet(out)

    def forward(
            self,
            node_features: Tensor,
            edge_index: Tensor,
            edge_features: Tensor,
            global_features: Tensor,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Output computation for a GraphNetwork

        Parameters
        ----------
        node_features: torch.Tensor
            Input node features of shape :math:`(|\mathcal{V}|, n2)`
        edge_index: torch.Tensor
            Edge indexes of shape :math:`(2, |\mathcal{E}|)`
        edge_features: torch.Tensor
            Edge features of the graph, shape: :math:`(|\mathcal{E}|, n2)`
        global_features: torch.Tensor
            Global features of the graph, shape: :math:`(batch_size, n2)`
        where :math:`|\mathcal{V}|` and :math:`|\mathcal{E}|` denotes the
            number of nodes and edges in all graphs.
        batch: torch.LongTensor (optional, default: None)
            A vector that maps each node to its respective graph identifier.
            The attribute is used only when more than one graph are batched
            together during a single forward pass.
        """
        if batch is None:
            batch = node_features.new_zeros(node_features.size(0),
                                            dtype=torch.int64)

        if self.residual_connection:
            node_features_copy, edge_features_copy, global_features_copy = node_features, edge_features, global_features

        edge_features = self.edge_dense(edge_features)
        node_features = self.node_dense(node_features)
        global_features = self.global_dense(global_features)

        if self.is_undirected is True:
            # holding bi-directional edges in case of undirected graphs
            edge_index = torch.cat((edge_index, edge_index.flip([0])), dim=1)
            edge_features_len = edge_features.shape[0]
            edge_features = torch.cat((edge_features, edge_features), dim=0)

        edge_batch_map = batch[edge_index[0]]
        edge_features = self._update_edge_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features,
                                                   edge_batch_map)
        node_features = self._update_node_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features, batch)
        global_features = self._update_global_features(node_features,
                                                       edge_features,
                                                       global_features, batch,
                                                       edge_batch_map)

        if self.is_undirected is True:
            # coonverting edge features to its original shape
            split = torch.split(edge_features,
                                [edge_features_len, edge_features_len])
            edge_features = (split[0] + split[1]) / 2

        if self.residual_connection:
            edge_features += edge_features_copy
            node_features += node_features_copy
            global_features += global_features_copy

        return node_features, edge_features, global_features
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_2, num_heads = 1):
        super(MultiHeadAttention, self).__init__()
        assert n_2 % num_heads == 0

        self.n_2 = n_2
        self.num_heads = num_heads
        self.d_k = n_2 // num_heads

        self.query = nn.Linear(n_2, n_2)
        self.key = nn.Linear(n_2, n_2)
        self.value = nn.Linear(n_2, n_2)
        self.out = nn.Linear(n_2, n_2)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query, key, value, mask = None):
        # Linear projections
        query = self.query(query).view(-1, self.num_heads, self.d_k).transpose(0, 1)
        key = self.key(key).view(-1, self.num_heads, self.d_k).transpose(0, 1)
        value = self.value(value).view(-1, self.num_heads, self.d_k).transpose(0, 1)

        # Apply attention
        x, attn = self.attention(query, key, value, mask)

        # Concatenate heads
        x = x.transpose(0, 1).contiguous().view(-1, self.n_2)

        # Final linear layer
        x = self.out(x)

        return x, attn


class TransformerBlock(nn.Module):
    def __init__(self, n_2, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(n_2, num_heads)
        self.norm1 = nn.LayerNorm(n_2)
        self.norm2 = nn.LayerNorm(n_2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, global_features, local_features, batch, mask = None):
        assert len(torch.unique(batch)) == 1, "The current transformer block only applies to a single graph"
        attn_output, _ = self.attention(global_features, local_features, local_features, mask)
        global_features = self.norm1(global_features + self.dropout(attn_output))
        return global_features


if __name__ == "__main__":
    n_nodes = 5
    n_edges = 5
    n_features = 32
    node_features = torch.randn(n_nodes, n_features)
    edge_features = torch.randn(n_edges, n_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
    global_features = torch.randn(1, n_features)
    gn = MEGNetBlock()
    node_features, edge_features, global_features = gn(node_features, edge_index, edge_features, global_features)
