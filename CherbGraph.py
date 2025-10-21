import torch
import math
import torch.nn.functional as F
import os
import cv2
import time
import torch.nn as nn
import numpy as np
from utils import tensor2img
import model.loss as loss
from torch_scatter import scatter_mean 
from torch.nn import LayerNorm
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn.functional as F
class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, hyperedges=200, heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyperedges = hyperedges
        self.heads = heads
        self.node2edge = nn.Linear(in_channels, out_channels * heads)
        self.edge2node = nn.Linear(out_channels * heads, out_channels)
        self.attention = nn.Parameter(torch.randn(1, heads, out_channels))
        self.residual = nn.Linear(in_channels, out_channels)
        nn.init.kaiming_uniform_(self.node2edge.weight, mode='fan_out')
        nn.init.kaiming_uniform_(self.edge2node.weight, mode='fan_in')
        nn.init.normal_(self.attention, mean=0, std=0.1)
    def forward(self, x, hyperedge_labels):
        B, C, H, W = x.size()
        num_nodes = H * W
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        hyperedge_flat = hyperedge_labels.view(B, -1)  # [B, N]
        outputs = []
        for b in range(B):
            x_b = x_flat[b]  # [N, C]
            edge_labels = hyperedge_flat[b]  # [N]
            unique_edges, edge_indices = torch.unique(
                edge_labels, return_inverse=True, sorted=True
            )
            num_edges = unique_edges.size(0)
            node_features = self.node2edge(x_b)  # [N, C_out*heads]
            node_features = node_features.view(-1, self.heads, self.out_channels)  # [N, heads, C_out]
            attn_logits = (node_features * self.attention).sum(dim=-1)  # [N, heads]
            attn = F.softmax(attn_logits, dim=0)
            edge_features = scatter_mean(
                src=node_features * attn.unsqueeze(-1),  # [N, heads, C_out]
                index=edge_indices,        # [N]
                dim=0, 
                dim_size=num_edges
            )  # [num_edges, heads, C_out]
            edge_features = edge_features.view(num_edges, -1)  # [num_edges, heads*C_out]
            edge_transformed = self.edge2node(edge_features)  # [num_edges, C_out]
            node_output = edge_transformed[edge_indices]
            residual = self.residual(x_b)  # [N, C_out]
            output = F.elu(node_output + residual)  # [N, C_out]
            output = output.view(H, W, self.out_channels).permute(2, 0, 1)  # [C_out, H, W]
            outputs.append(output)
        final = torch.stack(outputs, dim=0)  # [B, C_out, H, W]
        return final
class HyperEdgeGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = HypergraphConv(1, 64, hyperedges=200)
        self.conv2 = HypergraphConv(64, 32, hyperedges=200)
        self.conv3 = HypergraphConv(32, 1, hyperedges=200)
    def forward(self, x, hyperedges):
        x = F.relu(self.conv1(x, hyperedges))
        x = F.relu(self.conv2(x, hyperedges))
        return self.conv3(x, hyperedges)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj):
        # Normalized adjacency matrix A_tilde = D^(-1/2) * (A + I) * D^(-1/2)
        I = torch.eye(adj.size(-1)).to(adj.device)
        A_hat = adj + I
        D_hat = torch.diag_embed(torch.pow(A_hat.sum(dim=-1), -0.5))
        A_norm = torch.bmm(torch.bmm(D_hat, A_hat), D_hat)
        return torch.bmm(A_norm, torch.matmul(x, self.weight))
class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, heads=8):
        super(GraphAttention, self).__init__()
        self.heads = heads
        self.linears = nn.ModuleList([GraphConvolution(in_features, out_features) for _ in range(heads)])
    def forward(self, Q, K, V, adj):
        outputs = [torch.matmul(
                        F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1)**0.5), dim=-1), v)
                   for q, k, v in zip(
                       [linear(Q, adj) for linear in self.linears],
                       [linear(K, adj) for linear in self.linears],
                       [linear(V, adj) for linear in self.linears])]
        return torch.cat(outputs, dim=-1)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
class ChebGC(nn.Module):
    def __init__(self, in_features, K):
        super(ChebGC, self).__init__()
        self.K = K
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, in_features)) for _ in range(K)])
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)
    def forward(self, x, laplacian):
        B, N, F = x.size()
        Tx_0 = x
        Tx_1 = torch.bmm(laplacian, x)
        out = torch.bmm(Tx_0, self.Theta[0].unsqueeze(0).expand(B, -1, -1))  # Expand Theta to be 3D for batch processing
        if self.K > 1:
            out = out + torch.bmm(Tx_1, self.Theta[1].unsqueeze(0).expand(B, -1, -1))
        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(laplacian, Tx_1) - Tx_0
            out = out + torch.bmm(Tx_2, self.Theta[k].unsqueeze(0).expand(B, -1, -1))
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out
class GraformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads=8, cheb_K=3):
        super(GraformerLayer, self).__init__()
        self.ln1 = LayerNorm(d_model)
        self.mhga = GraphAttention(d_model, d_model // heads, heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln3 = LayerNorm(d_model)
        self.chebgc = ChebGC(d_model, cheb_K)

    def forward(self, Z, adj, laplacian):
        # Z: [B, N, F]
        Z_ = self.ln1(Z)
        H = self.mhga(Z_, Z_, Z_, adj)
        Z = Z + H
        Z_ = self.ln2(Z)
        H = self.ffn(Z_)
        Z = Z + H
        Z_ = self.ln3(Z)
        H = self.chebgc(Z_, laplacian)
        Z = Z + H
        return Z