import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected

from Model import GNNEncoder, EdgeDecoder, DegreeDecoder, RPI, Mask

output_dim = 909
SPLITS = ["unseen_pair", "unseen_protein", "unseen_rna"]
METRICS = ["AUC", "AP", "ACC", "SEN", "PRE", "SPE", "F1", "MCC"]


class RPIFileNegatives(RPI):
    def __init__(self, encoder, edge_decoder, degree_decoder, mask, neg_pool):
        super(RPIFileNegatives, self).__init__(encoder, edge_decoder, degree_decoder, mask)
        self.neg_pool = neg_pool

    def train_epoch(self, data, optimizer, alpha, batch_size=8192, grad_norm=1.0):
        x, edge_index = data.x, data.edge_index
        remaining_edges, masked_edges = self.mask(edge_index)
        num_neg = masked_edges.size(1)
        neg_pool = self.neg_pool
        if neg_pool.size(1) < num_neg:
            neg_edges = neg_pool[:, torch.randint(0, neg_pool.size(1), (num_neg,), device=neg_pool.device)]
        else:
            neg_edges = neg_pool[:, torch.randperm(neg_pool.size(1), device=neg_pool.device)[:num_neg]]
        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            z = self.encoder(x, remaining_edges)
            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.edge_decoder(z, batch_masked_edges)
            neg_out = self.edge_decoder(z, batch_neg_edges)
            loss = self.loss_fn(pos_out, neg_out)
            deg = degree(masked_edges[1].flatten(), data.num_nodes).float()
            loss += alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()


def build_model(neg_pool=None):
    encoder = GNNEncoder(in_channels=output_dim, hidden_channels=64, out_channels=128, heads=8)
    edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    if neg_pool is None:
        return RPI(encoder, edge_decoder, degree_decoder, Mask(p=0.4))
    return RPIFileNegatives(encoder, edge_decoder, degree_decoder, Mask(p=0.4), neg_pool)


def load_features(processed_dir, graph):
    rna = np.load(os.path.join(processed_dir, "features", "rna_features.npy"), mmap_mode="r")
    protein = np.load(os.path.join(processed_dir, "features", "protein_features.npy"), mmap_mode="r")
    ntr, ntp, ner, nep = graph["n_train_rna"], graph["n_train_protein"], graph["n_extra_rna"], graph["n_extra_protein"]
    if rna.shape[0] != ntr + ner or protein.shape[0] != ntp + nep:
        raise ValueError(f"Feature rows (RNA {rna.shape[0]}, protein {protein.shape[0]}) do not match graph nodes (RNA {ntr + ner}, protein {ntp + nep})")
    for name, feat in (("RNA", rna), ("protein", protein)):
        if feat.shape[1] > output_dim:
            raise ValueError(f"{name} features have {feat.shape[1]} columns, more than output_dim={output_dim}")
    print(f"RNA features: {rna.shape}, protein features: {protein.shape}")
    x = np.zeros((ntr + ntp + ner + nep, output_dim), dtype=np.float32)
    x[:ntr, :rna.shape[1]] = rna[:ntr]
    x[ntr:ntr + ntp, :protein.shape[1]] = protein[:ntp]
    x[ntr + ntp:ntr + ntp + ner, :rna.shape[1]] = rna[ntr:]
    x[ntr + ntp + ner:, :protein.shape[1]] = protein[ntp:]
    return torch.from_numpy(x)


def load_ciceklab(processed_dir, device):
    graph = torch.load(os.path.join(processed_dir, "data", "graph.pt"))
    feature = load_features(processed_dir, graph).to(device)
    n_train = graph["n_train_rna"] + graph["n_train_protein"]
    train_data = Data(x=feature[:n_train], edge_index=to_undirected(graph["train_pos"]).to(device))
    train_data.num_rna = graph["n_train_rna"]
    train_data.pos_edge_label_index = graph["train_pos"].to(device)
    train_data.neg_edge_label_index = graph["train_neg"].to(device)
    eval_data = {}
    for name, split in graph["eval"].items():
        pairs, labels = split["pairs"], split["labels"]
        eval_data[name] = (pairs[:, labels == 1].to(device), pairs[:, labels == 0].to(device))
    print(f"Nodes: {feature.size(0)} total, {n_train} in the training graph; training edges: {train_data.pos_edge_label_index.size(1)}")
    return feature, train_data, eval_data
