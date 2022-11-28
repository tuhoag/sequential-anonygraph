import numpy as np
import torch

import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)

        # print(h)
        h = {k: F.relu(v) for k, v in h.items()}

        # print(h)
        h = self.conv2(graph, h)

        # print(h)
        return h
