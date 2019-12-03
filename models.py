import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj,labels):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.elu(self.out_att(x, adj))
# maybe instead of softmax normalize for a particular document
        x = F.softmax(x,dim = 1)

        nclass = int(labels.max()) + 1
        nnodes = len(x)
        nfeat = len(x[0])
        feature_freq = np.zeros((nfeat,nclass))

        for i in range(nnodes):
            for j in range(nfeat):
                feature_freq[j][labels[i]] = feature_freq[j][labels[i]] + x[i][j]

        max_freq = np.amax(feature_freq,axis = 0)
        rel_deg = np.zeros((nfeat,nclass))
        for i in range(nfeat):
            for j in range(nclass):
                rel_deg[i][j] = feature_freq[i][j]/max_freq[j]

        V_parsed_text = np.zeros((nnodes,nfeat,nclass))
        for i in range(nnodes):
            for j in range(nfeat):
                for k in range(nclass):
                    V_parsed_text[i][j][k] = x[i][j]*rel_deg[j][k]

        avg_vpt = np.zeros((nnodes,nclass))

        for i in range(nnodes):
            for j in range(nclass):
                for k in range(nfeat):
                    avg_vpt[i][j] = avg_vpt[i][j] + V_parsed_text[i][k][j]
                avg_vpt[i][j] = avg_vpt[i][j]/nfeat

        print("vpt")
        print(avg_vpt)

        avg_rel_deg = np.zeros(nclass)

        for i in range(nclass):
            for j in range(nfeat):
                avg_rel_deg[i] = avg_rel_deg[i]+rel_deg[j][i]
            avg_rel_deg[i] = avg_rel_deg[i]/nfeat

        print("avgrel")
        print(avg_rel_deg)

        similarity = np.zeros((nnodes,nclass))

        for i in range(nnodes):
            for j in range(nclass):
                similarity[i][j] = avg_vpt[i][j]/avg_rel_deg[j]

        print("similarity")
        print(similarity)
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
