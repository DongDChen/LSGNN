# encoding:utf-8
import math
import numpy as np
import pandas as pd
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from tqdm import tqdm
import os
from openhgnn import models
from functools import partial
import torch as th
from sklearn.cluster import KMeans
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')



class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names, node_num, cluster, n_heads, attn_drop, dropout):
        super().__init__()

        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.dropout = dropout

        self.n_heads = n_heads
        self.cluster = cluster
        self.attn_drop = attn_drop
        residual = True

        self.bn = nn.BatchNorm1d(num_features=self.hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(self.dropout)

        self.attention = FeatureAttentionLayer(self.hidden_dim, self.n_heads, self.cluster, self.attn_drop, residual)

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)

        # self.SHGN = models.SimpleHGN(edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
        #         num_layers, heads, feat_drop, negative_slope,
        #         residual, beta)

        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h_dict):

        # h = g.ndata['feat']
        h = {'region':h_dict}

        h = self.rgcn(g, h)

        with g.local_scope():
            g.ndata['h'] = h['region']

            # print("node embedding:", g.ndata['h'].size())
            # print(g.ndata['h'])



            """feature cluster"""

            cluster_input = h['region'].T.detach().cpu().numpy()
            # print(np.shape(cluster_input))

            estimator = KMeans(n_clusters=self.cluster)
            estimator.fit(cluster_input)
            # print("cluster labels")
            # print(estimator.labels_)
            labels = estimator.predict(cluster_input)
            # print("labels", labels)
            # clusters = {}
            n = 0

            clusters_index = {}


            for item in labels:
                if item in clusters_index:
                    # clusters[item].append(cluster_input[n])
                    clusters_index[item].append(n)
                else:
                    # clusters[item] = [cluster_input[n]]
                    clusters_index[item] = [n]
                n += 1

            # print("clusters_index:", clusters_index)
            # print(len(clusters_index))


            """node_type mask"""
            node_mask_batch = g.ndata['nodelabel']
            node_embedding_batch = g.ndata['h']

            node_mask = torch.reshape(node_mask_batch, (-1,self.node_num))
            node_embedding = torch.reshape(node_embedding_batch, (-1, self.node_num, self.hidden_dim))

            # print("node_mask", node_mask.size())
            # print(node_mask)
            # print("node_embedding", node_embedding.size())


            node_embedding_dict = {}
            node_embedding_resize = {}

            for key in clusters_index:
                # print("key", key)
                feature_index_index = clusters_index[key]
                feature_filter_matrix = node_embedding[:,:, feature_index_index]
                key_mask = []
                for node_index, node_type in enumerate(node_mask[0]):
                    if node_type == key:
                        key_mask.append(node_index)
                node_feature_filter_matrix = feature_filter_matrix[:,key_mask,:]
                # print("node_feature_filter_matrix", node_feature_filter_matrix.size())

                node_feature_filter_embedding = torch.mean(node_feature_filter_matrix, dim=1, keepdim=False)
                # print("node_feature_filter_embedding", node_feature_filter_embedding.size())

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.nl = nn.Linear(len(node_feature_filter_embedding[0]),self.hidden_dim).to(device)
                node_feature_resize_matrix = self.nl(node_feature_filter_embedding)


                # print("node_feature_resize_matrix", node_feature_resize_matrix.size())

                node_embedding_resize[key] = node_feature_resize_matrix

                node_embedding_dict[key] = node_feature_filter_embedding


            node_embedding_input = torch.stack([node_embedding_resize[i] for i in node_embedding_resize], dim=1)     # batch_size*T*feature_dim
            # print("node_embedding_input", node_embedding_input.size())
            graph_embedding = self.attention(node_embedding_input)     #batch_size*feature_dim

            hg_raw = 0
            for ntype in g.ntypes:
                hg_raw = hg_raw + dgl.mean_nodes(g, 'h', ntype=ntype)

            # reconstruction_loss = torch.cdist(graph_embedding, hg_raw, p=2)
            reconstruction_loss = F.pairwise_distance(graph_embedding.reshape(-1), hg_raw.reshape(-1), p=2)
            # print("reconstruction_loss", reconstruction_loss)

            # graph_embedding = torch.cat([node_embedding_dict[i] for i in node_embedding_dict], axis=2)


            hg = graph_embedding


            # """traditional readout"""
            # hg = 0
            # for ntype in g.ntypes:
            #     hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)

            hg = self.bn(hg)

            hg = self.dropout(hg)

            label_embedding = self.classify(hg)
            # print("label embedding:", label_embedding.size())
            # print(label_embedding)



            return label_embedding, reconstruction_loss



class FeatureAttentionLayer(nn.Module):
    def __init__(self, in_dim, n_heads, cluster, attn_drop, residual):

        super(FeatureAttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.cluster = cluster
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(cluster, in_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(in_dim, in_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(in_dim, in_dim))
        # ff
        self.lin = nn.Linear(in_dim, in_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()


    def forward(self, inputs):
        # # 1: Add position embeddings to input
        # position_inputs = torch.arange(0, self.cluster).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
        #     inputs.device)
        # temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        # print("q size", q.size())

        k = torch.tensordot(inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]

        v = torch.tensordot(inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        # print("q_ sieze", q_.size())

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        # print("outputs", outputs.size())

        outputs = outputs / (self.cluster ** 0.5)


        outputs = F.softmax(outputs, dim=2)


        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)


        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + inputs
        # print("step 6", outputs.size())

        # 7: aggregation
        outputs = torch.mean(outputs, dim=1)
        # print("att outputs", outputs.size())

        return outputs

    def feedforward(self, inputs):
        # outputs = F.relu(self.lin(inputs))
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)



class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs是节点的特征
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h



class HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()

        self.mods = nn.ModuleDict(mods)
        if isinstance(aggregate, str):
            # 获取聚合函数的内部函数
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):

        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}


        if g.is_block:
            print("g.isbloc", g.is_block)
            src_inputs = inputs
            dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            src_inputs = dst_inputs = inputs


        # outputs: {dtype: [dstdata1, dstdata2, ...]}
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]

            # print("sytpe", stype)
            # print("etype", etype)
            # print("dtype", dtype)
            #
            # print("rel_graph", rel_graph)

            # rel_graph = dgl.add_self_loop(rel_graph)

            if rel_graph.num_edges() == 0:
                print("num_edges =0")
                continue
            if stype not in src_inputs or dtype not in dst_inputs:
                print("stype not in")
                continue

            rel_graph = dgl.add_self_loop(rel_graph)


            src_inputs_tensor = src_inputs[stype]
            dst_inputs_tensor = dst_inputs[dtype]

            # print(src_inputs_tensor)

            dstdata = self.mods[etype](
                rel_graph,
                (src_inputs_tensor.float(), dst_inputs_tensor.float()),
                *mod_args.get(etype, ()),
                **mod_kwargs.get(etype, {}))
            outputs[dtype].append(dstdata)


        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)

        return rsts


def _max_reduce_func(inputs, dim):
    return th.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return th.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return th.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return th.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return th.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = th.stack(inputs, dim=0)
    return fn(stacked, dim=0)


def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.

    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.

    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        print('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)