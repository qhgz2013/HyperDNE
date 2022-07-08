import logging
import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
from typing import *
import data_preprocess
import torch.utils.tensorboard as tb

_cuda_available = None
TorchInputType = Union[torch.Tensor, nn.Parameter]  # type check alias


def _check_cuda_available() -> bool:
    global _cuda_available
    if _cuda_available is None:
        # defer check (at least gpu is specified)
        logger = logging.getLogger('torch_cuda_check')
        _cuda_available = torch.cuda.is_available()
        logger.info('torch.cuda.is_available(): %s', str(_cuda_available))
    return _cuda_available


def to_cuda(x: Any) -> Any:
    if _check_cuda_available():
        return x.cuda()
    return x


def to_cpu(x: Any) -> Any:
    if _check_cuda_available():
        return x.cpu()
    return x


def scipy_sp_matrix_to_tensor(sp_matrix: Union[sp.spmatrix, torch.Tensor]) -> torch.Tensor:
    # Torch API update document: https://pytorch.org/docs/stable/sparse.html
    if isinstance(sp_matrix, torch.Tensor):
        return sp_matrix  # already converted to torch tensor
    if not sp.isspmatrix(sp_matrix):
        raise ValueError('Argument %s is not a scipy sparse matrix' % str(sp_matrix))
    if not isinstance(sp_matrix, sp.coo_matrix):
        sp_matrix = sp_matrix.tocoo()
    idx = torch.LongTensor(np.vstack([sp_matrix.row, sp_matrix.col]))  # [2, nnz] index tensor, nnz: nums of non-zeros
    val = torch.FloatTensor(sp_matrix.data)  # [nnz] value tensor
    return torch.sparse_coo_tensor(idx, val, size=sp_matrix.shape)


# Hypergraph conv layer (class HGNN_conv in HyperRec)
class HypergraphConv(nn.Module):
    def __init__(self, graph: data_preprocess.Subgraph, n_input_dims: int, n_output_dims: int, n_layers: int,
                 dropout_rate: Union[float, torch.Tensor], activation: Callable = torch.relu):
        super(HypergraphConv, self).__init__()
        # initializer: glorot uniform, -bias
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.coef1 = to_cuda(scipy_sp_matrix_to_tensor(graph.coef1))
        self.coef2 = to_cuda(scipy_sp_matrix_to_tensor(graph.coef2))
        self.n_layers = n_layers
        self.layers = nn.ModuleList([nn.Linear(n_input_dims, n_output_dims, bias=False)])
        for _ in range(1, n_layers+1):
            self.layers.append(nn.Linear(n_output_dims, n_output_dims, bias=False))
        self.dropout = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_layers+1)])
        self.activation = activation

    def forward(self, inputs):
        # inputs: [n_row, n_input_dims (=embedding_size)]
        x = self.layers[0](inputs)  # [n_row, n_output_dims (=embedding_size)]
        x = self.activation(torch.sparse.mm(self.coef1, x))  # [n_row, n_output_dims]
        for i in range(self.n_layers):
            x = self.dropout[i](x)
            x = self.layers[i+1](x)
            x = self.activation(torch.sparse.mm(self.coef1, x))  # [n_row, n_output_dims]
        y = self.dropout[-1](x)
        y = self.activation(torch.sparse.mm(self.coef2, y))  # [n_col, n_output_dims]
        # outputs: x: [n_row, n_output_dims], y: [n_col, n_output_dims]
        return x, y

    def __repr__(self):
        return f'<{self.__class__.__name__} with {self.n_layers} layer(s): {self.n_input_dims} -> {self.n_output_dims}>'


# LineConv from DHCN
# (modified, behave different from DHCN, checkout commit "3dd86fc6" for the previous implementation)
class LineConv(nn.Module):
    def __init__(self, n_layers: int, embedding_size: int, coef_matrix_da: Union[TorchInputType, sp.spmatrix],
                 include_trainable_weight: bool = True):
        super(LineConv, self).__init__()
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        coef_matrix_da = scipy_sp_matrix_to_tensor(coef_matrix_da).cuda()
        self.coef_matrix_da = coef_matrix_da  # precomputed D^{-1}A (in DHCN formula 8), [n_col, n_col], sparse tensor!
        # n_col = coef_matrix_da.size(0)
        if include_trainable_weight:
            # Line graph in DHCN is not trainable since it does not contain any parameters
            self.line_graph_weights = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
                                                     for _ in range(n_layers)])
        else:
            self.line_graph_weights = None

    def forward(self, edge_embedding: TorchInputType):
        # inputs: the mean embedding of each hyperedge in current hypergraph, [n_col, embedding_size]
        inputs = edge_embedding
        outputs = [inputs]  # [n_layers, n_col, embedding_size]
        for i in range(self.n_layers):
            if self.line_graph_weights is not None:
                inputs = self.line_graph_weights[i](inputs)
            inputs = torch.sparse.mm(self.coef_matrix_da, inputs)
            outputs.append(inputs)
        # output: [n_col, embedding_size]
        return torch.sum(torch.stack(outputs), 0)


# a minor modification of torch.nn.TransformerEncoderLayer
class TransformerEncoder(nn.Module):
    def __init__(self, seq_len: int, n_layers: int, n_output_dims: int, n_heads: int, dropout: float,
                 use_positional_embedding: bool = True, no_scale: bool = False):
        super(TransformerEncoder, self).__init__()
        self.dropout = dropout
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_output_dims = n_output_dims
        self.no_scale = no_scale  # a debug option
        self.use_positional_embedding = use_positional_embedding
        # position embedding
        self.pos_matrix = nn.Embedding(seq_len, n_output_dims) if use_positional_embedding else None
        # multi-head self-attention layers
        self.dropout1 = nn.Dropout(dropout)
        self.multihead_attn = nn.ModuleList([nn.MultiheadAttention(n_output_dims, n_heads, dropout, bias=False)
                                             for _ in range(n_layers)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(n_output_dims) for _ in range(n_layers)])
        # feed forward layers
        self.feed_forward = nn.ModuleList([PointWiseFeedForward(n_output_dims, dropout, residual=True)
                                           for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(n_output_dims)

    def forward(self, inputs: TorchInputType, mask: Optional[TorchInputType] = None,
                summary_writer: Optional[tb.SummaryWriter] = None, global_step: Optional[int] = None) -> torch.Tensor:
        # inputs: [batch_size, seq_len, embedding_size]
        # mask: [batch_size, seq_len]
        # re-scale input
        if not self.no_scale:
            scale = self.n_output_dims ** 0.5
            inputs = inputs * scale
        # position embedding
        if self.use_positional_embedding:
            inputs = inputs + torch.unsqueeze(self.pos_matrix.weight, 0)
            if summary_writer:
                summary_writer.add_histogram('embedding/position', self.pos_matrix.weight, global_step)
        mask_unsqueeze = mask and torch.unsqueeze(mask, -1)
        inputs = self.dropout1(inputs)
        if mask is not None:
            inputs = inputs * mask_unsqueeze
        # lower triangle matrix (for multi-head attention causality masking)
        lower_triangle_matrix = to_cuda(torch.tril(torch.ones([self.seq_len, self.seq_len])))
        # true values will be ignored (reverse mask), [batch_size, seq_len]
        attn_key_mask = mask and torch.logical_not(mask)
        for i in range(self.n_layers):
            # transpose: [seq_len, batch_size, emb]
            inputs = torch.transpose(inputs, 0, 1)
            # multi-head self attention
            output = self.multihead_attn[i](query=self.layer_norm[i](inputs), key=inputs, value=inputs,
                                            attn_mask=lower_triangle_matrix, key_padding_mask=attn_key_mask,
                                            need_weights=False)[0]
            # residual connection
            inputs = inputs + output  # [seq_len, batch_size, embedding_size]

            # feed forward
            inputs = torch.transpose(inputs, 0, 1)  # to: [batch_size, seq_len, embedding_size]
            inputs = self.feed_forward[i](inputs)  # [batch_size, seq_len, embedding_size]
            # masking
            # inputs = inputs.permute(0, 2, 1)  # [batch_size, seq_len, embedding_size]
            if mask is not None:
                inputs = inputs * mask_unsqueeze
        # [batch_size, seq_len, embedding_size]
        return self.output_norm(inputs)


# "Residual Gating" or "Fusion Layer" in HyperRec
class GatedFusionLayer(nn.Module):
    def __init__(self, n_input_dims: int, n_hidden_dims: Optional[int] = None, dropout: float = 0.0):
        super(GatedFusionLayer, self).__init__()
        # normally, n_input_dims = n_hidden_dims = embedding_size
        if n_hidden_dims is None:
            n_hidden_dims = n_input_dims  # set to n_input_dims if it is not specified
        self.n_input_dims = n_input_dims
        self.n_hidden_dims = n_hidden_dims
        self.linear_layer1 = nn.Linear(n_input_dims, n_hidden_dims)  # features -> features_transformed
        self.linear_layer2 = nn.Linear(n_hidden_dims, 1)  # features_transformed -> features_score
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, inputs: TorchInputType) -> torch.Tensor:
        # input: float tensor, shapes: [n_fusion_dims, n_inputs, n_input_dims]
        # output: float tensor, shape: [n_inputs, n_input_dims]
        if len(inputs.size()) == 2:
            # n_fusion_dims not exists
            return inputs
        assert len(inputs.size()) == 3, 'GatedFusionLayer expects inputs are in shape [n_fusion_dims, n_inputs, ' \
                                        'n_input_dims], but found: %d dims' % len(inputs.size())
        if inputs.size()[0] == 1:
            return torch.squeeze(inputs, 0)
        input_transformed = self.linear_layer1(inputs)  # [n_fusion_dims, n_inputs, n_hidden_dims]
        input_transformed = torch.tanh(input_transformed)
        input_score = self.linear_layer2(input_transformed)  # [n_fusion_dims, n_inputs, 1]
        input_score = torch.softmax(input_score, 0)
        input_score = self.dropout_layer(input_score)
        output = torch.sum(inputs * input_score, 0)  # [n_inputs, n_input_dims]
        return output

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.n_input_dims} (hidden: {self.n_hidden_dims})>'


# input user static embeddings, output user dynamic embeddings for each time period
class StructuralLayerHyperRec(nn.Module):
    def __init__(self, n_users: int, subgraphs: List[data_preprocess.Subgraph], n_hgcn_layers: int, embedding_size: int,
                 dropout_graph: float, dropout: float, initial_uid_u: Optional[np.ndarray] = None,
                 initial_uid_v: Optional[np.ndarray] = None, n_lg_layers: int = 2,
                 include_line_graph_weight: bool = True):
        super(StructuralLayerHyperRec, self).__init__()
        self.n_users = n_users
        self.subgraphs = subgraphs
        self.n_hgcn_layers = n_hgcn_layers
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.dropout_graph = dropout_graph
        duid_trace_u, duid_trace_v, next_duid_u, next_duid_v = \
            data_preprocess.build_dynamic_uid_trace(subgraphs, n_users, initial_uid_u, initial_uid_v)
        self.duid_trace_u = to_cuda(torch.tensor(duid_trace_u, dtype=torch.int64))
        self.duid_trace_v = to_cuda(torch.tensor(duid_trace_v, dtype=torch.int64))
        self.duid_trace_u_np = duid_trace_u  # also leave a numpy array copy
        self.duid_trace_v_np = duid_trace_v
        self.next_duid_u = next_duid_u  # not used variable, maybe it will used in the future?
        self.next_duid_v = next_duid_v
        self.subgraph_guid_v = [to_cuda(torch.LongTensor(g.row_id_mapper_rev)) for g in subgraphs]
        self.fusion_layers = nn.ModuleList([GatedFusionLayer(embedding_size, dropout=dropout) for _ in subgraphs])
        self.hgcn_layers = nn.ModuleList([HypergraphConv(g, embedding_size, embedding_size, n_hgcn_layers,
                                                         dropout_graph) for g in subgraphs])
        if n_lg_layers > 0:
            self.lg_layers = nn.ModuleList([LineConv(n_lg_layers, embedding_size, g.lg_coef_da,
                                                     include_line_graph_weight) for g in subgraphs])
        else:
            self.lg_layers = None

    def forward(self, static_embeddings: TorchInputType, summary_writer: Optional[tb.SummaryWriter] = None,
                global_step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: static_embeddings: [n_users, embedding_size]
        # outputs:
        # final_embedding_u: dynamic embedding for u: [n_users, len(subgraphs)+1, embedding_size]
        # final_embedding_v: dynamic embedding for v: [n_users, len(subgraphs)+1, embedding_size]
        user_embedding_dynamic_v = static_embeddings
        user_embedding_dynamic_u = [static_embeddings]
        for t in range(len(self.subgraphs)):
            guid_v = self.subgraph_guid_v[t]
            # n_row: # nodes that in-degree > 0 in subgraph t
            static_features = static_embeddings[guid_v, :]  # [n_row, embedding_size]
            dynamic_features = user_embedding_dynamic_v[self.duid_trace_v[guid_v, t], :]  # [n_row, embedding_size]
            features = torch.stack([static_features, dynamic_features])  # [2, n_row, embedding_size]
            features_gated_fused = self.fusion_layers[t](features)  # [n_row, embedding_size]
            # features_v: [n_row, embedding_size], features_u: [n_col, embedding_size]
            features_v, features_u = self.hgcn_layers[t](features_gated_fused)
            # include line-graph here?
            if self.lg_layers is not None:
                features_u = self.lg_layers[t](features_u)
            user_embedding_dynamic_v = torch.cat([user_embedding_dynamic_v, features_v], 0)
            user_embedding_dynamic_u.append(features_u)
        user_embedding_dynamic_u = torch.cat(user_embedding_dynamic_u, 0)
        if summary_writer:
            summary_writer.add_histogram('embedding/dynamic_u', user_embedding_dynamic_u, global_step)
            summary_writer.add_histogram('embedding/dynamic_v', user_embedding_dynamic_v, global_step)
        # [n_users, len(subgraphs)+1, embedding_size]
        final_embedding_u = user_embedding_dynamic_u[self.duid_trace_u, :]
        final_embedding_v = user_embedding_dynamic_v[self.duid_trace_v, :]
        return final_embedding_u, final_embedding_v


# Point-wise feed forward layer of SASRec, modified from https://github.com/pmixer/SASRec.pytorch/blob/master/model.py
class PointWiseFeedForward(nn.Module):
    def __init__(self, embedding_size: int, dropout_rate: float, residual: bool = True):
        super(PointWiseFeedForward, self).__init__()
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.residual = residual
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv1d(embedding_size, embedding_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv1d(embedding_size, embedding_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: TorchInputType) -> torch.Tensor:
        # inputs: [batch_size, seq_len, embedding_size]
        output = torch.transpose(inputs, -1, -2)  # [batch_size, embedding_size, seq_len]
        output = self.conv1(output)
        output = self.dropout1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.dropout2(output)
        output = torch.transpose(output, -1, -2)
        if self.residual:
            output = output + inputs
        # output: [batch_size, seq_len, embedding_size]
        return output


class HGSocialRec(nn.Module):
    def __init__(self, n_users: int, embedding_size: int, subgraphs: List[data_preprocess.Subgraph], n_hgcn_layers: int,
                 dropout: float, dropout_graph: float, predict_time: int, n_transformer_layers: int,
                 use_positional_embedding: bool = True,
                 n_line_graph_layers: bool = False, include_line_graph_weight: bool = True,
                 model_name_prefix: Optional[str] = None, model_name_postfix: Optional[str] = None):
        super(HGSocialRec, self).__init__()
        self.model_name_prefix = model_name_prefix
        self.model_name_postfix = model_name_postfix
        self.n_hgcn_alyers = n_hgcn_layers
        self.dropout = dropout
        self.dropout_graph = dropout_graph
        self.n_transformer_layers = n_transformer_layers
        self.embedding_size = embedding_size
        self.use_positional_embedding = use_positional_embedding
        self.n_line_graph_layers = n_line_graph_layers
        self.include_line_graph_weight = include_line_graph_weight
        # TODO: fix initializer as the origin implementation?
        # static embedding [n_users, embedding_size], float
        # initializer: xavier, regularizer: L2 normalization (args.l2_arg (=0))
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.subgraphs = subgraphs
        self.predict_time = predict_time
        subgraphs_dysat = subgraphs[:predict_time]
        # as what DySAT did: appending a same graph for predict_time as predict_time-1
        subgraphs_dysat.append(subgraphs[predict_time - 1])
        self.structural_layer = StructuralLayerHyperRec(n_users, subgraphs_dysat, n_hgcn_layers,
                                                        embedding_size, dropout_graph, dropout,
                                                        n_lg_layers=n_line_graph_layers,
                                                        include_line_graph_weight=include_line_graph_weight)
        # self.line_conv_layer = LineConv(n_line_conv_layers, self.user_embedding)
        # SASRec based multi-head attention feed forward layer, called TemporalAttentionLayer in DySAT
        if n_transformer_layers > 0:
            # changed: n_transformer_layers = 0 is used to disable TransformerEncoder for ablation experiment
            self.transformer_layer = TransformerEncoder(predict_time+2, n_transformer_layers, embedding_size, 1,
                                                        dropout, use_positional_embedding=use_positional_embedding)
        else:
            self.transformer_layer = None
        # time_range: [1, predict_time+2], for mask broadcasting use
        self._time_range = torch.unsqueeze(to_cuda(torch.arange(predict_time + 2)), 0)
        # time_range_neg: [1, 1, predict_time+2]
        self._time_range_neg = torch.unsqueeze(self._time_range, 0)

    def forward(self, guid_pos_u, guid_pos_v, guid_neg_u, guid_neg_v, time_period_pos, time_period_neg,
                summary_writer: Optional[tb.SummaryWriter] = None, global_step: Optional[int] = None):
        # add summary histogram
        if summary_writer:
            summary_writer.add_histogram('embedding/static', self.user_embedding.weight, global_step)
        # time_period_pos: [batch_size], time_period_neg: [batch_size, neg_sample_size]
        batch_size, neg_sample_size = guid_neg_v.size()[:2]
        if guid_neg_u.size()[1] == 1:
            guid_neg_u = torch.tile(guid_neg_u, [1, neg_sample_size])
        if time_period_neg.size()[1] == 1:
            time_period_neg = torch.tile(time_period_neg, [1, neg_sample_size])
        static_embedding = self.user_embedding.weight
        # dynamic_embedding_*: [n_users, predict_time+2, embedding_size]
        dynamic_embedding_u, dynamic_embedding_v = self.structural_layer(static_embedding, summary_writer, global_step)

        # temporal layer (maybe dynamic embedding and static embedding should be fused first?)
        # [n_users, 1, embedding_size]
        # static_embedding_unsqueeze = torch.unsqueeze(static_embedding, 1)
        # [n_users, predict_time+2, embedding_size]
        if self.transformer_layer is not None:
            dynamic_embedding_u = self.transformer_layer(dynamic_embedding_u, None, summary_writer, global_step)
            dynamic_embedding_v = self.transformer_layer(dynamic_embedding_v, None, summary_writer, global_step)
        # final fusion (with static embedding)

        # changed to predict_time - 1 as DySAT, but it does not affect the model accuracy
        pred_time_tensor = torch.tensor(self.predict_time)
        time_restricted_pos = torch.minimum(time_period_pos, pred_time_tensor)
        time_restricted_neg = torch.minimum(time_period_neg, pred_time_tensor)
        # embedding_pos_*: [batch_size, embedding_size]
        embedding_pos_u = dynamic_embedding_u[guid_pos_u, time_restricted_pos, :] + static_embedding[guid_pos_u, :]
        embedding_pos_v = dynamic_embedding_v[guid_pos_v, time_restricted_pos, :] + static_embedding[guid_pos_v, :]
        # embedding_neg_*: [batch_size, neg_sample_size, embedding_size]
        embedding_neg_u = dynamic_embedding_u[guid_neg_u, time_restricted_neg, :] + static_embedding[guid_neg_u, :]
        embedding_neg_v = dynamic_embedding_v[guid_neg_v, time_restricted_neg, :] + static_embedding[guid_neg_v, :]

        embeddings = embedding_pos_u, embedding_pos_v, embedding_neg_u, embedding_neg_v

        # [batch_size]
        pos_logits = torch.sum(embedding_pos_u * embedding_pos_v, -1)
        # [batch_size, neg_sample_size]
        neg_logits = torch.sum(embedding_neg_u * embedding_neg_v, -1)

        logits = pos_logits, neg_logits

        return embeddings, logits

    @property
    def model_name(self) -> str:
        name_parts = []
        if self.model_name_prefix is not None and len(self.model_name_prefix) > 0:
            name_parts.append(self.model_name_prefix)
        name_parts.append(f'emb{self.embedding_size}')
        name_parts.append(f'hgcn{self.n_hgcn_alyers}')
        name_parts.append(f'd{self.dropout}')
        name_parts.append(f'dg{self.dropout_graph}')
        name_parts.append(f'tl{self.n_transformer_layers}')
        name_parts.append(f't{self.predict_time}')
        if not self.use_positional_embedding:
            name_parts.append('np')  # no positional embedding
        if self.n_line_graph_layers > 0:
            name_parts.append(f'lgw{self.n_line_graph_layers}' if self.include_line_graph_weight
                              else f'lg{self.n_line_graph_layers}')
        if self.model_name_postfix is not None and len(self.model_name_postfix) > 0:
            name_parts.append(self.model_name_postfix)
        return '_'.join(name_parts)

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.model_name}>'
