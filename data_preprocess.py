import numpy as np
import scipy.sparse as sp
import os
from collections import defaultdict, OrderedDict
from typing import *
import logging
from functools import partial
import threading

logger = logging.getLogger('data_preprocess')


# todo: implement random walk sampling
class Subgraph:
    def __init__(self, g: Dict[int, Set[int]]):
        # u lays in dimension 1 (col) and v lays in dimension 0 (row) of adjacent matrix for all u -> v edges
        row, col = [], []
        n_col = len(g.keys())
        row_id_mapper = OrderedDict()
        for i, u in enumerate(g.keys()):
            col.extend([i] * len(g[u]))
            for v in g[u]:
                if v in row_id_mapper:
                    v_id = row_id_mapper[v]
                else:
                    v_id = len(row_id_mapper)
                    row_id_mapper[v] = v_id
                row.append(v_id)
        n_row = len(row_id_mapper)
        val = np.ones(len(row), dtype=np.int32)
        adj_mat = sp.coo_matrix((val, (row, col)), shape=(n_row, n_col), dtype=np.int32)
        # each row in adjacent matrix represents a node in hypergraph, and each col represents a hyperedge
        # n_row = # nodes in hypergraph, n_col = # hyperedges in hypergraph
        self.n_row = n_row
        self.n_col = n_col  # same as: n_row, n_col = self.adj_mat.shape
        self.adj_mat = adj_mat
        self.adj_mat_t = adj_mat.T
        # guid -> 0-based col index for adjacent matrix
        self.col_id_mapper = {u: i for i, u in enumerate(g.keys())}
        # guid -> 0-based row index for adjacent matrix
        self.row_id_mapper = row_id_mapper
        # 0-based col index -> guid
        self.col_id_mapper_rev = np.array(list(g.keys()), dtype=np.int32)
        # 0-based row index -> guid
        self.row_id_mapper_rev = np.array(list(row_id_mapper.keys()), dtype=np.int32)

        # adjacent matrix preprocessing (in HyperRec)
        # todo: maybe the DHCN preprocessing can be added here too? as another option?
        self.dv = np.array(self.adj_mat.sum(1))  # D, [n_row]
        self.de = np.array(self.adj_mat.sum(0))  # B, [n_col]
        self.inv_dv = sp.diags(np.power(self.dv, -0.5).flatten())  # [n_row, n_row]
        self.inv_de = sp.diags(np.power(self.de, -0.5).flatten())  # [n_col, n_col]
        # coef matrix in HyperRec formula (2): B^{-0.5} H^T D^{-0.5}
        self.coef2 = self.inv_de * self.adj_mat_t * self.inv_dv  # [n_col, n_row]
        # coef matrix in HyperRec formula (1): D^{-0.5} H W B^{-1} H^T D^{-0.5}
        self.coef1 = self.inv_dv * self.adj_mat * self.inv_de * self.coef2  # [n_row, n_row]

        # line graph coefficient matrix, [n_col, n_col] sparse
        lg_adj_matrix_row = list(range(n_col))
        lg_adj_matrix_col = list(range(n_col))
        lg_adj_matrix_val = [1.0] * n_col
        # coo - csr for indexing
        adj_mat_t_csr = self.adj_mat_t.tocsr()  # type: sp.csr_matrix
        from time import time
        t1 = time()
        id_sets = []
        for u in range(n_col):
            id_sets.append(set(adj_mat_t_csr.indices[adj_mat_t_csr.indptr[u]:adj_mat_t_csr.indptr[u+1]]))
        for u in range(n_col):
            for v in range(u+1, n_col):
                inter = len(id_sets[u].intersection(id_sets[v]))
                if inter == 0:
                    continue
                union = len(id_sets[u].union(id_sets[v]))
                lg_adj_matrix_row.extend([u, v])
                lg_adj_matrix_col.extend([v, u])
                iou = inter / union
                lg_adj_matrix_val.extend([iou, iou])
        t2 = time()
        logger.debug('[Perf debug] Line graph construction time used: %f', t2 - t1)
        # line graph adjacent matrix A (defined in formula (8) in DHCN paper): $\hat{A} = A + I$
        lg_adj_matrix = sp.coo_matrix((lg_adj_matrix_val, (lg_adj_matrix_row, lg_adj_matrix_col)), shape=(n_col, n_col),
                                      dtype=np.float32)
        self.lg_adj_matrix = lg_adj_matrix
        # line graph diagonal matrix D
        deg = lg_adj_matrix.sum(1)
        # pre-computed coefficient matrix $\hat{A} D^{-1}$
        lg_coef_da = lg_adj_matrix * sp.diags(np.asarray(1.0 / deg).squeeze())
        self.lg_coef_da = lg_coef_da


class NegativeSamplingReuseCacheScope:
    def __init__(self):
        self._reuse_context = False

    def __enter__(self):
        self._reuse_context = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._reuse_context = False

    def __call__(self, *args, **kwargs):
        value = _negative_sampling(*args, **kwargs, reuse_context=self._reuse_context)
        self._reuse_context = True
        return value


_global_negative_sampling_context = threading.local()


def _negative_sampling(existed_edges_u: np.ndarray, existed_edges_v: np.ndarray, sample_size: int, n_users: int,
                       node_set: Optional[Tuple[np.ndarray, np.ndarray]] = None, unique: bool = False,
                       max_sample_iter: int = 20, reuse_context: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # NOTE: if "unique" is set to true, sample procedure will be terminated if sample iteration exceeded, and the number
    # of returned samples may less than the required "sample_size". If you do need "sample_size" samples any way, set
    # "max_sample_iter" to -1. Remind that if there are no more samples available, it may cause infinite loop here.
    # a unique id for each edge (0 ~ n_users^2-1), used to remove duplicated edges when parameter "unique" is true
    if reuse_context:
        # thread-independent local variable is used here
        edge_id, edge_id_arr = _global_negative_sampling_context.ctx
    else:
        # code optimization: computing all edge id per function call is a bit time consuming (~60% running time),
        # so a "reuse_context" option is added to reuse these intermediate results
        edge_id = set(existed_edges_u * n_users + existed_edges_v)
        edge_id_arr = np.fromiter(edge_id, dtype=np.int64)
        _global_negative_sampling_context.ctx = edge_id, edge_id_arr
    sampled_edge_u = np.empty(sample_size, dtype=np.int32)
    sampled_edge_v = np.empty_like(sampled_edge_u)
    n_sampled = 0
    if node_set is None:
        # draw node u and v from all users
        nodes_v_draw_func = nodes_u_draw_func = lambda n: np.random.randint(n_users, size=n, dtype=np.int32)
    else:
        # draw node u from node_set[0], and node v from node_set[1]
        nodes_u, nodes_v = node_set
        if nodes_u.size == 0 or nodes_v.size == 0:
            # one of the node candidates is empty
            if max_sample_iter < 0:
                raise ValueError('Empty node candidate set for negative sampling')
            empty_arr = np.array([], dtype=np.int32)
            return empty_arr, empty_arr
        nodes_u_draw_func = lambda n: nodes_u[np.random.randint(nodes_u.size, size=n, dtype=np.int32)]
        nodes_v_draw_func = lambda n: nodes_v[np.random.randint(nodes_v.size, size=n, dtype=np.int32)]
    itr = 0
    while n_sampled < sample_size and (max_sample_iter > 0 and itr < max_sample_iter):
        itr += 1
        sampled_u = nodes_u_draw_func(sample_size)
        sampled_v = nodes_v_draw_func(sample_size)
        if unique:
            sampled_edge_id = sampled_u * n_users + sampled_v
            # make sure the sampled edges are unique to each other
            _, idx = np.unique(sampled_edge_id, return_index=True)
            idx = np.sort(idx)
            sampled_edge_id = sampled_edge_id[idx]
            sampled_u = sampled_u[idx]
            sampled_v = sampled_v[idx]
            # make sure the sampled edges are not shown in the positive edges, too
            valid_mask = np.logical_not(np.isin(sampled_edge_id, edge_id_arr))
            sampled_u = sampled_u[valid_mask][:sample_size-n_sampled]
            sampled_v = sampled_v[valid_mask][:sample_size-n_sampled]
            edge_id_arr = np.concatenate((edge_id_arr, sampled_edge_id), dtype=np.int64)
        sampled_edge_u[n_sampled:n_sampled + len(sampled_u)] = sampled_u
        sampled_edge_v[n_sampled:n_sampled + len(sampled_v)] = sampled_v
        n_sampled += len(sampled_u)
    if n_sampled != sample_size:
        # not enough edges sampled
        return sampled_edge_u[:n_sampled], sampled_edge_v[:n_sampled]
    return sampled_edge_u, sampled_edge_v


# trace dynamic uid for each time period t and user u / v
# (same as function "subgraph_key_building" in "hypergraph_utils.py" in HyperRec)
# dynamic uid for u and v are independent and self-increment
def build_dynamic_uid_trace(subgraphs: List[Subgraph], n_users: int, initial_uid_u: Optional[np.ndarray] = None,
                            initial_uid_v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    # initial_uid_u/v is used for recursive dynamic uid generation that allows multiple subgraphs at the same time
    # period and shares the dynamic uid of all the predecessor subgraphs.
    # build_dynamic_uid_trace(subgraphs, n_users) is equivalent to:
    # duid_trace_u0, duid_trace_v0, _, _ = build_dynamic_uid_trace([subgraphs[0]], n_users)
    # duid_trace_u1, duid_trace_v1, _, _ = build_dynamic_uid_trace([subgraphs[1]], n_users,
    #                                                              duid_trace_u0[:, -1], duid_trace_v0[:, -1])
    # ....
    # duid_trace_u = np.stack([duid_trace_u0, duid_trace_u1[:, 1], ..., duid_trace_ut[:, 1]], 1)
    # duid_trace_v = ...
    duid_trace_u = np.empty((n_users, len(subgraphs)+1), dtype=np.int32)
    duid_trace_v = np.empty_like(duid_trace_u)
    # duid_trace[:, 0] is reserved as the initial dynamic uid array (before applying any hypergraph convolutions)
    if initial_uid_u is None:
        initial_uid_u = np.arange(n_users, dtype=np.int32)
    if initial_uid_v is None:
        initial_uid_v = np.arange(n_users, dtype=np.int32)
    duid_trace_u[:, 0] = initial_uid_u
    duid_trace_v[:, 0] = initial_uid_v

    def _update_trace_matrix(t_, uid_, duid_trace, id_mapper, next_duid_):
        if uid_ in id_mapper:
            # interacted user u / v in time period t, update the dynamic uid in next time period t+1
            duid_trace[uid_, t_ + 1] = next_duid_
            return next_duid_ + 1
        else:
            # non-interacted user u / v in time period t, keep current dynamic uid in next time period t+1
            duid_trace[uid_, t_ + 1] = duid_trace[uid_, t_]
            return next_duid_

    next_duid_u = np.max(initial_uid_u) + 1
    next_duid_v = np.max(initial_uid_v) + 1
    for t in range(len(subgraphs)):
        for uid in range(n_users):
            next_duid_u = _update_trace_matrix(t, uid, duid_trace_u, subgraphs[t].col_id_mapper, next_duid_u)
            next_duid_v = _update_trace_matrix(t, uid, duid_trace_v, subgraphs[t].row_id_mapper, next_duid_v)
    return duid_trace_u, duid_trace_v, next_duid_u, next_duid_v


class Dataset:
    def __init__(self, dataset_name: str, batch_size: int = 512, neg_sample_size: int = 10,
                 test_neg_sample_size: int = 1, predict_time: int = -1, undirected_graph: bool = True,
                 predict_step: int = 1):
        self.batch_size = batch_size
        dataset_file_path = os.path.join('data', dataset_name + '.txt')
        self.neg_sample_size = neg_sample_size
        self.test_neg_sample_size = test_neg_sample_size
        assert os.path.isfile(dataset_file_path), 'Dataset file "%s" not exists' % dataset_file_path
        self.dataset_name = dataset_name
        self.dataset_file_path = dataset_file_path
        self.predict_time = predict_time
        self.undirected_graph = undirected_graph
        self.predict_step = predict_step
        self._parse_dataset()
        # self._build_dynamic_uid_sequence()
        self._build_test_subgraph()

    # reads dataset file, construct subgraph for each time step, and obtain the relations
    def _parse_dataset(self):
        graphs = defaultdict(lambda: defaultdict(set))
        # index of variable "graphs":
        # graphs[t]: the adjacent dictionary "u" -> "v"s (list of "v") w.r.t. time "t"
        # graphs[t][u]: the direct neighbors (list of "v") w.r.t. time "t" and user "u"
        user_relations_v = defaultdict(list)
        user_relations_t = defaultdict(list)
        n_edges = 0
        uid_set = set()
        logger.debug('Open file "%s" for reading', self.dataset_file_path)
        with open(self.dataset_file_path, 'r') as f:
            for line in f:
                if len(line) == 0:
                    continue
                # u, v denotes the user id of each edge (u -> v), t denotes the time associated to the edge
                u, v, t = [int(x) for x in line.split('\t')]
                if v in graphs[t][u] or (self.undirected_graph and u in graphs[t][v]):
                    raise ValueError('Duplicated edge: t=%d, u=%d, v=%d' % (t, u, v))
                graphs[t][u].add(v)
                user_relations_v[u].append(v)
                user_relations_t[u].append(t)
                n_edges += 1
                if self.undirected_graph:
                    # undirected symmetric edges (if directed_graph set to false)
                    graphs[t][v].add(u)
                    user_relations_v[v].append(u)
                    user_relations_t[v].append(t)
                    n_edges += 1
                uid_set.add(u)
                uid_set.add(v)
        self.n_users = len(uid_set)
        self.n_time_periods = len(graphs)
        self.n_edges = n_edges
        self.graph_dicts = graphs
        logger.info('----------')
        logger.info('Basic statistics for dataset "%s":', self.dataset_name)
        logger.info('# users (in total): %d', self.n_users)
        logger.info('# time periods (in total): %d', self.n_time_periods)
        logger.info('# edges (in total): %d', self.n_edges)
        # adjust predict time when < 0
        if self.predict_time < 0:
            self.predict_time += self.n_time_periods
            # the first time period (t=0) should not be used for predicting
            assert self.predict_time > 0, 'predict_time out of range'
        logger.info('Time period to predict: %d', self.predict_time)
        actual_predict_step = min(self.predict_step, self.n_time_periods - self.predict_time)
        if actual_predict_step != self.predict_step:
            logger.warning('Parameter predict_step specified %d step(s) to predict, but only %d step(s) are available',
                           self.predict_step, actual_predict_step)
            self.predict_step = actual_predict_step
        time_step_str = f't={self.predict_time}' if self.predict_step == 1 \
            else f't={self.predict_time}~{self.predict_time+self.predict_step-1}'
        logger.info('Time steps to predict: %d (%s)', self.predict_step, time_step_str)

        logger.debug('Constructing subgraphs')
        subgraphs = []
        for t in range(self.n_time_periods):
            # build subgraphs
            subgraphs.append(Subgraph(graphs[t]))
            n_nodes = len(set(subgraphs[t].row_id_mapper.keys()) | set(subgraphs[t].col_id_mapper.keys()))
            logger.info('Subgraph t=%d: # nodes: %d, # edges: %d, adjacent matrix: %d x %d', t, n_nodes,
                        subgraphs[t].adj_mat.nnz, subgraphs[t].n_row, subgraphs[t].n_col)
        self.subgraphs = subgraphs

        # interaction sequence for each user
        logger.debug('Building interaction sequences')
        self.user_relation_v = np.empty(self.n_edges, dtype=np.int32)
        self.user_relation_t = np.empty(self.n_edges, dtype=np.int32)
        self.user_relation_ind = np.zeros(self.n_users + 1, dtype=np.int32)
        for u in range(self.n_users):
            if u in user_relations_v:
                v, t = user_relations_v[u], user_relations_t[u]
                # keep monotonically increasing
                t_order = np.argsort(t, kind='stable')
                v = np.array(v, dtype=np.int32)[t_order]
                t = np.array(t, dtype=np.int32)[t_order]
                length = len(v)
                self.user_relation_v[self.user_relation_ind[u]:self.user_relation_ind[u] + length] = v
                self.user_relation_t[self.user_relation_ind[u]:self.user_relation_ind[u] + length] = t
            else:
                length = 0
            self.user_relation_ind[u + 1] = self.user_relation_ind[u] + length

        # uid trace for the dataset ground-truth links
        logger.debug('Constructing uid trace')
        duid_trace_u, duid_trace_v, next_duid_u, next_duid_v = build_dynamic_uid_trace(self.subgraphs, self.n_users)
        logger.debug('Dynamic user IDs in used: row: %d, column: %d', next_duid_v, next_duid_u)
        self.duid_trace_u = duid_trace_u
        self.duid_trace_v = duid_trace_v

    def _build_test_subgraph(self):
        logger.debug('Building test subgraph')
        test_graph_dicts = [self.graph_dicts[self.predict_time+t].copy() for t in range(self.predict_step)]
        # test_subgraph = self.subgraphs[self.predict_time:self.predict_time+self.predict_step]
        # last_subgraph = self.subgraphs[self.predict_time - 1]
        # last_subgraph_existed_uids = set(last_subgraph.col_id_mapper_rev).union(last_subgraph.row_id_mapper_rev)
        node_set_row, node_set_col = set(), set()
        for t in range(self.predict_time):
            node_set_row.update(self.subgraphs[t].row_id_mapper_rev)
            node_set_col.update(self.subgraphs[t].col_id_mapper_rev)
        node_set = np.fromiter(node_set_col, np.int32), np.fromiter(node_set_row, np.int32)
        existed_uids = node_set_row.union(node_set_col)
        # same processing as DySAT:
        # keep all positive edges whose nodes are already existed in time period predict_time-1
        for test_graph_dict in test_graph_dicts:
            for u in list(test_graph_dict.keys()):
                if u in existed_uids:
                    test_graph_dict[u].intersection_update(existed_uids)
                    if len(test_graph_dict[u]) == 0:
                        del test_graph_dict[u]
                else:
                    del test_graph_dict[u]

        self.test_graph_dicts = test_graph_dicts
        self.test_subgraphs = [Subgraph(test_graph_dict) for test_graph_dict in test_graph_dicts]
        self.predict_pos_edge_train = []
        self.predict_pos_edge_val = []
        self.predict_pos_edge_test = []
        self.predict_neg_edge_train = []
        self.predict_neg_edge_val = []
        self.predict_neg_edge_test = []
        for t, test_subgraph in enumerate(self.test_subgraphs):
            test_adj_matrix = test_subgraph.adj_mat
            logger.info('# edges in test subgraph t=%d: %d', self.predict_time + t, test_adj_matrix.nnz)

            # split training, validation and test set as DySAT  # todo: parameterize
            # (only used when evaluating HAD based metrics)
            val_split_ratio = 0.2
            test_split_ratio = 0.6
            test_edges_u = test_subgraph.col_id_mapper_rev[test_adj_matrix.col]
            test_edges_v = test_subgraph.row_id_mapper_rev[test_adj_matrix.row]
            indices = np.arange(test_adj_matrix.nnz, dtype=np.int32)
            np.random.shuffle(indices)
            n_val_samples = int(round(val_split_ratio * test_adj_matrix.nnz))
            n_test_samples = int(round(test_split_ratio * test_adj_matrix.nnz))
            n_train_samples = test_adj_matrix.nnz - n_val_samples - n_test_samples
            indices_train = indices[:n_train_samples]
            indices_val = indices[n_train_samples:n_train_samples + n_val_samples]
            indices_test = indices[n_train_samples + n_val_samples:]
            self.predict_pos_edge_train.append((test_edges_u[indices_train], test_edges_v[indices_train]))
            self.predict_pos_edge_val.append((test_edges_u[indices_val], test_edges_v[indices_val]))
            self.predict_pos_edge_test.append((test_edges_u[indices_test], test_edges_v[indices_test]))
            logger.info('# positive train/val/test edges in t=%d: %d/%d/%d', self.predict_time + t,
                        n_train_samples, n_val_samples, n_test_samples)
            # negative edges = edges not existed in time period predict_time, but nodes of each edge must be appeared in
            # time period predict_time-1
            existed_edges_u = test_subgraph.col_id_mapper_rev[test_subgraph.adj_mat.col]
            existed_edges_v = test_subgraph.row_id_mapper_rev[test_subgraph.adj_mat.row]
            with NegativeSamplingReuseCacheScope() as scope:
                sample_func = partial(scope.__call__, existed_edges_u, existed_edges_v, n_users=self.n_users,
                                      node_set=node_set, unique=True, max_sample_iter=10000)
                self.predict_neg_edge_train.append(sample_func(sample_size=n_train_samples * self.test_neg_sample_size))
                self.predict_neg_edge_val.append(sample_func(sample_size=n_val_samples * self.test_neg_sample_size))
                self.predict_neg_edge_test.append(sample_func(sample_size=n_test_samples * self.test_neg_sample_size))
            logger.info('# negative train/val/test edges in t=%d: %d/%d/%d', self.predict_time + t,
                        self.predict_neg_edge_train[t][0].size, self.predict_neg_edge_val[t][0].size,
                        self.predict_neg_edge_test[t][0].size)


# data loader used for training
class DatasetLoader:
    def __init__(self, dataset: Dataset, unique_pos_edge: bool = True, unique_neg_edge: bool = False,
                 shuffle_user: bool = True):
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.predict_time = dataset.predict_time
        self.neg_sample_size = dataset.neg_sample_size
        self.n_users = dataset.n_users
        self.shuffle_user = shuffle_user
        # if set to true, all edges will be unique (when the number of edges is less than batch_size, the sampler will
        # use all the edges without making duplications)
        self.unique_pos_edge = unique_pos_edge
        self.unique_neg_edge = unique_neg_edge
        # positive edges for each time period
        pos_edges = []  # type: List[Tuple[np.ndarray, np.ndarray]]
        # as DySAT, time period predict_time is also included for embedding training
        for t in range(dataset.predict_time+1):
            u = dataset.subgraphs[t].col_id_mapper_rev[dataset.subgraphs[t].adj_mat.col]
            v = dataset.subgraphs[t].row_id_mapper_rev[dataset.subgraphs[t].adj_mat.row]
            pos_edges.append((u, v))
        self._pos_edges = pos_edges

    def __iter__(self):
        users = np.arange(self.n_users, dtype=np.int32)
        if self.shuffle_user:
            np.random.shuffle(users)
        for start_idx in range(0, self.n_users, self.batch_size):
            users_in_batch = users[start_idx:start_idx+self.batch_size]
            # sample positive and negative triplet for training
            guid_u, guid_pos_v, guid_neg_v = [], [], []
            time_period = []
            for t in range(self.predict_time+1):
                sampled_guid_u, sampled_pos_guid_v = self._pos_sampling(t, users_in_batch)
                sampled_neg_guid_v = self._neg_sampling(t, sampled_guid_u)
                guid_u.append(sampled_guid_u)
                guid_pos_v.append(sampled_pos_guid_v)
                guid_neg_v.append(sampled_neg_guid_v)
                time_period.extend([t] * len(sampled_guid_u))
            guid_u, guid_pos_v, guid_neg_v, time_period = [np.concatenate(x) for x in
                                                           [guid_u, guid_pos_v, guid_neg_v, [time_period]]]
            # shuffle
            idx = np.arange(guid_u.shape[0], dtype=np.int32)
            np.random.shuffle(idx)
            guid_u, guid_pos_v, guid_neg_v, time_period = [x[idx] for x in
                                                           [guid_u, guid_pos_v, guid_neg_v, time_period]]
            yield guid_u, guid_pos_v, guid_neg_v, time_period

    def __len__(self):
        return 1 + (self.n_users - 1) // self.batch_size  # = ceil(self.n_users / self.batch_size)

    # sample sample_size positive edges in time period time_period
    def _pos_sampling(self, time_period: int, candidate_users: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # pos_edges_u, pos_edges_v = self._pos_edges[time_period]
        edge_dict = self.dataset.graph_dicts[time_period]
        pos_edges_u, pos_edges_v = [], []
        for u in set(candidate_users).intersection(edge_dict.keys()):
            pos_edges_u.extend([u] * len(edge_dict[u]))
            pos_edges_v.extend(edge_dict[u])
        pos_edges_u, pos_edges_v = np.array(pos_edges_u, dtype=np.int32), np.array(pos_edges_v, dtype=np.int32)
        n_edges = pos_edges_u.size
        if n_edges == 0:
            # no positive edge here
            return pos_edges_u, pos_edges_v
        if n_edges < self.batch_size:
            # not enough edges to sample
            if self.unique_pos_edge:
                return pos_edges_u, pos_edges_v
            idx = np.random.choice(n_edges, size=self.batch_size, replace=True)
            return pos_edges_u[idx], pos_edges_v[idx]
        idx = np.random.choice(n_edges, size=self.batch_size, replace=not self.unique_pos_edge)
        return pos_edges_u[idx], pos_edges_v[idx]

    def _neg_sampling(self, time_period: int, pos_edges_u: np.ndarray) -> np.ndarray:
        # -1 as nan
        neg_edges_v = np.full((pos_edges_u.size, self.neg_sample_size), -1, dtype=np.int32)
        node_v = self.dataset.subgraphs[time_period].row_id_mapper_rev
        # aggregate sample operation (i.e., draw samples for each u once, no matter how many times u appear in
        # pos_edges_u)
        edge_u, u_cnt = np.unique(pos_edges_u, return_counts=True)
        edge_lookup_dict = {int(u): i for i, u in enumerate(edge_u)}
        neg_samples = []
        neg_sample_cnt = []
        with NegativeSamplingReuseCacheScope() as scope:
            for u, cnt in zip(edge_u, u_cnt):
                neg_sample_size = self.neg_sample_size * cnt
                sampled_v = scope(self._pos_edges[time_period][0], self._pos_edges[time_period][1], neg_sample_size,
                                  self.dataset.n_users, (np.array([u], dtype=np.int32), node_v),
                                  self.unique_neg_edge)[1]
                neg_samples.append(sampled_v)
                neg_sample_cnt.append(sampled_v.size)
        for i, u in enumerate(pos_edges_u):
            u = int(u)
            samples = neg_samples[edge_lookup_dict[u]]
            range_end = neg_sample_cnt[edge_lookup_dict[u]]
            range_begin = max(0, range_end - self.neg_sample_size)
            neg_edges_v[i, :range_end-range_begin] = samples[range_begin:range_end]
            neg_sample_cnt[edge_lookup_dict[u]] -= range_end - range_begin
        return neg_edges_v


# data loader used for evaluation
class PredictDataLoader:
    def __init__(self, dataset: Dataset, subset: str, predict_step: int = 0):
        # param predict_step is ranging from zero
        assert subset in ['train', 'val', 'test']
        assert predict_step < dataset.predict_step, 'Parameter predict_step out of range'
        arg_name = 'predict_%s_edge_%s'
        self._guid_pos_u, self._guid_pos_v = getattr(dataset, arg_name % ('pos', subset))[predict_step]
        self._guid_neg_u, self._guid_neg_v = getattr(dataset, arg_name % ('neg', subset))[predict_step]
        self._batch_size = dataset.batch_size
        self._t = np.full_like(self._guid_pos_u, dataset.predict_time)
        self._n_edges = self._guid_pos_u.size
        assert self._n_edges > 0

    def __iter__(self):
        for start_idx in range(0, self._n_edges, self._batch_size):
            idx = slice(start_idx, start_idx + self._batch_size)
            yield self._guid_pos_u[idx], self._guid_pos_v[idx], np.expand_dims(self._guid_neg_u[idx], 1), \
                np.expand_dims(self._guid_neg_v[idx], 1), self._t[idx], np.expand_dims(self._t[idx], 1)

    def __len__(self):
        return 1 + (self._n_edges - 1) // self._batch_size  # = ceil(self._n_edges / self._batch_size)
