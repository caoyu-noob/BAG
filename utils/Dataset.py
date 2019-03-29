import pickle
import numpy as np
import scipy.sparse

"""The class related to dataset
"""
class Dataset(object):
    def __init__(self, filename_prefix, use_elmo, use_glove, use_extra_feature, use_full_query_token=False,
                add_query_node=False, max_nodes=500, max_query_size=25, max_candidates=80, max_candidates_len=10,
                use_edge=True):
        if not use_elmo and not use_glove:
            raise Exception('At least one of ELMo, GloVe should be used')

        self.max_nodes, self.max_query_size, self.max_candidates, self.max_candidates_len = max_nodes,\
            max_query_size, max_candidates, max_candidates_len

        graph_file_name = '{}.preprocessed.pickle'.format(filename_prefix)
        self.data = [d for d in pickle.load(open(graph_file_name, 'rb')) if len(d['nodes_candidates_id']) > 0]
        self.use_elmo = use_elmo
        self.data_elmo = None
        if use_elmo:
            elmo_file_name = '{}.elmo.preprocessed.pickle'.format(filename_prefix)
            self.data_elmo = [d for d in pickle.load(open(elmo_file_name, 'rb')) if len(d['nodes_elmo']) > 0]
        self.data_glove = None
        self.use_glove = use_glove
        if use_glove:
            glove_file_name = '{}.glove.preprocessed.pickle'.format(filename_prefix)
            self.data_glove = [d for d in pickle.load(open(glove_file_name, 'rb')) if len(d['nodes_glove']) > 0]
        self.data_extra = None
        self.use_extra_feature = use_extra_feature
        if use_extra_feature:
            extra_file_name = '{}.extra.preprocessed.pickle'.format(filename_prefix)
            self.data_extra = [d for d in pickle.load(open(extra_file_name, 'rb')) if len(d['nodes_pos']) > 0]
            self.ner_dict = pickle.load(open('data/ner_dict.pickle', 'rb'))
            self.pos_dict = pickle.load(open('data/pos_dict.pickle', 'rb'))
        self.idx = list(range(len(self)))
        self.counter = len(self)
        self.use_full_query_token = use_full_query_token
        self.add_query_node = add_query_node
        self.use_edge = use_edge

    def __len__(self):
        return len(self.data)

    def getDataSize(self):
        return len(self.data)

    def getPosAndNerDictSize(self):
        return len(self.pos_dict), len(self.ner_dict)

    def buildElmoData(self, data_elmo_mb):
        filt = lambda c: np.array([c[:, 0].mean(0), c[0, 1], c[-1, 2]])

        nodes_elmo_mb = []
        for d in data_elmo_mb:
            nodes_elmo_mb.append(np.pad(np.array([filt(c) for c in d['nodes_elmo']]),
                ((0, self.max_nodes - len(d['nodes_elmo'])), (0, 0), (0, 0)), mode='constant'))
        nodes_elmo_mb = np.array(nodes_elmo_mb)

        query_elmo_mb = np.stack([np.pad(d['query_elmo'],
            ((0, self.max_query_size - d['query_elmo'].shape[0]), (0, 0), (0, 0)), mode='constant') for d in
                             data_elmo_mb], 0)
        return nodes_elmo_mb, query_elmo_mb

    """ Generating glove data"""
    def buildGloveData(self, idx, data_glove_mb):
        filt = lambda c: np.array(c[:].mean(0))
        # print(idx)
        # print([len(d['nodes_glove']) for d in data_glove_mb])
        nodes_glove_mb = np.array([
            np.pad(np.array([filt(c) for c in d['nodes_glove']]), ((0, self.max_nodes - len(d['nodes_glove'])), (0, 0)),
                mode='constant') for d in data_glove_mb])
        if not self.use_full_query_token:
            query_glove_mb = np.stack([np.pad(d['query_glove'],
                    ((0, self.max_query_size - d['query_glove'].shape[0]), (0, 0)), mode='constant')
                            for d in data_glove_mb], 0)
        else:
            query_glove_mb = np.stack([np.pad(d['query_full_token_glove'],
                ((0, self.max_query_size - d['query_full_token_glove'].shape[0]), (0, 0)), mode='constant')
                            for d in data_glove_mb], 0)
        return nodes_glove_mb, query_glove_mb

    """ Generate data for extra data like POS and NER"""
    def buildExtraData(self, data_extra_bm):
        filt = lambda c : np.argmax(np.bincount(c))
        node_ner_mb = np.array(
            [np.pad(np.array([filt(c) for c in d['nodes_ner']]), ((0, self.max_nodes - len(d['nodes_ner']))),
                mode='constant') for d in data_extra_bm])
        node_pos_mb = np.array(
            [np.pad(np.array([filt(c) for c in d['nodes_pos']]), ((0, self.max_nodes - len(d['nodes_pos']))),
                mode='constant') for d in data_extra_bm])
        if not self.use_full_query_token:
            query_pos_mb = np.stack([np.pad(d['query_pos'], (0, self.max_query_size - len(d['query_pos'])),
                mode='constant') for d in data_extra_bm], 0)
            query_ner_mb = np.stack([np.pad(d['query_ner'], (0, self.max_query_size - len(d['query_ner'])),
                mode='constant') for d in data_extra_bm], 0)
        else:
            query_pos_mb = np.stack([np.pad(d['query_pos_full_token'],
                (0, self.max_query_size - len(d['query_pos_full_token'])),
                mode='constant') for d in data_extra_bm], 0)
            query_ner_mb = np.stack([np.pad(d['query_ner_full_token'],
                (0, self.max_query_size - len(d['query_ner_full_token'])),
                mode='constant') for d in data_extra_bm], 0)
        return node_ner_mb, node_pos_mb, query_ner_mb, query_pos_mb

    """ We build edges in graph using adjacent matrices
    """
    def buildEdgeData(self, data_mb):
        adj_mb = []
        for d in data_mb:
            if self.use_edge:
                adj_ = []

                if len(d['edges_in']) == 0:
                    adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
                else:
                    adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_in'])), np.array(d['edges_in']).T),
                        shape=(self.max_nodes, self.max_nodes)).toarray()

                    adj_.append(adj)

                if len(d['edges_out']) == 0:
                    adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
                else:
                    adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_out'])), np.array(d['edges_out']).T),
                        shape=(self.max_nodes, self.max_nodes)).toarray()

                    adj_.append(adj)

                adj = np.pad(np.ones((len(d['nodes_candidates_id']), len(d['nodes_candidates_id']))),
                    ((0, self.max_nodes - len(d['nodes_candidates_id'])),
                     (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant') \
                      - adj_[0] - adj_[1] - np.pad(np.eye(len(d['nodes_candidates_id'])),
                    ((0, self.max_nodes - len(d['nodes_candidates_id'])),
                     (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant')

                adj_.append(np.clip(adj, 0, 1))

                adj = np.stack(adj_, 0)

                d_ = adj.sum(-1)
                d_[np.nonzero(d_)] **= -1
                adj = adj * np.expand_dims(d_, -1)
                adj_mb.append(adj)
            else:
                adj = np.pad(np.ones((len(d['nodes_candidates_id']), len(d['nodes_candidates_id']))),
                    ((0, self.max_nodes - len(d['nodes_candidates_id'])),
                     (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant') \
                        - np.pad(np.eye(len(d['nodes_candidates_id'])),
                    ((0, self.max_nodes - len(d['nodes_candidates_id'])),
                     (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant')
                adj_mb.append(adj)
        return adj_mb

    """ Truncate some nodes and edges if the node number in a graph exceeds the upper bound"""
    def truncateNodesAndEdge(self, data, data_elmo, data_glove, data_extra):
        # get the length of nodes in each data sample and truncate data if it exceeds the maximum length
        # including the elmo data, glove data and extra feature data
        nodes_length_mb = np.stack([len(d['nodes_candidates_id']) for d in data], 0)
        exceed_nodes_th = nodes_length_mb > self.max_nodes
        for index, exceed in enumerate(exceed_nodes_th):
            if exceed:
                data[index]['edges_in'] = self.truncateEdges(data[index]['edges_in'])
                data[index]['edges_out'] = self.truncateEdges(data[index]['edges_out'])
                data[index]['nodes_candidates_id'] = data[index]['nodes_candidates_id'][:self.max_nodes]
                if self.use_elmo:
                    data_elmo[index]['nodes_elmo'] = data_elmo[index]['nodes_elmo'][:self.max_nodes]
                if self.use_glove:
                    data_glove[index]['nodes_glove'] = data_glove[index]['nodes_glove'][:self.max_nodes]
                if self.use_extra_feature:
                    data_extra[index]['nodes_ner'] = data_extra[index]['nodes_ner'][:self.max_nodes]
                    data_extra[index]['nodes_pos'] = data_extra[index]['nodes_pos'][:self.max_nodes]
        return nodes_length_mb

    """ Sometimes if we truncate some nodes, then related edges should also be truncated"""
    def truncateEdges(self, edges):
        truncated_edges = []
        for edge_pair in edges:
            if edge_pair[0] >= self.max_nodes:
                break
            if edge_pair[1] < self.max_nodes:
                truncated_edges.append(edge_pair)
        return truncated_edges

    """ Truncate query if it exceeds the max length"""
    def truncateQuery(self, data, data_elmo, data_glove, data_extra):
        query_length_mb = np.stack([len(d['query']) for d in data], 0)
        exceed_query_th = query_length_mb > self.max_query_size
        for index, exceed in enumerate(exceed_query_th):
            if exceed:
                if self.use_elmo:
                    data_elmo[index]['query_elmo'] = data_elmo[index]['query_elmo'][:self.max_query_size]
                if self.use_glove:
                    data_glove[index]['query_glove'] = data_glove[index]['query_glove'][:self.max_query_size]
                if self.use_extra_feature:
                    data_extra[index]['query_ner'] = data_extra[index]['query_ner'][:self.max_query_size]
                    data_extra[index]['query_pos'] = data_extra[index]['query_pos'][:self.max_query_size]
        return query_length_mb

    """ The core function to generate next batch"""
    def next_batch_pro(self, idx, epoch_finished):
        data_mb = [self.data[i] for i in idx]
        data_elmo_mb = None
        data_glove_mb = None
        data_extra_mb = None
        if self.use_elmo:
            data_elmo_mb = [self.data_elmo[i] for i in idx]
        if self.use_glove:
            data_glove_mb = [self.data_glove[i] for i in idx]
        if self.use_extra_feature:
            data_extra_mb = [self.data_extra[i] for i in idx]

        id_mb = [d['id'] for d in data_mb]

        answer_candidate_id_mb = [d['answer_candidate_id'] for d in data_mb]

        nodes_length_mb = self.truncateNodesAndEdge(data_mb, data_elmo_mb, data_glove_mb, data_extra_mb)
        query_length_mb = self.truncateQuery(data_mb, data_elmo_mb, data_glove_mb, data_glove_mb)

        adj_mb = self.buildEdgeData(data_mb)

        nodes_elmo_mb, query_elmo_mb = None, None
        if self.use_elmo:
            nodes_elmo_mb, query_elmo_mb = self.buildElmoData(data_elmo_mb)

        nodes_glove_mb, query_glove_mb = None, None
        if self.use_glove:
            nodes_glove_mb, query_glove_mb = self.buildGloveData(idx, data_glove_mb)

        nodes_ner_mb, nodes_pos_mb, query_ner_mb, query_pos_mb = None, None, None, None
        if self.use_extra_feature:
            nodes_ner_mb, nodes_pos_mb, query_ner_mb, query_pos_mb = self.buildExtraData(data_extra_mb)

        if self.add_query_node:
            bmask_mb = np.array([np.pad(np.array([i == np.array(d['nodes_candidates_id'])
                                                  for i in range(len(d['candidates']) - 1)]),
                ((0, self.max_candidates - len(d['candidates']) + 1),
                 (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant')
                                 for d in data_mb])
        else:
            bmask_mb = np.array([np.pad(np.array([i == np.array(d['nodes_candidates_id'])
                                                  for i in range(len(d['candidates']))]),
                ((0, self.max_candidates - len(d['candidates'])), (0, self.max_nodes - len(d['nodes_candidates_id']))),
                mode='constant') for d in data_mb])

        return epoch_finished, {'id_mb': id_mb, 'nodes_length_mb': nodes_length_mb,
                                'query_length_mb': query_length_mb, 'bmask_mb': bmask_mb, 'adj_mb': adj_mb,
                                'answer_candidates_id_mb': answer_candidate_id_mb,
                                'nodes_elmo_mb': nodes_elmo_mb, 'query_elmo_mb': query_elmo_mb,
                                'nodes_glove_mb': nodes_glove_mb, 'query_glove_mb': query_glove_mb,
                                'nodes_ner_mb': nodes_ner_mb, 'nodes_pos_mb': nodes_pos_mb,
                                'query_ner_mb': query_ner_mb, 'query_pos_mb': query_pos_mb}

    """ Generate data for next batch"""
    def next_batch(self, batch_dim=None, use_multi_gpu=False):

        epoch_finished = False
        if batch_dim is not None:
            if self.counter >= len(self):
                np.random.shuffle(self.idx)
                self.counter = 0
            if self.counter + batch_dim >= len(self):
                idx = self.idx[self.counter:]
                epoch_finished = True
            else:
                idx = self.idx[self.counter:self.counter + batch_dim]
            self.counter += batch_dim
        else:
            idx = self.idx
        # try:
        if use_multi_gpu:
            if len(idx) % 2 != 0:
                idx.append(idx[-1])
        epoch_finished, feed_dict = self.next_batch_pro(idx, epoch_finished)
        # except:
        #     print(idx)
        return epoch_finished, feed_dict
