import torch
import pickle
import scipy
from scipy.sparse.coo import coo_matrix

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def load_raw_data(args, file_name_prefix):
    graph_file_name = '{}.preprocessed.pickle'.format(file_name_prefix)
    base_data = [d for d in pickle.load(open(graph_file_name, 'rb')) if len(d['nodes_candidates_id']) > 0]
    data_elmo = None
    if args.use_elmo:
        elmo_file_name = '{}.elmo.preprocessed.pickle'.format(file_name_prefix)
        data_elmo = [d for d in pickle.load(open(elmo_file_name, 'rb')) if len(d['nodes_elmo']) > 0]
    args.data_glove = None
    if args.use_glove:
        glove_file_name = '{}.glove.preprocessed.pickle'.format(file_name_prefix)
        data_glove = [d for d in pickle.load(open(glove_file_name, 'rb')) if len(d['nodes_glove']) > 0]
    data_extra = None
    if args.use_extra_feature:
        extra_file_name = '{}.extra.preprocessed.pickle'.format(file_name_prefix)
        data_extra = [d for d in pickle.load(open(extra_file_name, 'rb')) if len(d['nodes_pos']) > 0]
        ner_dict = pickle.load(open('data/ner_dict.pickle', 'rb'))
        pos_dict = pickle.load(open('data/pos_dict.pickle', 'rb'))
    return base_data, data_elmo, data_glove, data_extra, ner_dict, pos_dict

def truncate_edge(edges, max_nodes):
    truncated_edges = []
    for edge_pair in edges:
        if edge_pair[0] >= max_nodes:
            break
        if edge_pair[1] < max_nodes:
            truncated_edges.append(edge_pair)
    return truncated_edges

""" Truncate some nodes and edges if the node number in a graph exceeds the upper bound"""
def truncate_node_and_edge(args, base_data, data_elmo, data_glove, data_extra):
    # get the length of nodes in each data sample and truncate data if it exceeds the maximum length
    # including the elmo data, glove data and extra feature data
    if len(base_data['nodes_candidates_id']) > args.max_nodes:
        base_data['edges_in'] = truncate_edge(base_data['edges_in'], args.max_nodes)
        base_data['edges_out'] = truncate_edge(base_data['edges_out'], args.max_nodes)
        base_data['nodes_candidates_id'] = base_data['nodes_candidates_id'][:args.max_nodes]
        if args.use_elmo:
            data_elmo['nodes_elmo'] = data_elmo['nodes_elmo'][:args.max_nodes]
        if args.use_glove:
            data_glove['nodes_glove'] = data_glove['nodes_glove'][:args.max_nodes]
        if args.use_extra_feature:
            data_extra['nodes_ner'] = data_extra['nodes_ner'][:args.max_nodes]
            data_extra['nodes_pos'] = data_extra['nodes_pos'][:args.max_nodes]
    return len(base_data['nodes_candidates_id'])

""" Truncate query if it exceeds the max length"""
def truncate_query(args, base_data, data_elmo, data_glove, data_extra):
    if len(base_data['query']) > args.max_query_size:
        if args.use_elmo:
            data_elmo['query_elmo'] = data_elmo['query_elmo'][:args.max_query_size]
        if args.use_glove:
            data_glove['query_glove'] = data_glove['query_glove'][:args.max_query_size]
        if args.use_extra_feature:
            data_extra['query_ner'] = data_extra['query_ner'][:args.max_query_size]
            data_extra['query_pos'] = data_extra['query_pos'][:args.max_query_size]
    return len(base_data['query'])

""" We build edges in graph using adjacent matrices
    """
def build_edge_data(args, base_data):
    if args.use_edge:
        adj_ = []

        if len(base_data['edges_in']) == 0:
            adj_.append(np.zeros((args.max_nodes, args.max_nodes), dtype=np.float32))
        else:
            adj = coo_matrix((np.ones(len(base_data['edges_in'])), np.array(base_data['edges_in']).T),
                shape=(args.max_nodes, args.max_nodes), dtype=np.float32).toarray()

            adj_.append(adj)

        if len(base_data['edges_out']) == 0:
            adj_.append(np.zeros((args.max_nodes, args.max_nodes), dtype=np.float32))
        else:
            adj = coo_matrix((np.ones(len(base_data['edges_out'])), np.array(base_data['edges_out']).T),
                shape=(args.max_nodes, args.max_nodes), dtype=np.float32).toarray()

            adj_.append(adj)

        adj = np.pad(np.ones((len(base_data['nodes_candidates_id']), len(base_data['nodes_candidates_id'])), dtype=np.float32),
            ((0, args.max_nodes - len(base_data['nodes_candidates_id'])),
             (0, args.max_nodes - len(base_data['nodes_candidates_id']))), mode='constant') \
              - adj_[0] - adj_[1] - np.pad(np.eye(len(base_data['nodes_candidates_id'])),
            ((0, args.max_nodes - len(base_data['nodes_candidates_id'])),
             (0, args.max_nodes - len(base_data['nodes_candidates_id']))), mode='constant')

        adj_.append(np.clip(adj, 0, 1, dtype=np.float32))

        adj = np.stack(adj_, 0)

        d_ = adj.sum(-1)
        d_[np.nonzero(d_)] **= -1
        adj = adj * np.expand_dims(d_, -1)
        return torch.from_numpy(adj)
    else:
        adj = np.pad(np.ones((len(base_data['nodes_candidates_id']), len(d['nodes_candidates_id']))),
            ((0, args.max_nodes - len(base_data['nodes_candidates_id'])),
             (0, args.max_nodes - len(base_data['nodes_candidates_id']))), mode='constant') \
                - np.pad(np.eye(len(base_data['nodes_candidates_id'])),
            ((0, args.max_nodes - len(base_data['nodes_candidates_id'])),
             (0, args.max_nodes - len(base_data['nodes_candidates_id']))), mode='constant')
        return torch.from_numpy(adj)

def build_elmo_data(args, elmo):
    filt = lambda c: np.array([c[:, 0].mean(0), c[0, 1], c[-1, 2]])

    nodes_elmo = np.pad(np.array([filt(c) for c in elmo['nodes_elmo']], dtype=np.float32),
            ((0, args.max_nodes - len(elmo['nodes_elmo'])), (0, 0), (0, 0)), mode='constant')
    query_elmo = np.pad(np.array(elmo['query_elmo'], dtype=np.float32),
        ((0, args.max_query_size - elmo['query_elmo'].shape[0]), (0, 0), (0, 0)), mode='constant')
    return torch.from_numpy(nodes_elmo), torch.from_numpy(query_elmo)

""" Generating glove data"""
def build_glove_data(args, glove):
    filt = lambda c: np.array(c[:].mean(0))
    nodes_glove = np.pad(np.array([filt(c) for c in glove['nodes_glove']]),
                         ((0, args.max_nodes - len(glove['nodes_glove'])), (0, 0)), mode='constant')
    if not args.use_full_query_token:
        query_glove = np.pad(glove['query_glove'],
                ((0, args.max_query_size - glove['query_glove'].shape[0]), (0, 0)), mode='constant')
    else:
        query_glove = np.pad(glove['query_full_token_glove'],
            ((0, args.max_query_size - glove['query_full_token_glove'].shape[0]), (0, 0)), mode='constant')
    return torch.from_numpy(nodes_glove), torch.from_numpy(query_glove)

def build_extra_feature(args, extra):
    filt = lambda c : np.argmax(np.bincount(c))
    nodes_ner = np.pad(np.array([filt(c) for c in extra['nodes_ner']], dtype=np.int64),
                       ((0, args.max_nodes - len(extra['nodes_ner']))), mode='constant')
    nodes_pos = np.pad(np.array([filt(c) for c in extra['nodes_pos']], dtype=np.int64),
                       ((0, args.max_nodes - len(extra['nodes_pos']))), mode='constant')
    if not args.use_full_query_token:
        query_pos = np.pad(np.array(extra['query_pos'], dtype=np.int64),
                           (0, args.max_query_size - len(extra['query_pos'])), mode='constant')
        query_ner = np.pad(np.array(extra['query_ner'], dtype=np.int64),
                           (0, args.max_query_size - len(extra['query_ner'])), mode='constant')
    else:
        query_pos = np.pad(np.array(extra['query_pos_full_token'], dtype=np.int64),
            (0, args.max_query_size - len(extra['query_pos_full_token'])), mode='constant')
        query_ner = np.pad(np.array(extra['query_ner_full_token'], dtype=np.int64),
            (0, args.max_query_size - len(extra['query_ner_full_token'])), mode='constant')
    return torch.from_numpy(nodes_ner), torch.from_numpy(nodes_pos), torch.from_numpy(query_ner), \
           torch.from_numpy(query_pos)

'''Build output mask to show the relationship between each node and candidate'''
def build_output_nodes_mask(args, base_data):
    if args.add_query_node:
        output_mask = np.pad(np.array([i == np.array(base_data['nodes_candidates_id'])
                                              for i in range(len(base_data['candidates']) - 1)]),
                                    ((0, args.max_candidates - len(base_data['candidates']) + 1),
                                     (0, args.max_nodes - len(base_data['nodes_candidates_id']))), mode='constant')
    else:
        output_mask = np.pad(np.array([i == np.array(base_data['nodes_candidates_id'])
                                              for i in range(len(base_data['candidates']))]),
                                    ((0, args.max_candidates - len(base_data['candidates'])),
                                     (0, args.max_nodes - len(base_data['nodes_candidates_id']))), mode='constant')
    return torch.from_numpy(output_mask)

'''Build pytorch tensor data so as to used in dataset loader'''
def build_tensor_data(args, base_data, data_elmo, data_glove, data_extra):
    nodes_mask, query_lengths, adjs, output_masks, labels = [], [], [], [], []
    for i in tqdm(range(len(base_data)), desc='Building Tensor data'):
        base = base_data[i]
        elmo, glove, extra = None, None, None
        if args.use_elmo:
            elmo = data_elmo[i]
        if args.use_glove:
            glove = data_glove[i]
        if args.use_extra_feature:
            extra = data_extra[i]
        cur_node_len = truncate_node_and_edge(args, base, elmo, glove, extra)
        cur_query_len = truncate_query(args, base, elmo, glove, extra)
        adj_data = build_edge_data(args, base)
        if args.use_elmo:
            elmo_data = build_elmo_data(args, elmo)
            data_elmo[i] = elmo_data
        if args.use_glove:
            glove_data = build_glove_data(args, glove)
            data_glove[i] = glove_data
        if args.use_extra_feature:
            extra_data = build_extra_feature(args, extra)
            data_extra[i] = extra_data
        output_masks.append(build_output_nodes_mask(args, base))
        cur_nodes_mask = torch.zeros(args.max_nodes)
        cur_nodes_mask[:cur_node_len] = 1
        nodes_mask.append(cur_nodes_mask)
        query_lengths.append(cur_query_len)
        adjs.append(adj_data)
        labels.append(base['answer_candidate_id'])
    data_size = len(base_data)
    tensor_dataset = [torch.cat([x.unsqueeze(0) for x in adjs], dim=0)]
    nodes_elmo, query_elmo = torch.zeros(data_size, 1, dtype=torch.bool), torch.zeros(data_size, 1, dtype=torch.bool)
    if args.use_elmo:
        nodes_elmo = torch.cat([x[0].unsqueeze(0) for x in data_elmo], dim=0)
        query_elmo = torch.cat([x[1].unsqueeze(0) for x in data_elmo], dim=0)
    tensor_dataset.extend([nodes_elmo, query_elmo])
    nodes_glove, query_glove = torch.zeros(data_size, 1, dtype=torch.bool), torch.zeros(data_size, 1, dtype=torch.bool)
    if args.use_glove:
        nodes_glove = torch.cat([x[0].unsqueeze(0) for x in data_glove], dim=0)
        query_glove = torch.cat([x[1].unsqueeze(0) for x in data_glove], dim=0)
    tensor_dataset.extend([nodes_glove, query_glove])
    nodes_ner, nodes_pos = torch.zeros(data_size, 1, dtype=torch.bool), torch.zeros(data_size, 1, dtype=torch.bool)
    query_ner, query_pos = torch.zeros(data_size, 1, dtype=torch.bool), torch.zeros(data_size, 1, dtype=torch.bool)
    if args.use_extra_feature:
        nodes_ner = torch.cat([x[0].unsqueeze(0) for x in data_extra], dim=0)
        nodes_pos = torch.cat([x[1].unsqueeze(0) for x in data_extra], dim=0)
        query_ner = torch.cat([x[2].unsqueeze(0) for x in data_extra], dim=0)
        query_pos = torch.cat([x[3].unsqueeze(0) for x in data_extra], dim=0)
    tensor_dataset.extend([nodes_ner, nodes_pos, query_ner, query_pos])
    tensor_dataset.append(torch.cat([x.unsqueeze(0) for x in nodes_mask], dim=0))
    tensor_dataset.append(torch.LongTensor(query_lengths))
    tensor_dataset.append(torch.cat([x.unsqueeze(0) for x in output_masks], dim=0))
    tensor_dataset.append(torch.LongTensor(labels))
    return tensor_dataset

'''Build the list of ids and corresponding answer candidate text so as to generate the prediction json file'''
def build_id_candidate_list(base_data):
    res = []
    for d in base_data:
        res.append((d['id'], [' '.join(token) for token in d['candidates']]))
    return res

def get_pytorch_dataloader(args, file_name_prefix, shuffle=False, for_evaluation=True):
    base_data, data_elmo, data_glove, data_extra, ner_dict, pos_dict = load_raw_data(args, file_name_prefix)
    id_candidate_list = None
    if for_evaluation:
        id_candidate_list = build_id_candidate_list(base_data)
    tensor_dataset = build_tensor_data(args, base_data, data_elmo, data_glove, data_extra)
    tensor_dataset = TensorDataset(*tensor_dataset)
    data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=shuffle)
    return data_loader, len(ner_dict), len(pos_dict), id_candidate_list