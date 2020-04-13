# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import sys
import json

from utils.ConfigLogger import config_logger
from utils.str2bool import str2bool
from utils.pytorch_dataset import get_pytorch_dataloader
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, RunningAverage
from ignite.contrib.handlers import ProgressBar, LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

'''Feature layer who takes raw feature as input'''
class FeatureLayer(nn.Module):
    def __init__(self, args):
        super(FeatureLayer, self).__init__()
        self.use_elmo = args.use_elmo
        self.use_glove = args.use_glove
        self.use_extra_feature = args.use_extra_feature
        self.max_query_size = args.max_query_size
        self.max_nodes = args.max_nodes
        self.query_encoding_type = args.query_encoding_type
        self.encoding_size = args.encoding_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.encoder_input_size = 0
        if self.use_elmo:
            self.encoder_input_size += 3 * 1024
        if self.use_glove:
            self.encoder_input_size += 300
        if self.query_encoding_type == 'lstm':
            self.query_encoder = nn.LSTM(self.encoder_input_size, int(self.encoding_size / 2), 2, bidirectional=True,
                    batch_first=True)
        else:
            self.query_encoder = nn.Linear(self.encoder_input_size, self.encoding_size)
        self.node_encoder = nn.Linear(self.encoder_input_size, self.encoding_size)
        args.hidden_size = self.encoding_size
        if self.use_extra_feature:
            self.ner_embedding = nn.Embedding(args.ner_dict_size, args.ner_emb_size)
            self.pos_embedding = nn.Embedding(args.pos_dict_size, args.pos_emb_size)
            args.hidden_size += (args.ner_emb_size + args.pos_emb_size)

    def forward(self, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos, query_ner, query_pos,
                query_lengths):
        query_flat, nodes_flat = None, None
        if self.use_elmo:
            query_flat = query_elmo.view(-1, self.max_query_size, 3 * 1024)
            nodes_flat = nodes_elmo.view(-1, self.max_nodes, 3 * 1024)
        if self.use_glove:
            if query_flat is None:
                query_flat, nodes_flat = query_glove, nodes_glove
            else:
                query_flat = torch.cat((query_flat, query_glove), dim=-1)
                nodes_flat = torch.cat((nodes_flat, nodes_glove), dim=-1)
        if self.query_encoding_type == 'lstm':
            query_flat = nn.utils.rnn.pack_padded_sequence(query_flat, query_lengths, batch_first=True,
                                                           enforce_sorted=False)
            query_compress = F.tanh(
                torch.nn.utils.rnn.pad_packed_sequence(self.query_encoder(query_flat)[0], batch_first=True)[0])
        else:
            query_compress = F.tanh(self.query_encoder(query_flat))
        nodes_compress = F.tanh(self.node_encoder(nodes_flat))
        if self.use_extra_feature:
            query_ner_emb = self.ner_embedding(query_ner)
            query_pos_emb = self.pos_embedding(query_pos)
            nodes_ner_emb = self.ner_embedding(nodes_ner)
            nodes_pos_emb = self.pos_embedding(nodes_pos)
            nodes_compress = torch.cat([nodes_compress, nodes_ner_emb, nodes_pos_emb], dim=-1)
            new_query_length = query_compress.shape[1]
            query_compress = torch.cat([query_compress, query_ner_emb[:, :new_query_length, :], query_pos_emb[:, :new_query_length, :]], dim=-1)
        return nodes_compress, query_compress

'''A single gated GCN layer'''
class GCNLayer(nn.Module):
    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.use_edge = args.use_edge
        if args.use_edge:
            self.gcns = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(3)])
        else:
            self.gcns = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size)])
        self.update_gate = nn.Linear(args.hidden_size, args.hidden_size)
        self.att_gate = nn.Linear(args.hidden_size * 2, args.hidden_size)

    def forward(self, adj, nodes_hidden, nodes_mask):
        accumulated_nodes_hidden = torch.stack([gcn(nodes_hidden) for gcn in self.gcns], dim=1) * \
                                   nodes_mask.unsqueeze(1).unsqueeze(-1)
        update = torch.sum(torch.matmul(adj, accumulated_nodes_hidden), dim=1) + \
                 self.update_gate(nodes_hidden) * nodes_mask.unsqueeze(-1)
        att = F.sigmoid(self.att_gate(torch.cat([update, nodes_hidden], -1)))
        output = att * F.tanh(update) + (1 - att) * nodes_hidden
        return output

'''Bidirectional attention layer'''
class BiAttention(nn.Module):
    def __init__(self, args):
        super(BiAttention, self).__init__()
        self.max_nodes = args.max_nodes
        self.max_query_size = args.max_query_size
        self.attention_linear = nn.Linear(args.hidden_size * 3, 1, bias=False)

    def forward(self, nodes_compress, query_compress, nodes_hidden):
        query_size = query_compress.shape[1]
        expanded_query = query_compress.unsqueeze(1).repeat((1, self.max_nodes, 1, 1))
        expanded_nodes = nodes_compress.unsqueeze(2).repeat((1, 1, query_size, 1))
        nodes_query_similarity = expanded_nodes * expanded_query
        concatenated_data = torch.cat((expanded_nodes, expanded_query, nodes_query_similarity), -1)
        similarity = self.attention_linear(concatenated_data).squeeze(-1)
        nodes2query = torch.matmul(F.softmax(similarity, dim=-1), query_compress)
        b = F.softmax(torch.max(similarity, dim=-1)[0], dim=-1)
        query2nodes = torch.matmul(b.unsqueeze(1), nodes_compress).repeat(1, self.max_nodes, 1)
        attention_output = torch.cat(
            (nodes_compress, nodes2query, nodes_compress * nodes2query, nodes_compress * query2nodes), dim=-1)
        return attention_output

'''Output layer who takes output from attention layer and node output mask to generate the predictions'''
class OutputLayer(nn.Module):
    def __init__(self, ags):
        super(OutputLayer, self).__init__()
        self.linear1 = nn.Linear(args.hidden_size * 4, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, attention_output, output_mask):
        raw_preds = self.linear2(F.tanh(self.linear1(attention_output))).squeeze(-1)
        preds = output_mask.float() * raw_preds.unsqueeze(1)
        preds = preds.masked_fill(preds == 0.0, -float("inf"))
        preds = torch.max(preds, dim=-1)[0]
        return preds

'''BAG Model class'''
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.feature_layer = FeatureLayer(args)
        self.gcn_layers = nn.ModuleList([GCNLayer(args) for _ in range(args.hop_num)])
        self.bi_attention = BiAttention(args)
        self.output_layer = OutputLayer(args)

    def forward(self, adj, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos, query_ner, query_pos,
                query_lengths, nodes_mask, output_mask):
        nodes_compress, query_compress = self.feature_layer(nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner,
                nodes_pos, query_ner, query_pos, query_lengths)
        nodes_hidden = nodes_compress
        for gcn_layer in self.gcn_layers:
            nodes_hidden = gcn_layer(adj, nodes_hidden, nodes_mask)
        attention_output = self.bi_attention(nodes_compress, query_compress, nodes_hidden)
        preds = self.output_layer(attention_output, output_mask)
        return preds

""" Check whether the preprocessed file existed in current directory
"""
def checkPreprocessFile(file_name, add_query_node):
    preprocess_file_name = file_name
    if add_query_node:
        preprocess_file_name = preprocess_file_name + '.add_query_node'
    if not os.path.isfile('{}.preprocessed.pickle'.format(preprocess_file_name)):
        return preprocess_file_name, False
    return preprocess_file_name, True

def write_best_accuray_txt(model_path, accuracy):
    file_name = model_path + '/best.txt'
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as f:
        f.write(str(accuracy) + '\n')
    f.close()

def generate_answer_json(answer_dict, in_file):
    with open(in_file, 'r') as f:
        data = json.load(f)
    final_answer_dict = {}
    for d in data:
        final_answer_dict[d['id']] = d['candidates'][answer_dict[d['id']]]
    with open('predictions.json', 'w') as f:
        json.dump(final_answer_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('dev_file', type=str)
    parser.add_argument('--model_checkpoint', type=str, default=None, help='The initial model checkpoint')
    parser.add_argument('--is_masked', type=bool, default=False, help='using masked data or not')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue last train using saved model or' +
            ' start a new training')
    parser.add_argument('--cpu_number', type=int, default=4, help='The number of CPUs used in current script')
    parser.add_argument('--add_query_node', type=bool, default=False, help='Whether the entity in query is involved '+
            'in the construction of graph')
    parser.add_argument('--evaluation_mode', type=bool, default=False, help='Whether using evaluation mode')
    parser.add_argument('--use_elmo', type=str2bool, default=True, help='Whether use the ELMo as a feature for each node')
    parser.add_argument('--use_glove', type=str2bool, default=True, help='Whether use the GloVe as a feature for each node')
    parser.add_argument('--use_extra_feature', type=str2bool, default=True, help='Whether use extra feature for ' +
            'each node, e.g. NER and POS')
    parser.add_argument('--use_edge', type=str2bool, default=True, help='Whether use edges in graph')
    parser.add_argument('--use_full_query_token', type=str2bool, default=False, help='Tokens in query will be splitted by underlines')
    parser.add_argument('--dynamic_change_learning_rate', type=str2bool, default=True, help='Whether the learning rate will change along training')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--hop_num', type=int, default=5, help='Hop num in GCN layer, in other words the depth of GCN')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch number for the training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--info_interval', type=int, default=1000, help='The interval to display training loss info')
    parser.add_argument('--dropout', type=float, default=0.8, help="Keep rate for dropout in model")
    parser.add_argument('--encoding_size', type=int, default=512, help='The encoding output size for both query and nodes')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='The hidden size for lstm intermediate layer')
    parser.add_argument('--pos_emb_size', type=int, default=8, help='The size of POS embedding')
    parser.add_argument('--ner_emb_size', type=int, default=8, help='The size of NER embedding')
    parser.add_argument('--query_encoding_type', type=str, default='lstm', help='The function to encode query')
    parser.add_argument('--max_nodes', type=int, default=500, help='Maximum node number in graph')
    parser.add_argument('--max_query_size', type=int, default=25, help='Maximum length for query')
    parser.add_argument('--max_candidates', type=int, default=80, help='Maximum number of answer candidates')
    parser.add_argument('--max_candidates_len', type=int, default=10, help='Maximum length for a candidates')
    parser.add_argument('--loss_log_interval', type=int, default=1000, help='Iteration interval to print loss')
    parser.add_argument('--patience', type=int, default=5, help='Epoch early stopping patience')

    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    encoding_type_map = {'lstm':'lstm','linear':'linear'}
    best_res = {'acc': 0.0}
    pred_res = []

    model_name = 'BAG-pytorch'
    if args.evaluation_mode:
        logger = config_logger('evaluation-pytorch/' + model_name)
    else:
        logger = config_logger('BAG-pytorch')

    model_path = os.getcwd() + '/models-pytorch/' + model_name + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, 'best_model.bin')

    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    '''Check whether preprocessed files are existed'''
    train_file_name_prefix, fileExist = checkPreprocessFile(args.in_file, args.add_query_node)
    if not fileExist:
        logger.info('Cannot find preprocess data %s, program will shut down.',
            '{}.preprocessed.pickle'.format(train_file_name_prefix))
        sys.exit()
    dev_file_name_prefix, fileExist = checkPreprocessFile(args.dev_file, args.add_query_node)
    if not fileExist:
        logger.info('Cannot find preprocess data %s, program will shut down.',
            '{}.preprocessed.pickle'.format(dev_file_name_prefix))
        sys.exit()
    args.pos_dict_size, args.ner_dict_size = 0, 0
    dev_data_loader, args.ner_dict_size, args.pos_dict_size, id_candidate_list = \
            get_pytorch_dataloader(args, dev_file_name_prefix, for_evaluation=True)

    '''Initialize the BAG model'''
    model = Model(args)
    if args.model_checkpoint is not None:
        model.load_state_dict(torch.load(args.model_checkpoint))
        logger.info('Load previous model from %s', args.model_checkpoint)

    '''Handler function for training'''
    def train(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        adj, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos, query_ner, query_pos, nodes_mask, \
                query_lengths, output_mask, labels = batch
        preds = model(adj, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos, query_ner, query_pos,
              query_lengths, nodes_mask, output_mask)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(preds, labels)
        if loss.item() == float("inf"):
            tmp_labels, tmp_preds = [], []
            for i in range(labels.shape[0]):
                if preds[i][labels[i]] != -float("inf"):
                    tmp_labels.append(labels[i].unsqueeze(0))
                    tmp_preds.append(preds[i].unsqueeze(0))
            loss = loss_fct(torch.cat(tmp_preds, dim=0), torch.cat(tmp_labels, dim=0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    '''Handler function for evaluation'''
    def evaluation(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            adj, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos, query_ner, query_pos, \
                    nodes_mask, query_lengths, output_mask, labels = batch
            preds = model(adj, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos, query_ner,
                          query_pos, query_lengths, nodes_mask, output_mask)
            if args.evaluation_mode:
                pred_res.extend(torch.argmax(preds, dim=1).tolist())
            return preds, labels
    evaluator = Engine(evaluation)
    Accuracy(output_transform=lambda x: (x[0], x[1])).attach(evaluator, 'accuracy')
    dev_pbar = ProgressBar(persist=True, desc='Validation')
    dev_pbar.attach(evaluator)

    '''Handler function to save best models after each evaluation'''
    def after_evaluation(engine):
        acc = engine.state.metrics['accuracy']
        logger.info('Evaluation accuracy on Epoch %d is %.3f', engine.state.epoch, acc * 100)
        if acc > best_res['acc']:
            logger.info('Current model BEATS the previous best model, previous best accuracy is %.5f', best_res['acc'])
            torch.save(model.state_dict(), model_path)
            logger.info('Best model has been saved')
        else:
            logger.info('Current model CANNOT BEAT the previous best model, previous best accuracy is %.5f',
                        best_res['acc'])

    def score_function(engine):
        return engine.state.metrics['accuracy']

    if not args.evaluation_mode:
        '''If current run is training'''
        train_data_loader, _, _, _ = get_pytorch_dataloader(args, train_file_name_prefix, shuffle=True)
        optimizer = Adam(model.parameters(), lr=args.lr)
        '''Learning rate decays every 5 epochs'''
        optimizer_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        scheduler = LRScheduler(optimizer_scheduler)
        trainer = Engine(train)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda _: evaluator.run(dev_data_loader))

        pbar = ProgressBar(persist=True, desc='Training')
        pbar.attach(trainer, metric_names=["loss"])
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args.loss_log_interval), lambda engine: \
                logger.info('Loss at iteration %d is %.5f', engine.state.iteration, engine.state.metrics['loss']))
        early_stop_handler = EarlyStopping(patience=args.patience, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, lambda engine: after_evaluation(engine))
        evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_data_loader, max_epochs=args.epochs)
    else:
        '''If current run is evaluation, it will generate prediction json file'''
        evaluator.run(dev_data_loader)
        evaluator.add_event_handler(Events.COMPLETED,
                lambda engine: logger.info('Current evaluation accuracy is %.3f', engine.state.metrics['accuracy'] * 100))
        pred_dict = {}
        for i, pred_label in enumerate(pred_res):
            pred_dict[id_candidate_list[i][0]] = id_candidate_list[i][1][pred_label]
        with open('predictions.json', 'w') as f:
            json.dump(pred_dict, f)
