import argparse
import os
import json
import pickle
import numpy as np
import nltk
import spacy
import unicodedata

from allennlp.commands.elmo import ElmoEmbedder
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA
from utils.str2bool import str2bool
from utils.ConfigLogger import config_logger
from nltk.tokenize import TweetTokenizer

nltk.internals.config_java(options='-Xmx4g')

class Preprocesser:
    def __init__(self, file_name, logger, is_masked=False, use_elmo=True, use_glove=True, use_extra_feature=True,
            use_bert=True, options_file=None, weight_file=None, glove_file=None):
        self.file_name = file_name
        self.is_masked = is_masked
        self.use_elmo = use_elmo
        self.use_glove = use_glove
        self.use_extra_feature = use_extra_feature
        self.use_bert = use_bert
        self.options_file = options_file
        self.weight_file = weight_file
        self.glove_file = glove_file
        self.tokenizer = TweetTokenizer()
        self.nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser'])
        self.tag_dict = {'<PAD>': 0, '<UNK>': 1, '<POS>': 2, '<EOS>': 3}
        self.ner_dict = {'<PAD>': 0, '<UNK>': 1, '<POS>': 2, '<EOS>': 3}
        self.logger = logger
        self.elmo_split_interval = 16
        self.bert_split_interval = 4
        self.max_support_length = 512

    """ check whether string of current span matches the candidate
    """
    def check(self, support, word_index, candidate, for_unmarked=False):
        if for_unmarked:
            return sum(
                [self.is_contain_special_symbol(c_, support[word_index + j].lower()) for j, c_ in enumerate(candidate) if
                 word_index + j < len(support)]) == len(candidate)
        else:
            return sum([support[word_index + j].lower() == c_ for j, c_ in enumerate(candidate) if
                        word_index + j < len(support)]) == len(candidate)


    def is_contain_special_symbol(self, candidate_tok, support_tok):
        if candidate_tok.isdigit():
            return support_tok.find(candidate_tok) >= 0
        else:
            return support_tok == candidate_tok or candidate_tok + 's' == support_tok or \
                   (support_tok.find(candidate_tok) >= 0 and (
                               support_tok.find('-') > 0 or support_tok.find('\'s') > 0 or
                               support_tok.find(',') > 0))

    """ Check whether the mask is valid via its length
    """
    def check_masked(self, support, word_index, candidate):
        return sum([support[word_index + j] == c_ for j, c_ in enumerate(candidate) if
                    word_index + j < len(support)]) == len(candidate)

    """ generating index for candidates in the original document
    """
    def ind(self, support_index, word_index, candidate_index, candidate, marked_candidate):
        marked_candidate[candidate_index] = True
        return [[support_index, word_index + i, candidate_index] for i in range(len(candidate))]

    """ some candidates may not be found in the original document so we have to merge it with the node masks who were 
        found in original document
    """
    def merge_two_masks(self, mask, unmarked_mask):
        for i in range(len(mask)):
            if len(unmarked_mask[i]) != 0:
                if len(mask[i]) == 0:
                    mask[i] = unmarked_mask[i]
                else:
                    for unmarked_index in range(len(unmarked_mask[i])):
                        mask[i].append(unmarked_mask[i][unmarked_index])
                    mask[i].sort(key=lambda x: x[0][1])
        return mask

    """ if some new POS or NER tags are found in data, we need to merge it with previous POS or NER dict
    """
    def mergeTwoDictFile(self, file_name, dict):
        with open(file_name, 'rb') as f:
            prev_dict = pickle.load(f)
        for name in dict:
            if not prev_dict.__contains__(name):
                prev_dict[name] = len(prev_dict)
        with open(file_name, 'wb') as f:
            pickle.dump(prev_dict, f)

    """ The main function to pre-processing WIKIHOP dataset and save it as several pickle files
    """
    def preprocess(self):
        preprocess_graph_file_name = self.file_name
        preprocess_elmo_file_name = self.file_name + '.elmo'
        preprocess_glove_file_name = self.file_name + '.glove'
        preprocess_extra_file_name = self.file_name + '.extra'
        supports = self.doPreprocessForGraph(preprocess_graph_file_name)
        with open('{}.preprocessed.pickle'.format(preprocess_graph_file_name), 'rb') as f:
            data_graph = [d for d in pickle.load(f)]
        # text data including supporting documents, queries and node mask
        text_data = []
        for index, graph_d in enumerate(data_graph):
            tmp = {}
            tmp['query'] = graph_d['query']
            tmp['query_full_token'] = graph_d['query_full_token']
            tmp['nodes_mask'] = graph_d['nodes_mask']
            tmp['candidates'] = graph_d['candidates']
            tmp['nodes_candidates_id'] = graph_d['nodes_candidates_id']
            tmp['supports'] = supports[index]['supports']
            text_data.append(tmp)
        if self.use_elmo:
            self.doPreprocessForElmo(text_data, preprocess_elmo_file_name)
        if self.use_glove:
            self.doPreprocessForGlove(text_data, preprocess_glove_file_name)
        if self.use_extra_feature:
            self.doPreprocessForExtraFeature(text_data, preprocess_extra_file_name)

    """ Build entity graph base on input json data and save graph as a pickle
    """
    def doPreprocessForGraph(self, preprocess_graph_file_name):
        with open(self.file_name, 'r') as f:
            data = json.load(f)
            self.logger.info('Load json file:' + self.file_name)
            supports = self.doPreprocess(data, mode='supports')
        if not os.path.isfile('{}.preprocessed.pickle'.format(preprocess_graph_file_name)):
            self.logger.info('Preprocsssing Json data for Graph....')
            data = self.doPreprocess(data, mode='graph', supports=supports)
            self.logger.info('Preprocessing Graph data finished')
            with open('{}.preprocessed.pickle'.format(preprocess_graph_file_name), 'wb') as f:
                pickle.dump(data, f)
                self.logger.info('Successfully save preprocessed Graph data file %s',
                        '{}.preprocessed.pickle'.format(preprocess_graph_file_name))
        else:
            self.logger.info('Preprocessed Graph data is already existed, no preprocessing will be executed.')
        return supports

    """ Generating pickle file for ELMo embeddings of queries and nodes in graph
    """
    def doPreprocessForElmo(self, text_data, preprocess_elmo_file_name):
        if not os.path.isfile('{}.preprocessed.pickle'.format(preprocess_elmo_file_name)):
            elmoEmbedder = ElmoEmbedder(cuda_device=0, options_file=self.options_file, weight_file=self.weight_file)
            self.logger.info('Preprocsssing Json data for Elmo....')
            data = self.doPreprocess(text_data, mode='elmo', ee=elmoEmbedder)
            self.logger.info('Preprocessing Elmo data finished')
            with open('{}.preprocessed.pickle'.format(preprocess_elmo_file_name), 'wb') as f:
                pickle.dump(data, f)
                self.logger.info('Successfully save preprocessed Elmo data file %s',
                        '{}.preprocessed.pickle'.format(preprocess_elmo_file_name))
        else:
            self.logger.info('Preprocessed Elmo data is already existed, no preprocessing will be executed.')

    """ Generating pickle file for GLoVe embeddings of queries and nodes in graph
    """
    def doPreprocessForGlove(self, text_data, preprocess_glove_file_name):
        if self.use_glove:
            if not os.path.isfile('{}.preprocessed.pickle'.format(preprocess_glove_file_name)):
                self.logger.info('Building vocabulary dictionary....')
                vocab2index, index2vocab = self.buildVocabMap(text_data)
                self.logger.info('Building GloVe Embedding Map....')
                gloveEmbMap = self.buildGloveEmbMap(vocab2index)
                self.logger.info('Prerpocessing Json data for Glove....')
                data_glove = self.doPreprocess(text_data, mode='glove', gloveEmbMap=gloveEmbMap, vocab2index=vocab2index)
                with open('{}.preprocessed.pickle'.format(preprocess_glove_file_name), 'wb') as f:
                    pickle.dump(data_glove, f)
                    self.logger.info('Successfully save preprocessed Glove data file %s',
                            '{}.preprocessed.pickle'.format(preprocess_glove_file_name))
            else:
                self.logger.info('Preprocessed Glove data is already existed, no preprocessing will be executed.')

    """ Generating pickle file for extra feature (NER, POS) of nodes and queries
    """
    def doPreprocessForExtraFeature(self, text_data, preprocess_extra_file_name):
        if not os.path.isfile('{}.preprocessed.pickle'.format(preprocess_extra_file_name)):
            data_extra = self.doPreprocess(text_data, mode='extra')
            with open('{}.preprocessed.pickle'.format(preprocess_extra_file_name), 'wb') as f:
                pickle.dump(data_extra[0], f)
                self.logger.info('Successfully save preprocessed Extra feature data file %s',
                    '{}.preprocessed.pickle'.format(preprocess_extra_file_name))
            if not os.path.isfile('data/pos_dict.pickle'):
                with open('data/pos_dict.pickle', 'wb') as f:
                    pickle.dump(data_extra[2], f)
                    self.logger.info('Successfully save pos dict data file pos_dict.pickle')
            else:
                self.mergeTwoDictFile('data/pos_dict.pickle', data_extra[2])
                self.logger.info('Successfully merge current pos dict with pos_dict.pickle')
            if not os.path.isfile('data/ner_dict.pickle'):
                with open('data/ner_dict.pickle', 'wb') as f:
                    pickle.dump(data_extra[1], f)
                    self.logger.info('Successfully save ner dict data file ner_dict.pickle')
            else:
                self.mergeTwoDictFile('data/ner_dict.pickle', data_extra[1])
                self.logger.info('Successfully merge current ner dict with ner_dict.pickle')
        else:
            self.logger.info('Preprocessed Extra data is already existed, no preprocessing will be executed.')

    """ Core preprocessing function
    """
    def doPreprocess(self, data_mb, mode, supports=None, ee=None, gloveEmbMap=None, vocab2index=None, bert_model=None):
        data_gen = []
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(data_mb)).start()

        data_count = 0
        for index, data in enumerate(data_mb):
            # if index < 19999:
            #     data_count += 1
            #     continue
            # try:
            if mode == 'supports':
                tmp = {}
                tmp['supports'] = [self.tokenizer.tokenize(support) for support in data['supports']]
                for index in range(len(tmp['supports'])):
                    if len(tmp['supports'][index]) > self.max_support_length:
                        tmp['supports'][index] = tmp['supports'][index][:self.max_support_length]
                data_gen.append(tmp)
            elif mode == 'graph':
                preprocessGraphData = self.preprocessForGraph(data, supports[index]['supports'])
                data_gen.append(preprocessGraphData)
            elif mode == 'elmo':
                preprocessElmoData = self.preprocessForElmo(data, ee)
                data_gen.append(preprocessElmoData)
            elif mode == 'glove':
                preprocessGloveData = self.preprocessForGlove(data, gloveEmbMap, vocab2index)
                data_gen.append(preprocessGloveData)
            elif mode == 'extra':
                preprocessExtraData = self.preprocessForExtra(data)
                data_gen.append(preprocessExtraData)
            elif mode == 'bert':
                preprocessBertData = self.preprocessForBert(data, bert_model)
                data_gen.append(preprocessBertData)
            # except:
            #     print(index)
            #     pass
            # data_count += 1
            # pbar.update(data_count)
            # if data_count >= 96:
            #     break
        pbar.finish()
        if mode == 'extra':
            return [data_gen, self.ner_dict, self.tag_dict]
        return data_gen

    """ build vocabulary map for all tokens included in dataset
    """
    def buildVocabMap(self, data_elmo):
        vocab2index, index2vocab = {}, {}
        count = 1
        vocab2index['unk'] = 0
        index2vocab[0] = 'unk'
        for data_mb in data_elmo:
            for candidate in data_mb['candidates']:
                for token in candidate:
                    if not vocab2index.__contains__(token):
                        vocab2index[token] = count
                        index2vocab[count] = token
                        count += 1
            for token in data_mb['query']:
                if not vocab2index.__contains__(token):
                    vocab2index[token] = count
                    index2vocab[count] = token
                    count += 1
            for token in data_mb['query_full_token']:
                if not vocab2index.__contains__(token):
                    vocab2index[token] = count
                    index2vocab[count] = token
                    count += 1
        return vocab2index, index2vocab

    """ The core function to build graph
    """
    def preprocessForGraph(self, data, supports):
        if data.__contains__('annotations'):
            data.pop('annotations')

        ## The first token in the query is combined with underline so we have to divided it into several words by
        ## removing underlines
        first_blank_pos = data['query'].find(' ')
        if first_blank_pos > 0:
            first_token_in_query = data['query'][:first_blank_pos]
        else:
            first_token_in_query = data['query']
        query = data['query'].replace('_', ' ')
        data['query'] = self.tokenizer.tokenize(query)
        ## query_full_token means split the relation word in query based on "_"
        data['query_full_token'] = query

        candidates_orig = list(data['candidates'])

        data['candidates'] = [self.tokenizer.tokenize(candidate) for candidate in data['candidates']]

        marked_candidate = {}

        ## find all matched candidates in documents and mark their positions
        if self.is_masked:
            mask = [[self.ind(sindex, windex, cindex, candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(data['candidates'])
                     if self.check_masked(support, windex, candidate)] for sindex, support in
                    enumerate(supports)]
        else:
            mask = [[self.ind(sindex, windex, cindex, candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(data['candidates'])
                     if self.check(support, windex, candidate)] for sindex, support in enumerate(supports)]
            tok_unmarked_candidates = []
            unmarked_candidates_index_map = {}
            for candidate_index in range(len(data['candidates'])):
                if not marked_candidate.__contains__(candidate_index):
                    tok_unmarked_candidates.append(data['candidates'][candidate_index])
                    unmarked_candidates_index_map[len(tok_unmarked_candidates) - 1] = candidate_index
            if len(tok_unmarked_candidates) != 0:
                unmarked_mask = [
                    [self.ind(sindex, windex, unmarked_candidates_index_map[cindex], candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(tok_unmarked_candidates)
                     if self.check(support, windex, candidate, for_unmarked=True)] for sindex, support in
                    enumerate(supports)]
                mask = self.merge_two_masks(mask, unmarked_mask)

        nodes_id_name = []
        count = 0
        for e in [[[x[-1] for x in c][0] for c in s] for s in mask]:
            u = []
            for f in e:
                u.append((count, f))
                count += 1

            nodes_id_name.append(u)

        data['nodes_candidates_id'] = [[node_triple[-1] for node_triple in node][0]
                                       for nodes_in_a_support in mask for node in nodes_in_a_support]

        ## find two kinds of edges between nodes
        ## edges_in means nodes within a document, edges_out means nodes with same string across different document
        edges_in, edges_out = [], []
        for e0 in nodes_id_name:
            for f0, w0 in e0:
                for f1, w1 in e0:
                    if f0 != f1:
                        edges_in.append((f0, f1))

                for e1 in nodes_id_name:
                    for f1, w1 in e1:
                        if e0 != e1 and w0 == w1:
                            edges_out.append((f0, f1))

        data['edges_in'] = edges_in
        data['edges_out'] = edges_out

        data['nodes_mask'] = mask

        data['relation_index'] = len(first_token_in_query)
        for index, answer in enumerate(candidates_orig):
            if answer == data['answer']:
                data['answer_candidate_id'] = index
                break
        return data

    """ gerating ELMo embeddings for nodes and query
    """
    def preprocessForElmo(self, text_data, ee):
        data_elmo = {}

        mask_ = [[x[:-1] for x in f] for e in text_data['nodes_mask'] for f in e]
        supports, query, query_full_tokens = text_data['supports'], text_data['query'], text_data['query_full_token']
        first_tokens_in_query = query[0].split('_')

        split_interval = self.elmo_split_interval
        if len(supports) <= split_interval:
            candidates, _ = ee.batch_to_embeddings(supports)
            candidates = candidates.data.cpu().numpy()
        else:
            ## split long support data into several parts to avoid possible OOM
            count = 0
            candidates = None
            while count < len(supports):
                current_candidates, _ = \
                    ee.batch_to_embeddings(supports[count:min(count + split_interval, len(supports))])
                current_candidates = current_candidates.data.cpu().numpy()
                if candidates is None:
                    candidates = current_candidates
                else:
                    if candidates.shape[2] > current_candidates.shape[2]:
                        current_candidates = np.pad(current_candidates,
                            ((0, 0), (0, 0), (0, candidates.shape[2] - current_candidates.shape[2]), (0, 0)), 'constant')
                    elif current_candidates.shape[2] > candidates.shape[2]:
                        candidates = np.pad(candidates,
                            ((0, 0), (0, 0), (0, current_candidates.shape[2] - candidates.shape[2]), (0, 0)), 'constant')
                    candidates = np.concatenate((candidates, current_candidates))
                count += split_interval

        data_elmo['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[np.array(m).T.tolist()]).astype(np.float16)
                              for m in mask_]

        query, _ = ee.batch_to_embeddings([query])
        query = query.data.cpu().numpy()
        data_elmo['query_elmo'] = (query.transpose((0, 2, 1, 3))).astype(np.float16)[0]
        if len(first_tokens_in_query) == 1:
            data_elmo['query_full_token_elmo'] = data_elmo['query_elmo']
        else:
            query_full_tokens, _ = ee.batch_to_embeddings([first_tokens_in_query])
            query_full_tokens = query_full_tokens.cpu().numpy()
            data_elmo['query_full_token_elmo'] = np.concatenate(
                    (query_full_tokens.transpose((0, 2, 1, 3)).astype(np.float16)[0], data_elmo['query_elmo'][1:,:,:]), 0)
        return data_elmo

    """ generating GLoVe for nodes and query
    """
    def preprocessForGlove(self, data_elmo, gloveEmbMap, vocab2index):
        data = {}
        nodes_glove = []
        for candidate_id in data_elmo['nodes_candidates_id']:
            candidate_token = data_elmo['candidates'][candidate_id]
            node_glove = []
            for token in candidate_token:
                node_glove.append(gloveEmbMap[vocab2index[token]])
            nodes_glove.append(np.array(node_glove).astype(np.float32))
        data['nodes_glove'] = nodes_glove
        query_glove = []
        for token in data_elmo['query']:
            query_glove.append(gloveEmbMap[vocab2index[token]])
        data['query_glove'] = np.array(query_glove).astype(np.float32)
        query_full_token_glove = []
        for token in data_elmo['query_full_token']:
            query_full_token_glove.append(gloveEmbMap[vocab2index[token]])
        data['query_full_token_glove'] = np.array(query_full_token_glove).astype(np.float32)
        return data

    """ generating POS and NER tags for every token in nodes or query
    """
    def preprocessForExtra(self, data):
        nodes_mask = data['nodes_mask']
        supports, query = data['supports'], data['query']
        recovered_support = []
        for support in supports:
            recovered_support.append(self.recoverTokens(support))
        tokened_supports = [doc for doc in self.nlp.pipe(recovered_support, batch_size=1000)]
        recovered_query = self.recoverTokens(query)
        tokened_query = [doc for doc in self.nlp.pipe([recovered_query], batch_size=1000)]
        tag_dict = self.tag_dict
        ner_dict = self.ner_dict
        pos = [self.postagFunc(tokened_support, tag_dict) for tokened_support in tokened_supports]
        ner = [self.nertagFunc(tokened_support, ner_dict) for tokened_support in tokened_supports]
        query_pos = self.postagFunc(tokened_query[0], tag_dict)
        query_ner = self.nertagFunc(tokened_query[0], ner_dict)
        nodes_ner, nodes_pos = [], []
        ## give each node a ner and pos tag and the order in each sample is the same as it in Elmo data
        for support_index, support_nodes in enumerate(nodes_mask):
            for node_mask in support_nodes:
                node_pos, node_ner = [], []
                for mask in node_mask:
                    if mask[1] > len(pos[support_index]) - 1:
                        node_pos.append(tag_dict['<UNK>'])
                        node_ner.append(tag_dict['<UNK>'])
                    else:
                        node_pos.append(pos[support_index][mask[1]])
                        node_ner.append(ner[support_index][mask[1]])
                nodes_ner.append(np.array(node_ner))
                nodes_pos.append(np.array(node_pos))

        data = {'nodes_ner': nodes_ner, 'nodes_pos': nodes_pos, 'query_ner': query_ner, 'query_pos': query_pos}
        return data

    """ Read GLoVe file and generating a dict mapping it from token to embedding
    """
    def buildGloveEmbMap(self, vocab2index, dim=300):
        vocab_size = len(vocab2index)
        emb = np.zeros((vocab_size, dim))
        emb[0] = 0
        unknown_mask = np.zeros(vocab_size, dtype=bool)
        with open(self.glove_file, encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                elems = line.split()
                token = self.normalize_text(' '.join(elems[0:-dim]))
                if token in vocab2index:
                    emb[vocab2index[token]] = [float(v) for v in elems[-dim:]]
                    unknown_mask[vocab2index[token]] = True
        for index, mask in enumerate(unknown_mask):
            if not mask:
                emb[index] = emb[0]
        return emb

    def normalize_text(self, text):
        return unicodedata.normalize('NFD', text)

    def recoverTokens(self, tokens):
        res = ''
        for token in tokens:
            res = res + token + ' '
        return res[:-1]

    def tagFunc(self, toks, pos_dict):
        postag = []
        for w in toks:
            if not pos_dict.__contains__(w[1]):
                current_index = len(pos_dict)
                pos_dict[w[1]] = current_index
            postag.append(pos_dict[w[1]])
        return postag, pos_dict

    def postagFunc(self, toks, tag_dict):
        postag = []
        for w in toks:
            if len(w.text) > 0:
                if not tag_dict.__contains__(w.tag_):
                    current_index = len(tag_dict)
                    tag_dict[w.tag_] = current_index
                postag.append(tag_dict[w.tag_])
        return postag

    def nertagFunc(self, toks, ner_dict):
        nertag = []
        for w in toks:
            if len(w.text) > 0:
                ner_type = '{}_{}'.format(w.ent_type_, w.ent_iob_)
                if not ner_dict.__contains__(ner_type):
                    current_index = len(ner_dict)
                    ner_dict[ner_type] = current_index
                nertag.append(ner_dict[ner_type])
        return nertag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('--is_masked', type=str2bool, default=False, help='using masked data or not')
    parser.add_argument('--use_glove', type=str2bool, default=True, help='Using Glove embedding or not')
    parser.add_argument('--use_extra_feature', type=str2bool, default=True, help='Using extra feature, e.g. '+
            'NER, POS, ')

    args = parser.parse_args()
    file_name = args.file_name
    is_masked = args.is_masked
    use_glove = args.use_glove
    use_extra_feature = args.use_extra_feature
    options_file = 'data/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = 'data/elmo_2x4096_512_2048cnn_2xhighway_weights'
    glove_file = 'data/glove.840B.300d.txt'
    logger = config_logger('Preprocess')

    preprocesser = Preprocesser(file_name, logger, is_masked=is_masked,
            use_glove=use_glove, use_extra_feature=use_extra_feature, options_file=options_file,
            weight_file=weight_file, glove_file=glove_file)
    preprocesser.preprocess()
