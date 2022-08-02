''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
from itertools import chain


sys.path.append('build')
sys.path.append('r2r_src_MAF')

import MatterSim
import string
import json
import time
import math
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from param import args
import pickle
from MAF.utils.wordembeddings import WordEmbeddings
from MAF.utils.wordindexer import WordIndexer

# nlp = en_core_web_lg.load()

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')
config = {
    'split_file' : '/VL/space/zhan1624/obj-vln/tasks/R2R/dictionaries/split_dictionary.txt',
    'motion_indicator_file' : '/VL/space/zhan1624/obj-vln/tasks/R2R/dictionaries/motion_dict.txt',
    'stop_words_file': '/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/stop_words.txt',
    'position_file': '/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/data/spatial_position_dic.txt'
}
def split_oder(dictionary):
    return sorted(dictionary, key = lambda x: len(x.split()), reverse=True)

with open(config['split_file']) as f_dict:
    dictionary = f_dict.read().split('\n')
    dictionary = split_oder(dictionary)
    tmp_dict = list(filter(None,[each_phrase+"," for each_phrase in dictionary]))
    dictionary = dictionary + tmp_dict
    dictionary = [" "+each_phrase.strip()+" " for each_phrase in dictionary]

with open(config["stop_words_file"]) as f_stop_word:
    stopword = f_stop_word.read().split('\n')
    stopword = split_oder(stopword)

# later for motion indicator and landmarks
with open(config['motion_indicator_file']) as f_dict:
    motion_dict = f_dict.read().split('\n')
    motion_dict = [each_motion.strip() for each_motion in motion_dict]
    motion_dict = split_oder(motion_dict)

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5
    
    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3], 
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits):
    """
    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?x
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            with open('tasks/R2R/data/R2R_%s3.json' % split) as f:
                new_data = json.load(f)
        else:
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data

def get_configurations(sentence):
    sentence = sentence.lower().strip()
    sentence_list = sentence.split('.')
    sentence_list = [each_sentence.strip() for each_sentence in sentence_list]
    configs = []
    for each_sentence in sentence_list:  
        tmp_configs = []
        num = 0
        first_v = 0
        for id, token in enumerate(nlp(each_sentence)):
#             print(token.text, token.tag_)
            if token.tag_ == "VB" or token.tag_ == "VBP":
                if num == 0:
                    first_v = token.i
                tmp_configs.append("*")
                tmp_configs.append(token.text)
                num += 1
            else:
                tmp_configs.append(token.text)
        if first_v != 0:
            del tmp_configs[first_v]
            tmp_configs.insert(0,"*")
        tmp_configs = " ".join(tmp_configs).split('*')
        configs += tmp_configs
    configs1 = post_processing_sentence1(list(filter(None,configs)))
    configs = post_processing_sentence2(configs1)
    return configs

def post_processing_sentence2(sentence_list):
    special_startwith = ['are','is']
    new_sentence_list = []
    tmp_id = 100
    for id, sent in enumerate(sentence_list):
        if sent:
            for s_s in special_startwith:
                if sent.startswith(s_s):
                    tmp_sent = sentence_list[id-1] + " " + sent
                    del new_sentence_list[-1]
                    new_sentence_list.append(tmp_sent)
                    break
            else:
                new_sentence_list.append(sent.strip())
    return new_sentence_list

def post_processing_sentence1(sentence_list):
    special_endwith = ['that', 'you',"you \'ll", "you will need"]
    special_startwith = ['are','is']
    def func(sl):
        sl = sl.strip().strip(',').strip('.').strip()
        for sw in stopword:
            if sl.endswith(" %s"%sw):
                sl = sl[:-(len(sw)+1)]
            elif sl.endswith("%s"%sw):
                sl = sl[:-len(sw)]
            elif sl.startswith("%s "%sw) or sl.startswith(" %s"%sw):
                sl = sl[len(sw)+1:]
            elif sl.startswith("%s"%sw):
                sl = sl[len(sw):]
        return sl
    sentence_list = list(map(func, sentence_list))
    new_sentence_list = []
    tmp_id = 100
    flag = 0
    config_num = len(sentence_list)
    for id, sent in enumerate(sentence_list):
        if sent:
            for s_e in special_endwith:
                if sent.endswith(s_e) and id !=config_num-1:
                    tmp_sent = sent + " " + sentence_list[id+1]
                    tmp_id = id + 1
                    new_sentence_list.append(tmp_sent)
                    flag = 1
                    break
            else:
                if id == tmp_id:
                    continue
                else:
                    new_sentence_list.append(sent.strip())
            
    return new_sentence_list


def get_motion_indicator(test_sentence):
    motion_indicator = ""
    for each_word in motion_dict:
        if each_word in test_sentence:
            motion_indicator = each_word
            motion_doc = nlp(motion_indicator)
            break
    else:
        doc = nlp(test_sentence)
        try:
            for token_id, token in enumerate(doc):
                if token.tag_ == "VB" or token.tag_ == "VBP":
                    if doc[token_id+1].pos_ == "ADP":
                        motion_indicator = token.text + " " + doc[token_id+1].text
                        motion_doc = [token, doc[token_id+1]]
                        break
                    else:
                        motion_indicator = token.text
                        motion_doc = [token]
                        break
            else:
                motion_indicator = doc[0].text
                motion_doc = [doc[0]]
        except IndexError:
                motion_indicator = token.text
                motion_doc = [token]
   
    sum_list = []
    for motion_token in motion_doc:
        sum_list.append(motion_token.vector)
    motion_indicator_vector = np.mean(np.array(sum_list), axis=0)
        
    return (motion_indicator, motion_indicator_vector)

# def get_landmark(each_configuration):
#     landmark_stopwords = ['right', 'left','front','them', 'you','end','top', 'bottom','it','middle','side', 'a left', 'a right', 'steps', 'step', 'exit']
#     doc = nlp(each_configuration)
#     landmark_list =[]
#     landmark_text_list = []
#     landmark_flag = 1
#     if each_configuration in motion_dict:
#         landmark_flag = 0
#         landmark_list=[("", np.zeros(300))]
#         return [landmark_list, landmark_flag]

#     for chunk in doc.noun_chunks:
#         landmark_text = chunk.root
#         if landmark_text not in landmark_text_list and landmark_text.text not in landmark_stopwords:
#             landmark_vector = landmark_text.vector
#             landmark_list.append((landmark_text.text, landmark_vector))
#             landmark_text_list.append(landmark_text)
#     if not landmark_list:
#         landmark_flag = 0
#         landmark_list=[("", np.zeros(300))]
#     return [landmark_list, landmark_flag]

def get_landmark(each_configuration, whether_root=False):
    landmark_stopwords = ['right', 'left','front','them', 'you','end','top', 'bottom','it','middle','side', 'a left', 'a right', 'level', 'exit', "each", 'other', "far", 'your', 'area', 'flight']
    definite_article = ['a', 'an', 'the']
    area_stopwords = ['area','areas']

    doc = nlp(each_configuration)
    landmark_list =[]
    landmark_text_list = []
    landmark_flag = 1
    if each_configuration in motion_dict:
        landmark_flag = 0
        landmark_list=[["", np.zeros(300),-100, [0,0,0,0,0,0]]]
        return [landmark_list, landmark_flag]

    for chunk in doc.noun_chunks:
        if whether_root:
            for token_id, each_token in enumerate(chunk):
                if each_token.text in area_stopwords:
                    if chunk[0].text in definite_article:
                        chunk = chunk[1:]
                        chunk = chunk[token_id-2]
                    else:
                        chunk = chunk[token_id-1]
                    landmark_text = chunk
                    break
            else:
                landmark_text = chunk.root
            if landmark_text not in landmark_text_list:
                if landmark_text.text in landmark_stopwords:
                        continue
                else:
                    landmark_vector = landmark_text.vector
                    landmark_list.append([landmark_text.text, landmark_vector, landmark_text.idx, [0,0,0,0,0,0]])
                    landmark_text_list.append(landmark_text)

        else:
            if chunk[0].text in definite_article:
                landmark_text = chunk[1:]
            else:
                landmark_text = chunk
                          
    if not landmark_list:
        landmark_flag = 0
        landmark_list=[["", np.zeros(300),-100, [0,0,0,0,0,0]]]
    return [landmark_list, landmark_flag]

def get_glove_matrix(index_to_word, vector_dim):
    ''' load GloVE '''
    glove_path = '/egr/research-hlr/joslin/Matterdata/v1/scans/img_features/glove.6B'
    vectors = bcolz.open(f'{glove_path}/6B.{vector_dim}.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.{vector_dim}_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.{vector_dim}_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    ''' create the weight matrix '''
    weights_matrix = np.zeros((len(index_to_word), vector_dim))

    for i in range(len(index_to_word)):
        word = index_to_word[i]
        if word == 'others':
            weights_matrix[i] = np.zeros((vector_dim, ))
        else:
            try:
                weights_matrix[i] = glove[word]
            except KeyError:
                print("ERROR")

    return torch.from_numpy(weights_matrix).cuda()

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
  
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        old = self.vocab_size()
        self.add_word('<BOS>')
        assert self.vocab_size() == old+1
        print("OLD_VOCAB_SIZE", old)
        print("VOCAB_SIZE", self.vocab_size())
        print("VOACB", len(vocab))

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        for word in self.split_sentence(sentence):
            encoding.append(self.word_to_index[word])   # Default Dict
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            encoding[max_length - 1] = self.word_to_index['<EOS>']                  # Cut the length with EOS

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def read_img_features(feature_store):
    import csv
    import base64
    from tqdm import tqdm

    print("Start loading the image feature")
    start = time.time()

    if "detectfeat" in args.features:
        views = int(args.features[10:])
    else:
        views = 36

    args.views = views

    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
    features = {}
    with open(feature_store, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                                   dtype=np.float32).reshape((views, -1))   # Feature of long_id is (36, 2048)

    print("Finish Loading the image feature from %s in %0.4f seconds" % (feature_store, time.time() - start))
    return features

def read_candidates(candidates_store):
    import csv
    import base64
    from collections import defaultdict
    print("Start loading the candidate feature")

    start = time.time()

    TSV_FIELDNAMES = ['scanId', 'viewpointId', 'heading', 'elevation', 'next', 'pointId', 'idx', 'feature']
    candidates = defaultdict(lambda: list())
    items = 0
    with open(candidates_store, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=TSV_FIELDNAMES)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            candidates[long_id].append(
                {'heading': float(item['heading']),
                 'elevation': float(item['elevation']),
                 'scanId': item['scanId'],
                 'viewpointId': item['next'],
                 'pointId': int(item['pointId']),
                 'idx': int(item['idx']) + 1,   # Because a bug in the precompute code, here +1 is important
                 'feature': np.frombuffer(
                     base64.decodestring(item['feature'].encode('ascii')),
                     dtype=np.float32)
                    }
            )
            items += 1

    for long_id in candidates:
        assert (len(candidates[long_id])) != 0

    assert sum(len(candidate) for candidate in candidates.values()) == items

    # candidate = candidates[long_id]
    # print(candidate)
    print("Finish Loading the candidates from %s in %0.4f seconds" % (candidates_store, time.time() - start))
    candidates = dict(candidates)
    return candidates

def add_exploration(paths):
    explore = json.load(open("tasks/R2R/data/exploration.json", 'r'))
    inst2explore = {path['instr_id']: path['trajectory'] for path in explore}
    for path in paths:
        path['trajectory'] = inst2explore[path['instr_id']] + path['trajectory']
    return paths

def angle_feature(heading, elevation):
    import math
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (args.angle_feat_size // 4),
                    dtype=np.float32)

def new_simulator():
    import MatterSim
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.init()

    return sim

def get_point_angle_feature(baseViewId=0):
    sim = new_simulator()

    feature = np.empty((36, args.angle_feat_size), np.float32)
    angles = np.empty((36, 2), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
        elif ix % 12 == 0:
            sim.makeAction(0, 1.0, 1.0)
        else:
            sim.makeAction(0, 1.0, 0)

        state = sim.getState()
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        angles[ix, :] = np.array([heading, state.elevation])
        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature

def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]


def add_idx(inst):
    toks = Tokenizer.split_sentence(inst)
    return " ".join([str(idx)+tok for idx, tok in enumerate(toks)])

import signal
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

from collections import OrderedDict

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)


stop_word_list = [
    ",", ".", "and", "?", "!"
]


def stop_words_location(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    if len(sws) == 0 or sws[-1] != (len(toks)-1):     # Add the index of the last token
        sws.append(len(toks)-1)
    sws = [x for x, y in zip(sws[:-1], sws[1:]) if x+1 != y] + [sws[-1]]    # Filter the adjacent stop word
    sws_mask = np.ones(len(toks), np.int32)         # Create the mask
    sws_mask[sws] = 0
    return sws_mask if mask else sws

def get_segments(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    sws = [-1] + sws + [len(toks)]      # Add the <start> and <end> positions
    segments = [toks[sws[i]+1:sws[i+1]] for i in range(len(sws)-1)]       # Slice the segments from the tokens
    segments = list(filter(lambda x: len(x)>0, segments))     # remove the consecutive stop words
    return segments

def clever_pad_sequence(sequences, batch_first=True, padding_value=0):
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    max_len = max(seq.size()[0] for seq in sequences)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    if padding_value is not None:
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

import torch
def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

def average_length(path2inst):
    length = []

    for name in path2inst:
        datum = path2inst[name]
        length.append(len(datum))
    return sum(length) / len(length)

def tile_batch(tensor, multiplier):
    _, *s = tensor.size()
    tensor = tensor.unsqueeze(1).expand(-1, multiplier, *(-1,) * len(s)).contiguous().view(-1, *s)
    return tensor

def viewpoint_drop_mask(viewpoint, seed=None, drop_func=None):
    local_seed = hash(viewpoint) ^ seed
    torch.random.manual_seed(local_seed)
    drop_mask = drop_func(torch.ones(2048).cuda())
    return drop_mask


class FloydGraph:
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


def load_vocabulary(embeddings_file: str) -> WordEmbeddings:
    f = open(embeddings_file)
    word_indexer = WordIndexer()
    vectors = []

    word_indexer.add_and_get_index("PAD")
    word_indexer.add_and_get_index("UNK")

    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)

            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    return WordEmbeddings(word_indexer, np.array(vectors))


def glove_embedding(embedding_file):
    wordEmbedding = load_vocabulary(embedding_file)
    glove_indexer = wordEmbedding.word_indexer
    glove_vector = wordEmbedding.vectors
    return glove_indexer, glove_vector
