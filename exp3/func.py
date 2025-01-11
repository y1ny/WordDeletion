import glob
import pandas as pd
import csv
import sys 
sys.path.append("..") 
from utils import *
import random
from collections import defaultdict
def load_label_dict(label_type):

    path = f'stimulus/{label_type}.csv'
    label_dict = {}
    label_file = pd.read_csv(path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
    for row_idx, row in label_file.iterrows():
        label_dict[row['demonstration']] = [row['demon_label'], row['demon_cons'], row['demon_parent']]
    return label_dict

def random_deletion(test_sentence, N=10000):
    lst = []
    if is_chinese(test_sentence[0]):
        is_ctb = True
    else:
        is_ctb = False
    for _ in range(N):
        random_string = []
        for sent in test_sentence:
            if is_ctb:
                tok_lst = list(sent)
            else:
                tok_lst = sent.split(' ')
            processed = []
            indices = random.sample(list(range(len(tok_lst) + 1)), 2)
            while abs(indices[1] - indices[0]) == len(tok_lst):
                 indices = random.sample(list(range(len(tok_lst) + 1)), 2)
            if indices[0] > indices[1]:
                indices[1], indices[0] = indices[0], indices[1]
            for i, tok in enumerate(tok_lst):
                if i not in range(indices[0], indices[1]):
                    processed.append(tok)
            if is_ctb:
                processed = ''.join(processed)
            else:
                processed = ' '.join(processed)
            random_string.append([sent, sent, processed])
        lst.append(random_string)
    return lst


def convert_to_char_dict(node_dict):
    word_seq = node_dict['S-D0-N0']
    char_dict = {}
    for node in node_dict:
        tok_lst = node_dict[node]
        char_lst = []
        for tok in tok_lst:
            tmp = []
            index = tok[0]
            previous_word = word_seq[:index]
            previous_word = [item[1] for item in previous_word]
            previous_word = ''.join(previous_word)
            char_index = len(previous_word)
            word = tok[1]
            for t_idx, t in enumerate(word):
                tmp.append([t_idx + char_index, t])
            char_lst.extend(tmp)
        char_dict[node] = char_lst
    return char_dict

def get_linear_feautre(result, label_dict, cons_feature_dict, is_ctb):
    lst = []
    for r in result:
        if '##' in r[2] or 'fail to follow' in r[2]:
            lst.append([r[0], r[2], 'other'])
            continue
        
        label_info = label_dict[r[-1]]
        cons_feature = cons_feature_dict[r[-1]]
        if is_ctb:
            char_dict = convert_to_char_dict(cons_feature[2])
            demon_deleted = char_dict[label_info[2]]
        else:
            demon_deleted = cons_feature[2][label_info[2]]
        demon_remained = label_info[0].lower()
        test_remained = r[2]
        if is_ctb:
            orig = list(r[0].strip())
            n_op, ops = minDeletionops(orig, list(r[2].strip()))
            demon_remained = list(demon_remained)
            test_remained = list(test_remained)
        else:
            orig = r[0].split(' ')
            orig = list(filter(lambda x: x and x.strip(), orig))
            orig = ' '.join(orig).lower().split(' ')
            n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
            demon_remained = demon_remained.split(' ')
            test_remained = test_remained.split(' ')
        if not ops:
            lst.append([r[0], r[2], 
                    None,
                    None,
                    None])  
            continue   
        test_deleted = ops
        # print(test_deleted, r)
        lst.append([r[0], r[2], 
                orig.index(test_remained[0]),
                test_deleted[0][0],
                len(test_remained),
                len(test_deleted),])
    return lst


def is_connected(data, nodes):
    visited = set()
    nodes = [(n[0], n[1].lower()) for n in nodes]  

    data_lower = {
        (k[0], k[1].lower()): [(v[0][0], v[0][1].lower() if isinstance(v[0][1], str) else v[0][1]), v[1]]
        for k, v in data.items()
    }
    
    def dfs(node):
        visited.add(node)
        if node in data_lower:
            target_node = data_lower[node][0]  
            if target_node != -1 and target_node not in visited and target_node in nodes:
                dfs(target_node)
        for key, value in data_lower.items():
            if value[0] == node and key not in visited and key in nodes:
                dfs(key)
    dfs(nodes[0])  
    return visited == set(nodes)

def largest_connected_subset(data, nodes):
    visited = set()
    largest_subset = set()
    nodes = [(n[0], n[1].lower()) for n in nodes]  
    
    data_lower = {
        (k[0], k[1].lower()): [(v[0][0], v[0][1].lower() if isinstance(v[0][1], str) else v[0][1]), v[1]]
        for k, v in data.items()
    }
    
    def dfs(node, current_set):
        visited.add(node)
        current_set.add(node)
        if node in data_lower:
            target_node = data_lower[node][0]
            if target_node != -1 and target_node not in visited and target_node in nodes:
                dfs(target_node, current_set)
        for key, value in data_lower.items():
            if value[0] == node and key not in visited and key in nodes:
                dfs(key, current_set)
    
    for node in nodes:
        if node not in visited:
            current_set = set()
            dfs(node, current_set)
            if len(current_set) > len(largest_subset):
                largest_subset = current_set
    return largest_subset

def get_mean_dependency_distance(lst, dep_feature):
    distance = []
    length = len(dep_feature.keys()) - 1
    lst = [tuple(key) for key in lst]
    if len(lst) == 1:
        return length
    if not lst:
        return -1
    if is_connected(dep_feature, lst):
        connected = lst
        independent = []
    else:
        largest_subset = largest_connected_subset(dep_feature, lst)
        connected = list(largest_subset)
        independent = list(set(lst).difference(set(connected)))
    for key in lst:
        if key not in dep_feature:
            keys = list(dep_feature.keys())
            keys = sorted(keys, key = lambda x: x[0])
            key_tmp = keys[key[0]]
            if key_tmp[1].lower() != key[1]:
                if set(key[1]).issubset(set(key_tmp[1].lower())):
                    distance.append(length)
                else:
                    print('error', lst, dep_feature, key, key_tmp)
                continue
            else:
                key = key_tmp
        
        head = dep_feature[key]
        if (key[0], key[1].lower()) in independent:
            distance.append(length)
        else:
            if head[1] == 'root':
                distance.append(0)
            elif (head[0][0], head[0][1].lower()) not in connected:
                distance.append(0)
            else:
                distance.append(abs(key[0] - head[0][0]))
    return np.sum(distance) / (len(lst)-1)

def merge_char_to_word(char_seq, word_seq):
    word_c_index = []
    current_pointer = 0
    for w_idx, w in enumerate(word_seq):
        word_c_index.append((current_pointer, w[1]))
        current_pointer += len(w[1])
    merge_lst = []

    for c in char_seq:
        c_index = c[0]
        for w_idx, w in enumerate(word_c_index):
            if w_idx == len(word_c_index) -1 :
                r = list(range(w[0], w[0] + len(w[1])))
            else:
                r = list(range(w[0], word_c_index[w_idx+1][0]))
            if c_index in r:
                merge_lst.append((w_idx, c[1]))
    merge_lst = sorted(merge_lst, key = lambda x: x[0])

    result = defaultdict(list)

    for number, char in merge_lst:
        result[number].append(char)

    merged_list = [(key, ''.join(value)) for key, value in result.items()]

    return merged_list

def get_mdd_lst(result, dep_feature_dict, is_ctb):

    mdd_lst = []
    for r in result:
        if '##' in r[2]  or 'fail to follow' in r[2]:
            mdd_lst.append([r[0], r[2], -1, -1])
            continue
        if is_ctb:
            orig = list(r[0].strip())
            n_op, ops = minDeletionops(orig, list(r[2].strip()))
        else:
            orig = r[0].split(' ')
            orig = list(filter(lambda x: x and x.strip(), orig))
            orig = ' '.join(orig).lower().split(' ')
            n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
        deleted_idx = [item[0] for item in ops]
        deleted = []
        remained = []
        for idx, tok in enumerate(orig):
            if idx in deleted_idx:
                deleted.append([idx, tok])
            else:
                remained.append([idx, tok])
        dep_feature = dep_feature_dict[r[0]]
        if is_ctb:
            tok_lst = list(dep_feature.keys())
            tok_lst = sorted(tok_lst, key = lambda x: x[0])
            deleted = merge_char_to_word(deleted, tok_lst)
            remained = merge_char_to_word(remained, tok_lst)
        mdd_d = get_mean_dependency_distance(deleted, dep_feature)
        mdd_r = get_mean_dependency_distance(remained, dep_feature)
        mdd_lst.append([r[0], r[2], mdd_d, mdd_r])
    return mdd_lst

from scripts.process_ptb import ptb_dependency_analysis, ptb_constituent_analysis
from scripts.process_ctb import ctb_dependency_analysis, ctb_constituent_analysis
def get_dependency_feature(result, dep_feature_dict, is_ctb, ):

    lst = []

    if is_ctb:
        phrase_sent, word_sent, fail_sent = ctb_dependency_analysis(result, dep_feature_dict)
        other_sent = fail_sent
    else:
        phrase_sent, word_sent, fail_sent = ptb_dependency_analysis(result, dep_feature_dict)
        # c_phrase_sent, _, c_word_sent, _ = ptb_constituent_analysis(result, cons_feature_dict)
        other_sent = fail_sent
    phrase_sent = [(item[0], item[1].lower()) for item in phrase_sent]
    word_sent = [(item[0], item[1].lower()) for item in word_sent]
    other_sent = [(item[0], item[1].lower()) for item in other_sent]
    lst = []
    for r in result:
        if (r[0], r[2].lower()) in phrase_sent:
            lst.append([r[0], r[2], 'connected'])
        elif (r[0], r[2].lower()) in word_sent:
            lst.append([r[0], r[2], 'disconnected'])
        else:
            lst.append([r[0], r[2], 'other'])
    mdd_lst = []
    for r in result:
        if '##' in r[2]  or 'fail to follow' in r[2]:
            mdd_lst.append([r[0], r[2], -1, -1])
            continue
        if is_ctb:
            orig = list(r[0].strip())
            n_op, ops = minDeletionops(orig, list(r[2].strip()))
        else:
            orig = r[0].split(' ')
            orig = list(filter(lambda x: x and x.strip(), orig))
            orig = ' '.join(orig).lower().split(' ')
            n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
        deleted_idx = [item[0] for item in ops]
        deleted = []
        remained = []
        for idx, tok in enumerate(orig):
            if idx in deleted_idx:
                deleted.append([idx, tok])
            else:
                remained.append([idx, tok])
        dep_feature = dep_feature_dict[r[0]]
        if is_ctb:
            tok_lst = list(dep_feature.keys())
            tok_lst = sorted(tok_lst, key = lambda x: x[0])
            # print(deleted)
            deleted = merge_char_to_word(deleted, tok_lst)
            # print(deleted)
            remained = merge_char_to_word(remained, tok_lst)

        mdd_d = get_mean_dependency_distance(deleted, dep_feature)
        mdd_r = get_mean_dependency_distance(remained, dep_feature)
        mdd_lst.append([r[0], r[2], mdd_d, mdd_r])
    return lst, mdd_lst

def get_constituent_feature(result, cons_feature_dict, is_ctb):

    if is_ctb:
        phrase_sent, word_sent, char_sent, fail_sent = ctb_constituent_analysis(result, cons_feature_dict)
        other_sent = char_sent + fail_sent
    else:
        phrase_sent, mixed_sent, word_sent, fail_sent = ptb_constituent_analysis(result, cons_feature_dict)
        other_sent = fail_sent
    phrase_sent = [(item[0], item[1].lower()) for item in phrase_sent]
    word_sent = [(item[0], item[1].lower()) for item in word_sent]
    other_sent = [(item[0], item[-1].lower()) for item in other_sent]
    lst = []
    for r in result:
        if (r[0], r[2].lower()) in phrase_sent:
            lst.append([r[0], r[2], 'constituent'])
        elif (r[0], r[2].lower()) in word_sent:
            lst.append([r[0], r[2], 'nonconstituent'])
        else:
            lst.append([r[0], r[2], 'other'])
    return lst

def is_xp(cons_feature, parent_type='VP', node_type='NP'):
    parent_dict = cons_feature[5]
    for node in parent_dict.keys():
        if node.split('-')[0] != node_type:
            continue
        for idx, parent in enumerate(parent_dict[node][::-1]):
            if parent.split('-')[0] == parent_type and idx != 0 :
                return True
    return False

def get_label_cons(label_tok, cons_feature):

    if is_chinese(label_tok):
        is_zh = True
    else:
        is_zh = False
    label_cons = None
    for node_name in cons_feature[0]:
        tok_lst = cons_feature[2][node_name]
        tok_lst = [x[1] for x in tok_lst]
        if is_zh:
            if ''.join(tok_lst) == label_tok:
                label_cons = node_name
                break
        else:
            if ' '.join(tok_lst).lower() == label_tok.lower():
                label_cons = node_name
                break
    return label_cons