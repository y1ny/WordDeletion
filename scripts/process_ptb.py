import glob
import pandas as pd
import csv
import re
import copy
from itertools import permutations
from argparse import ArgumentParser
import sys 
sys.path.append("..") 
from utils import *
from itertools import combinations
def combine(temp_list, n):
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2
cons_label = ['ADJP','-ADV','ADVP','-BNF','CC','CD','-CLF','-CLR',
              'CONJP','-DIR','DT','-DTV','EX','-EXT','FRAG','FW',
              '-HLN','IN','INTJ','JJ','JJR','JJS','-LGS','-LOC',
              'LS','LST','MD','-MNR','NAC','NN','NNS','NNP','NNPS',
              '-NOM','NP','NX','PDT','POS','PP','-PRD','PRN','PRP',
              '-PRP','PRP$', 'PRP-S','PRT','-PUT','QP','RB','RBR',
              'RBS','RP','RRC','S','SBAR','SBARQ','-SBJ','SINV','SQ',
              'SYM','-TMP','TO','-TPC','-TTL','UCP','UH','VB','VBD',
              'VBG','VBN','VBP','VBZ','-VOC','VP','WDT','WHADJP','WHADVP',
              'WHNP','WHPP','WP','WP$', 'WP-S','WRB','X']

phrase_label = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ',
                'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN',
                'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP',
                'WHAVP', 'WHNP', 'WHPP', 'X','WHADVP']

sent_label = ['S','SBAR', 'SBARQ', 'SINV', 'SQ']

unwanted_punc = r"""#&()*+/<=>@[\]^_`{|}~"""
unfold_dict = {"'re": 'are', "'s": 'is'}
def extract_sentence(parse_string):
    sentence = []
    parse_string = parse_string.split(" ")
    parse_string = list(filter(lambda x: x and x.strip(), parse_string))
    for node in parse_string:
        if node[-1] == ')':
            if '*' in node:
                continue
            sentence.append(node.replace(')', ''))
    return sentence

def preprocessing(parsed_lst, frequency_rank):
    filtered_parsing = []
    for parse_string in parsed_lst:
        sentence = []
        if 'NONE' in parse_string:
            continue
        # extract sentences
        sentence = extract_sentence(parse_string)
        
        # filter
        if sentence[-1] not in ['.', ',']:
            continue
        del sentence[-1]
        text = ' '.join(sentence)
        if len(sentence) not in range(4, 16):
            continue
        if len(set(sentence) & set(unwanted_punc)) > 0:
            continue
        if len(re.findall(r"\d+", ' '.join(sentence))) != 0:
            continue
        s_token = re.findall(r"\W", ' '.join(sentence))
        s_token = list(filter(lambda x: x and x.strip(), s_token))
        if len(s_token) != 0:
            continue
        is_common = True
        for tok in sentence:
            if tok.isupper():
                is_common = False
                break
            if tok.lower() not in frequency_rank.keys():
                is_common = False
                break
            if frequency_rank[tok.lower()] > 110000:
                is_common = False
                break
        if not is_common:
            continue
        filtered_parsing.append([text, parse_string])
    return filtered_parsing


from itertools import combinations
def combine(temp_list, n):
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2

def permute(temp_list, n):
    temp_list2 = []
    for c in permutations(temp_list, n):
        temp_list2.append(c)
    return temp_list2
import numpy as np

def get_cons_feature(parsed_lst):
    cons_feature_dict = {}
    syntax_feature = []
    for (sentence, parse_string) in parsed_lst:
        parse_string = parse_string.replace("(", ' ( ')
        parse_string = parse_string.replace(")", ' ) ')
        parse_string = parse_string.split(" ")
        parse_string = list(filter(lambda x: x and x.strip(), parse_string))
        phrase_sent_level = []
        all_level = []
        for node in parse_string:
            if '-' in node and node.isupper():
                node = node.split('-')[0]
            if '=' in node and node.isupper():
                node = node.split('=')[0]
            if node in ['(', ')']:
                if node == ')' and phrase_sent_level[-1] == '(':
                    phrase_sent_level.pop(-1)
                    all_level.pop(-1)
                else:
                    phrase_sent_level.append(node)
                    all_level.append(node)
                continue
            if node not in phrase_label and node not in sent_label:
                if node in cons_label:
                    all_level.append(node)
                    continue
                else:
                    phrase_sent_level.append(node)
                    all_level.append(node)
            else:
                phrase_sent_level.append(node)
                all_level.append(node)

        # for constituency level
        n_pointer_all = 0
        depth_all = 0
        n_node_all = []
        parent_all = []
        parent_node_all = []
        depth_lst_all = []
        depth_node_all = []

        for _ in range(30):
            parent_all.append([])
            depth_lst_all.append([])
            parent_node_all.append([])
            depth_node_all.append([])
        current_parent_all = [-1]
        current_parent_node_all = ['ROOT']
        current_node_all = []
        phrase_dict_all = {}
        parent_node_dict_all = {}
        tok_id = 0
        for node in all_level[1:-1]:
            if node in ['.', ',']:
                continue
            if n_pointer_all >= 30:
                break
            if node == '(':
                n_pointer_all += 1
                current_parent_all.append(len(depth_lst_all[n_pointer_all-1]) - 1)
                current_parent_node_all.append(f"{depth_lst_all[n_pointer_all-1][-1]}-D{n_pointer_all-1}-N{len(depth_lst_all[n_pointer_all-1]) - 1}")
                
                continue
            if node == ')':
                
                n_pointer_all -= 1
                current_parent_all.pop(-1)
                current_parent_node_all.pop(-1)
                continue
            if node not in cons_label:
                for par in current_parent_node_all:
                    if par not in phrase_dict_all.keys():
                        phrase_dict_all[par] = [[tok_id, node]]
                    else:
                        phrase_dict_all[par].append([tok_id, node])
                if depth_node_all[n_pointer_all][-1] not in phrase_dict_all.keys():
                    phrase_dict_all[depth_node_all[n_pointer_all][-1]] = [[tok_id, node]]
                    tok_id += 1
                else:
                    phrase_dict_all[depth_node_all[n_pointer_all][-1]].append([tok_id, node])
                    tok_id += 1
                continue
            depth_lst_all[n_pointer_all].append(node)
            parent_all[n_pointer_all].append(current_parent_all[-1])
            parent_node_all[n_pointer_all].append(current_parent_node_all[-1])
            parent_node_dict_all[f"{node}-D{n_pointer_all}-N{len(depth_lst_all[n_pointer_all]) - 1}"] = copy.deepcopy(current_parent_node_all)
            depth_node_all[n_pointer_all].append(f"{node}-D{n_pointer_all}-N{len(depth_lst_all[n_pointer_all]) - 1}")
        depth_lst_all = list(filter(lambda x: x!=[], depth_lst_all))
        parent_node_all = list(filter(lambda x: x!=[], parent_node_all))
        depth_node_all = list(filter(lambda x: x!=[], depth_node_all))
        parent_all = list(filter(lambda x: x!=[], parent_all))
        depth = len(depth_lst_all)
        n_node = list(map(lambda x: len(x), depth_lst_all))
        feature = [depth, n_node, parent_all, depth_node_all, parent_node_all, phrase_dict_all, parent_node_dict_all]
        assert depth == len(depth_node_all)
        assert depth == len(parent_node_all)
        n = ['ROOT']
        for item in depth_node_all:
            n += item
        t = list(phrase_dict_all.keys())
        t = sorted(t)
        n = sorted(n)
        assert set(n) == set(t)
        syntax_feature.append([sentence, tmp, feature])
        node_dict = {}
        for node in feature[5].keys():
            if node == 'ROOT':
                continue
            node_dict[node] = feature[5][node]
        node_parent_dict = {}
        for node in feature[6].keys():
            if node == 'ROOT':
                continue
            node_parent_dict[node] = feature[6][node]
        node_depth = feature[3]
        node_lst = []
        for sub_lst in node_depth:
            node_lst.extend(sub_lst)
        
        node_child_dict = {}
        for node_name in node_lst:
            child_lst = []
            for key in node_lst:
                if node_name == node_parent_dict[key][-1]:
                    child_lst.append(key)
            node_child_dict[node_name] = child_lst
        
        node_height_dict = {}
        for node_name in node_child_dict.keys():
            height = 1
            child_lst = node_child_dict[node_name]
            while child_lst != []:
                height += 1
                tmp = []
                for child in child_lst:
                    tmp.extend(node_child_dict[child])
                child_lst = tmp

            node_height_dict[node_name] = height
        node_brother_dict = {}
        for node in node_dict.keys():
            if node == 'ROOT':
                continue
            parent = node_parent_dict[node][-1]
            if parent == 'ROOT':
                node_brother_dict[node] = [node]
                continue
            child_lst = node_child_dict[parent]
            node_brother_dict[node] = child_lst
        
        cons_feature_dict[sentence] = [node_lst, node_depth, node_dict, node_height_dict, 
                                    node_child_dict, node_parent_dict, node_brother_dict]

    return cons_feature_dict, syntax_feature

def ptb_constituent_analysis(result, cons_feature_dict, demon = None):
    phrase_sent = []
    mixed_sent = []
    word_sent = []
    fail_sent = []
    for idx, r in enumerate(result):
        if r[2] == 'fail to follow' or '##' in r[2]:
            if len(r) == 4:
                r = r[:3]
            if demon:
                r = r + [demon[idx]]
            fail_sent.append(r)
            continue
        cons_feature = cons_feature_dict[r[0]]
        orig = r[0].split(' ')
        orig = list(filter(lambda x: x and x.strip(), orig))
        orig = ' '.join(orig).lower().split(' ')
        n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
        if not ops:
            if demon:
                fail_sent.append([r[0], r[2], '## unknown error', demon[idx]])
            else:
                fail_sent.append([r[0], r[2], '## unknown error'])
            continue
        temp_dict = cons_feature[2]
        phrase_node = []
        word_node = []
        combs = []
        combs.extend(combine(cons_feature[0], 1))


        comb_op = ops
        comb_op = sorted(comb_op, key = lambda x: x[0])
        comb_op = list(map(lambda x: x[1].lower(), comb_op))
        for comb in combs:
            tmp_phrase = []
            tmp_word = []
            item_lst = []
            for item in comb:
                item_lst.extend(item.split(' '))
            comb_ordered = []

            for item in item_lst:
                comb_ordered.extend(temp_dict[item])

            comb_ordered = sorted(comb_ordered, key = lambda x: x[0])
            comb_ordered = list(map(lambda x: x[1].lower(), comb_ordered))
            
            if comb_ordered == comb_op:
                for c in comb:
                    if c.split('-')[0] in phrase_label or c.split('-')[0] in sent_label:
                        tmp_phrase.append(c)
                    else:
                        tmp_word.append(c)
                
                # when find the phrase, break
                if tmp_word == []:
                    phrase_node = tmp_phrase
                    word_node = tmp_word
                    break
                
                if tmp_phrase:
                    phrase_height = -1
                    for c in phrase_node:
                        if cons_feature[3][c] > phrase_height:
                            phrase_height = cons_feature[3][c]
                    for c in tmp_phrase:
                        if cons_feature[3][c] > phrase_height:
                            phrase_node = tmp_phrase
                            word_node = tmp_word
                else:
                    if phrase_node == [] and word_node == []:
                        phrase_node = tmp_phrase
                        word_node = tmp_word
        if ' '.join(comb_op).lower() not in r[0].lower():
            fail_sent.append([r[0], r[2], 'discrete span'])
            continue
        if demon == None:
            if word_node == [] and phrase_node != []:
                phrase_sent.append([r[0], r[2], phrase_node, word_node])
            else:
                if phrase_node != []:
                    word_sent.append([r[0], r[2], phrase_node, word_node])
                else:
                    if word_node == []:
                        for op in ops:
                            for cons in cons_feature[2].keys():
                                tmp = cons_feature[2][cons]
                                tmp = list(map(lambda x: [x[0], x[1].lower()], tmp))
                                if tmp == [op]:
                                    word_node.append(cons)
                    word_sent.append([r[0], r[2], phrase_node, word_node])
        else:
            if word_node == [] and phrase_node != []:
                phrase_sent.append([r[0], r[2], phrase_node, word_node, demon[idx]])
            else:
                if phrase_node != []:
                    word_sent.append([r[0], r[2], phrase_node, word_node, demon[idx]])
                else:
                    if word_node == []:
                        for op in ops:
                            for cons in cons_feature[2].keys():
                                tmp = cons_feature[2][cons]
                                tmp = list(map(lambda x: [x[0], x[1].lower()], tmp))
                                if tmp == [op]:
                                    word_node.append(cons)
                    word_sent.append([r[0], r[2], phrase_node, word_node, demon[idx]])
    return phrase_sent, mixed_sent, word_sent, fail_sent


def visit_from_node_to_leaf(node, dep_feature):
    child_lst = get_dep_child(node, dep_feature)
    if not child_lst:
        return [[node]]
    
    path_lst = []
    for child in child_lst:
        
        lst = visit_from_node_to_leaf(child, dep_feature)
        path_lst.extend(lst)
    combs = []
    for n in range(1, len(child_lst)+1):
        combs.extend(combine(path_lst, n))
    paths = []
    for c in combs:
        tmp = []
        for item in c:
            tmp.extend(item)
        tmp.append(node)
        tmp = list(set(tmp))
        tmp = sorted(tmp, key = lambda x: x[0])
        if tmp not in paths:
            paths.append(tmp)
    return paths

def get_dep_child(node, dep_feature):
    child_lst = []
    for key, head in dep_feature.items():
        if head[1] == 'punct':
            continue
        if head[0] == node:
            child_lst.append(key)
    return child_lst

def get_dep_head(node, dep_feature):
    if dep_feature[node][1] == 'root':
        return []
    head = dep_feature[node]
    head_lst = [head[0]]
    while True:
        head = dep_feature[head[0]]
        if head[1] == 'root':
            break
        head_lst.append(head[0])
    return head_lst

def get_dep_child_graph(node, dep_feature):
    child_graph_lst = []
    child_lst = get_dep_child(node, dep_feature)
    for child_node in child_lst:
        child_graph = [child_node]
        for key, head in dep_feature.items():
            if head[1] == 'punct':
                continue
            head_lst = get_dep_head(key, dep_feature)
            if child_node in head_lst:
                child_graph.append(key)
        child_graph = sorted(child_graph, key = lambda x: x[0])
        child_graph_lst.append([child_graph, dep_feature[child_node][1]])
    return child_graph_lst

def visit_from_node_to_node(node, dep_feature):
    child_lst = get_dep_child(node, dep_feature)
    if not child_lst:
        return [[node]]
    
    path_lst = [[node]]
    for child in child_lst:
        lst = visit_from_node_to_node(child, dep_feature)
        path_lst.extend(lst)
    combs = []
    for n in range(1, len(child_lst)+1):
        combs.extend(combine(path_lst, n))
    paths = []
    for c in combs:
        tmp = []
        for item in c:
            tmp.extend(item)
        tmp.append(node)
        tmp = list(set(tmp))
        tmp = sorted(tmp, key = lambda x: x[0])
        if tmp not in paths:
            paths.append(tmp)
    return paths

def get_all_graph_without_leaf(dep_feature):
    graph_lst = []

    for key, head in dep_feature.items():
        paths = visit_from_node_to_node(key, dep_feature)
        graph_lst.extend(paths)
    graph_lst = sorted(graph_lst, key = lambda x: len(x))
    return graph_lst

def get_all_graph(dep_feature):
    graph_lst = []
    for key, head in dep_feature.items():
        paths = visit_from_node_to_leaf(key, dep_feature)
        graph_lst.extend(paths)
    return graph_lst
def ptb_dependency_analysis(result, dep_feature_dict, demon = None):
    phrase_sent = []
    word_sent = []
    fail_sent = []
    for idx, r in enumerate(result):
        if r[2] == 'fail to follow' or '##' in r[2]:
            r = [r[0], r[2],[], []]
            if demon:
                r = r + [demon[idx]]
            fail_sent.append(r)
            continue
        dep_feature = dep_feature_dict[r[0]]
        orig = r[0].split(' ')
        orig = list(filter(lambda x: x and x.strip(), orig))
        orig = ' '.join(orig).lower().split(' ')
        n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
        
        graph_lst = get_all_graph_without_leaf(dep_feature)

        combs = []
        combs.extend(combine(graph_lst, 1))

        comb_op = ops
        comb_op = sorted(comb_op, key = lambda x: x[0])
        comb_op = list(map(lambda x: x[1].lower(), comb_op))
        deleted_label = []
        deleted_key = []
        is_phrase = False
        for comb in combs:
            item_lst = []

            for item in comb:
                item_lst.extend(item)
            comb_ordered  = sorted(item_lst, key = lambda x: x[0])

            comb_ordered = list(map(lambda x: x[1].lower(), comb_ordered))
            
            if comb_ordered == comb_op:

                is_phrase = True
                break

        if demon == None:
            if is_phrase:
                phrase_sent.append([r[0], r[2], deleted_key, deleted_label])
            else:
                word_sent.append([r[0], r[2], [], []])

        else:
            if is_phrase:
                phrase_sent.append([r[0], r[2], deleted_key, deleted_label, demon[idx]])
            else:
                word_sent.append([r[0], r[2], [], [], demon[idx]])
        
    return phrase_sent, word_sent, fail_sent


def ptb_dependency_analysis_keep(result, dep_feature_dict, demon = None):
    phrase_sent = []
    word_sent = []
    fail_sent = []
    for idx, r in enumerate(result):
        if r[2] == 'fail to follow' or '##' in r[2]:
            r = [r[0], r[2],[], []]
            if demon:
                r = r + [demon[idx]]
            fail_sent.append(r)
            continue
        dep_feature = dep_feature_dict[r[0]]
        orig = r[0].split(' ')
        orig = list(filter(lambda x: x and x.strip(), orig))
        orig = ' '.join(orig).lower().split(' ')
        pred = r[2].lower().split(" ")
        graph_lst = get_all_graph_without_leaf(dep_feature)

        combs = []
        combs.extend(combine(graph_lst, 1))

        comb_op = pred
        deleted_label = []
        deleted_key = []
        is_phrase = False
        for comb in combs:
            item_lst = []
            for item in comb:
                item_lst.extend(item)
            comb_ordered  = sorted(item_lst, key = lambda x: x[0])
            comb_ordered = list(map(lambda x: x[1].lower(), comb_ordered))
            if comb_ordered == comb_op:
                is_phrase = True
                break

        if demon == None:
            if is_phrase:
                phrase_sent.append([r[0], r[2], deleted_key, deleted_label])
            else:
                word_sent.append([r[0], r[2], [], []])
        else:
            if is_phrase:
                phrase_sent.append([r[0], r[2], deleted_key, deleted_label, demon[idx]])
            else:
                word_sent.append([r[0], r[2], [], [], demon[idx]])
        
    return phrase_sent, word_sent, fail_sent

def get_spans_from_sentence(sentence, new_string):
    sentence_words = sentence.lower().split()
    new_words = new_string.lower().split()
    assert set(new_words).issubset(sentence_words)
    spans = []
    i = 0  
    j = 0  
    while i < len(new_words) and j < len(sentence_words):
        # If words match, begin a span
        if new_words[i] == sentence_words[j]:
            start = j
            while i < len(new_words) and j < len(sentence_words) and new_words[i] == sentence_words[j]:
                i += 1
                j += 1
            spans.append(" ".join(sentence_words[start:j]))
        else:
            j += 1
    assert " ".join(spans) == " ".join(new_words)
    return spans
def n_gram_match(orig, pred):
    for n in range(len(pred), 0, -1):
        for i in range(0, len(orig)-n +1):
            if orig[i: i+n] == pred:
                return [i, i+n]
    return []
def ptb_constituent_analysis_keep(result, cons_feature_dict, demon = None):

    phrase_sent = []
    mixed_sent = []
    word_sent = []
    fail_sent = []
    for idx, r in enumerate(result):
        if r[2] == 'fail to follow' or '##' in r[2]:
            if len(r) == 4:
                r = r[:3]
            if demon:
                r = r + [demon[idx]]
            fail_sent.append(r)
            continue
        cons_feature = cons_feature_dict[r[0]]
        orig = r[0].split(' ')
        orig = list(filter(lambda x: x and x.strip(), orig))
        orig = ' '.join(orig).lower().split(' ')
        pred = r[2].lower().split(" ")
        idx_lst = n_gram_match(orig, pred)
        if not idx_lst:
            fail_sent.append(r)
            continue
        
        temp_dict = cons_feature[2]
        phrase_node = []
        word_node = []
        combs = []
        combs.extend(combine(cons_feature[0], 1))
        ops = [[idx, orig[idx]] for idx in range(idx_lst[0], idx_lst[1])]
        comb_op = ops
        comb_op = sorted(comb_op, key = lambda x: x[0])
        comb_op = list(map(lambda x: x[1].lower(), comb_op))
        for comb in combs:
            tmp_phrase = []
            tmp_word = []
            item_lst = []
            for item in comb:
                item_lst.extend(item.split(' '))
            comb_ordered = []

            for item in item_lst:
                comb_ordered.extend(temp_dict[item])

            comb_ordered = sorted(comb_ordered, key = lambda x: x[0])
            comb_ordered = list(map(lambda x: x[1].lower(), comb_ordered))
            
            if comb_ordered == comb_op:
                for c in comb:
                    if c.split('-')[0] in phrase_label or c.split('-')[0] in sent_label:
                        tmp_phrase.append(c)
                    else:
                        tmp_word.append(c)
                
                if tmp_word == []:
                    phrase_node = tmp_phrase
                    word_node = tmp_word
                    break
                
                if tmp_phrase:
                    phrase_height = -1
                    for c in phrase_node:
                        if cons_feature[3][c] > phrase_height:
                            phrase_height = cons_feature[3][c]
                    for c in tmp_phrase:
                        if cons_feature[3][c] > phrase_height:
                            phrase_node = tmp_phrase
                            word_node = tmp_word
                else:
                    if phrase_node == [] and word_node == []:
                        phrase_node = tmp_phrase
                        word_node = tmp_word
        if demon == None:
            if word_node == [] and phrase_node != []:
                phrase_sent.append([r[0], r[2], phrase_node, word_node])
            else:

                if phrase_node != []:
                    word_sent.append([r[0], r[2], phrase_node, word_node])
                else:
                    if word_node == []:
                        for op in ops:
                            for cons in cons_feature[2].keys():
                                tmp = cons_feature[2][cons]
                                tmp = list(map(lambda x: [x[0], x[1].lower()], tmp))
                                if tmp == [op]:
                                    word_node.append(cons)
                    word_sent.append([r[0], r[2], phrase_node, word_node])
        else:
            if word_node == [] and phrase_node != []:
                phrase_sent.append([r[0], r[2], phrase_node, word_node, demon[idx]])
            else:
                if phrase_node != []:
                    word_sent.append([r[0], r[2], phrase_node, word_node, demon[idx]])
                else:
                    if word_node == []:
                        for op in ops:
                            for cons in cons_feature[2].keys():
                                tmp = cons_feature[2][cons]
                                tmp = list(map(lambda x: [x[0], x[1].lower()], tmp))
                                if tmp == [op]:
                                    word_node.append(cons)
                    word_sent.append([r[0], r[2], phrase_node, word_node, demon[idx]])
    return phrase_sent, mixed_sent, word_sent, fail_sent

def clean_str(raw_str):
    raw_str = raw_str.split(' ')
    raw_str = list(filter(lambda x: x and x.strip(), raw_str))
    return ' '.join(raw_str)

import random

def random_delete_XP_n_shot(cons_feature_dict, n_shot=1):
    depth_lst = [[] for _ in range(15)]
    for sent in cons_feature_dict.keys():
        depth = len(cons_feature_dict[sent][1])
        depth_lst[depth-1].append(sent)
    
    sent_pool = []
    # only keep the sentence whose depth is within 3~9
    for depth in range(3, 9):
        sent_lst = depth_lst[depth]
        sent_pool.extend(sent_lst)
    
    choiced_item = []
    choiced_test = []
    n_iter = 0
    while n_iter < 2400:
        demon_lst = []
        target_node = ""
        target_parent = ""
        processed_demon = []
        demon_iter = 0
        while len(demon_lst) < n_shot:
            demon_iter += 1
            if demon_iter == 100000:
                break
            demon_sent = random.sample(sent_pool, 1)[0]
            if demon_sent in processed_demon:
                continue
            cons_feature = cons_feature_dict[demon_sent]
            cons_item = list(cons_feature[2].keys())
            # only keep phrase node, filter the word and sentence node
            cons_item = list(filter(lambda x: x.split('-')[0] in phrase_label, cons_item))
            demon_cons = random.sample(cons_item, 1)[0]

            # no root node
            if int(demon_cons.split('-')[1][-1]) == 0:
                continue
            demon_parent = cons_feature[-2][demon_cons][-1]
            # parent node can not be sentence node
            if demon_parent.split('-')[0] not in phrase_label:
                continue
            # parent node type should not equal to the node type
            if demon_parent.split('-')[0] == demon_cons.split('-')[0]:
                continue
            if cons_feature[2][demon_parent] == cons_feature[2][demon_cons]:
                continue

            delete_item = cons_feature[2][demon_cons]
            delete_item = [item[0] for item in delete_item]
            # must be a phrase
            if len(delete_item) < 2:
                continue
            if target_node:
                if demon_cons.split('-')[0] != target_node or demon_parent.split('-')[0] != target_parent:
                    continue
            else:
                target_node = demon_cons.split('-')[0]
                target_parent = demon_parent.split('-')[0]
            demon_label = []
            for idx, tok in enumerate(demon_sent.split(' ')):
                if idx in delete_item:
                    continue
                demon_label.append(tok)
            # a demonstration label
            demon_label = ' '.join(demon_label)
            demon_lst.append([demon_sent, demon_label, demon_cons, demon_parent])
            processed_demon.append(demon_sent)
        if demon_iter == 100000:
            continue
        candiate_sent = []
        for sent in sent_pool:
            # no same sentence
            if sent in processed_demon:
                continue
            sent_cons_feature = cons_feature_dict[sent]
            sent_cons_item = list(sent_cons_feature[2].keys())
            sent_cons_item = list(map(lambda x: x.split('-')[0], sent_cons_item))
            
            # test sentence should contain the delete node type
            if target_node in sent_cons_item:
                
                # set a label
                node_item = list(filter(lambda x: x.split('-')[0] == target_node, list(sent_cons_feature[2].keys())))
                # construct a fake label
                delete_item = sent_cons_feature[2][node_item[0]]
                delete_item = [item[0] for item in delete_item]
                label = []
                for idx, tok in enumerate(sent.split(' ')):
                    if idx in delete_item:
                        continue
                    label.append(tok)
                label = ' '.join(label)
                
                # for the matched node, the parent should contain the parent node of demonstrations
                is_parent = False
                for node in node_item:
                    word = [item[1] for item in sent_cons_feature[2][node]]
                    if len(word) == 1:
                        continue
                    parent_lst = sent_cons_feature[-2][node][:-1]
                    for parent in parent_lst:
                        if parent.split('-')[0] == target_parent:
                            is_parent = True
                            break
                    if is_parent:
                        break
                if is_parent:
                    candiate_sent.append([sent, label])
        
        if len(candiate_sent) < 10:
            continue
        else:
            test_sent = random.sample(candiate_sent, 1)[0]
            while all([(item[0], test_sent[0]) in choiced_item for item in demon_lst]):
                test_sent = random.sample(candiate_sent, 1)[0]
            for item in demon_lst:
                choiced_item.append((item[0], test_sent[0]))
                choiced_item.append((test_sent[0], item[0]))
            choiced_test.append([demon_lst, target_node, target_parent, test_sent[0], test_sent[1]])
            n_iter += 1
            print("iteration: ", n_iter)
    return choiced_test


def get_node_height(node, cons_feature):
    child_lst = cons_feature[4][node]
    child_lst = list(map(lambda x: x.split('-')[1], child_lst))
    return len(list(set(child_lst))) - 1 

def get_n_node(node, cons_feature):
    lst = [node]
    n_node = 0
    while lst:
        tmp = []
        for n in  lst:
            if len(cons_feature[-3][n]) != 0:
                
                tmp.extend(cons_feature[-3][n])
                n_node += 1
        lst = tmp
    return n_node


def get_node_distance_same_tree(node1, node2, cons_feature):
    node1_dep = int(node1.split('-')[1][-1])
    node2_dep = int(node2.split('-')[1][-1])
    if node1_dep < node2_dep:
        tmp = node1
        node1 = node2
        node2 = tmp
    
    distance = 0
    parent_lst = cons_feature[5][node1]
    for par in parent_lst[::-1]:
        distance += 1
        if par == node2:
            break

    return distance

def get_node_distance(node1, node2, cons_feature):
    path_1 = cons_feature[-2][node1] + [node1]
    path_2 = cons_feature[-2][node2] + [node2]

    diff_tree = False
    # search the biggest common node
    for idx, (v, t) in enumerate(zip(path_1, path_2)):
        if v != t:
            diff_tree = True
            break
    
    if diff_tree:
        v = path_1[idx-1]
        t = path_2[idx-1]
        v_distance = get_node_distance_same_tree(v, node1, cons_feature)
        t_distance = get_node_distance_same_tree(t, node2, cons_feature)
        tree_distance = v_distance + t_distance
    else:
        
        tree_distance = get_node_distance_same_tree(node1, node2, cons_feature)

    return tree_distance

def extract_ellipsis_item_from_single(r):
    item_lst =[]
    if r[2] == 'fail to follow':
        return
    orig = r[0].split(' ')
    orig = list(filter(lambda x: x and x.strip(), orig))
    orig = ' '.join(orig).lower().split(' ')
    n_op, ops = minDeletionops(orig, r[2].lower().split(' '))
    
    for idx, op in enumerate(ops):
        if idx == 0:
            cons = [op]
        
        else:
            if op[0] - ops[idx-1][0] != 1:
                item_lst.append(cons)
                cons = [op]
            else:
                cons.append(op)
        if idx == len(ops) - 1:
            item_lst.append(cons)
    return item_lst
from string import punctuation
import pyconll
def process_dependency(paths):
    dep_paths = glob.glob(paths)
    dep_feature_dict = {}
    for p in dep_paths:
        train = pyconll.load_from_file(p)
        for sentence in train:
            dep_feature = {}
            tok_lst = []
            for token in sentence:
                tok_lst.append(token.form)
            for token in sentence:
                key = (int(token.id)-1, token.form)
                head_key = (int(token.head)-1, tok_lst[int(token.head)-1])
                
                dep_feature[key] = [head_key, token.deprel, token.upos]
            if tok_lst[-1] in punctuation:
                tok_lst = tok_lst[:-1]
            dep_feature_dict[' '.join(tok_lst)] = dep_feature
    save_pkl("data/ptb_dep_feature.pkl", dep_feature_dict)
if __name__ == "__main__":
    
    argp = ArgumentParser()
    argp.add_argument('input_path')
    argp.add_argument('output_path')
    argp.add_argument('process', help='dependency/filter/vpnp/random')
    args = argp.parse_args()
    if args.process == 'dependency':
        process_dependency(args.input_path)
    elif args.process == 'filter':
        
        paths = glob.glob(args.input_path)
        parsed_lst = []
        frequency_file = pd.read_csv("ptb/unigram_freq.csv",
                            delimiter=',',
                            quoting=csv.QUOTE_NONE,
                            quotechar=None,)
        frequency_rank = {word:rank for rank, word in enumerate(frequency_file['word'])}
        
        for p in paths:
            parsed_file = pd.read_csv(p,
                        delimiter='\t',
                        quoting=csv.QUOTE_NONE,
                        quotechar=None,
                        header=None)
            tmp = parsed_file.values.tolist()
            tmp = list(map(lambda x: x[0], tmp))
            parsed_lst.extend(preprocessing(tmp, frequency_rank))
        
        cons_feature_dict, syntax_feature = get_cons_feature(parsed_lst)
        # save the sentence and corresponding structures
        save_pkl("data/ptb_cons_feature.pkl", cons_feature_dict)
    elif args.process == 'vpnp':
        cons_feature_dict = read_pkl("data/ptb_cons_feature.pkl")
        depth = []

        depth_lst = [[] for _ in range(6)]
        for sentence in cons_feature_dict.keys():
            depth = len(cons_feature_dict[sentence][1])
            if depth not in range(3, 9):
                continue
            depth_lst[depth - 3].append(sentence)
        delet_lst = []
        delet_test_lst = []
        for lst in depth_lst:
            delet_test_dict = {}
            delet_dict = {}
            for sent in lst.keys():
                cons_feature = cons_feature_dict[sent]
                current_parent = []
                ordered_node = []
                for node in cons_feature[2].keys():
                    ordered_node.append([node, cons_feature[2][node][0][0], cons_feature[3][node]])
                ordered_node = sorted(ordered_node, key = lambda x: x[2], reverse=True)
                ordered_node = sorted(ordered_node, key = lambda x: x[1])
                ordered_node = [item[0] for item in ordered_node]
                target_node = None

                NP_lst = []
                for node in ordered_node:
                    if node.split('-')[0] != 'NP':
                        continue
                    # only save the phrase, not the word
                    if len(cons_feature[2][node]) < 2:
                        continue
                    # only save the NP whose parents have VP
                    for par in cons_feature[5][node]:
                        if par.split('-')[0] == 'VP':
                            NP_lst.append(node)
                            break
                if not NP_lst:
                    print('No VP-NP: ', sent)
                    continue

                NP_info_lst = []
                for NP in NP_lst:
                    for par in cons_feature[5][NP][::-1]:
                        if par.split('-')[0] == 'VP':
                            break
                    linear_distance = cons_feature[2][NP][0][0]
                    tree_distance = get_node_distance(NP, par, cons_feature)
                    NP_info_lst.append([NP, linear_distance, tree_distance, cons_feature[3][NP]])

                # for test, we only need the VP node the NP child
                # the test node is not important
                NP_info_lst = sorted(NP_info_lst, key = lambda x: x[2], reverse=True)
                test_node = NP_info_lst[0][0]
                test_word = cons_feature[2][test_node]
                test_word = [item[1] for item in test_word]
                delet_test_dict[sent] = [test_node, test_word, 'None', 'None']
                
                # for demonstration, we keep the NP directly connect to the VP
                tmp = [item for item in NP_info_lst if item[2] == 1]
                tmp = sorted(tmp, key = lambda x: x[1])
                tmp = sorted(tmp, key = lambda x: x[2])

                if tmp == []:
                    continue
                # the nearest NP
                target_node = tmp[0][0]
                target_word = cons_feature[2][target_node]
                target_word = [item[1] for item in target_word]
                target_dep = -1

                if not target_dep:
                    print("error: ", sent)
                else:
                    if len(target_word) == 1:
                        print('one-word NP: ', sent )
                    else:
                        delet_dict[sent] = [target_node, target_word, 'None', 'None']
            delet_lst.append(delet_dict)
            delet_test_lst.append(delet_test_dict)
        for d_idx, delet_dict in enumerate(delet_lst):
            data = []
            for sent in delet_dict.keys():

                origin = sent
                origin = origin.split(' ')
                origin = list(filter(lambda x: x and x.strip(), origin))
                origin = ' '.join(origin)
                delete_item = ' '.join(delet_dict[sent][1])
                cons_info = delet_dict[sent][0]
                
                label = origin.replace(delete_item, '')
                label = label.split(' ')
                label = list(filter(lambda x: x and x.strip(), label))
                label = ' '.join(label)
                if delete_item == origin:
                    print("sentence is NP: ", origin)
                    continue
                if len(label.split(' ')) + len(delete_item.split(' ')) != len(origin.split(' ')):
                    print(sent)
                    continue
                
                data.append([origin, label, delete_item, cons_info])
            
            save_path = f'{args.output_path}/demonstration.csv'
            createDir(f'{args.output_path}')
            if not os.path.exists(save_path):
                with open(save_path, 'w') as f:
                    f.write("sentence\tlabel\tdelete\tcons\n")
            with open(save_path, 'w') as f:
                for item in data:
                    f.write("\t".join(item) + "\n")
        for d_idx, delet_test_dict in enumerate(delet_test_lst):
            data = []
            for sent in delet_test_dict.keys():
                origin = sent
                origin = origin.split(' ')
                origin = list(filter(lambda x: x and x.strip(), origin))
                origin = ' '.join(origin)
                delete_item = ' '.join(delet_test_dict[sent][1])
                cons_info = delet_test_dict[sent][0]
                
                label = origin.replace(delete_item, '')
                label = label.split(' ')
                label = list(filter(lambda x: x and x.strip(), label))
                label = ' '.join(label)
                if len(label.split(' ')) + len(delete_item.split(' ')) != len(origin.split(' ')):
                    print(sent)
                    continue
            

                data.append([origin, label, delete_item, cons_info])
            
            save_path = f'{args.output_path}/test_sentence.csv'
            createDir(f'{args.output_path}')
            if not os.path.exists(save_path):
                with open(save_path, 'w') as f:
                    f.write("sentence\tlabel\tdelete\tcons\n")
            with open(save_path, 'w') as f:
                for item in data:
                    f.write("\t".join(item) + "\n")
    elif args.process == 'random':
        cons_feature_dict = read_pkl("data/ptb_cons_feature.pkl")
        # n_shot can be set to N to peform N-shot learning
        random_test = random_delete_XP_n_shot(cons_feature_dict, n_shot=1)
        createDir(args.output_path)
        with open(f"{args.output_path}/english.csv", 'w') as f:
            f.write("demonstration\tdemon_label\tdemon_cons\tdemon_parent\ttest_sentence\n")
            for item in random_test:
                f.write("\t".join(item) + '\n')
        
                    