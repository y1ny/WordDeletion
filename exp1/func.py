import glob
import pandas as pd
import csv
import sys 
sys.path.append("..") 
from utils import *

def load_label_dict(label_type):
    path = f'stimulus/treebank/{label_type}/test_sentence'

    label_dict = {}
    label_file = pd.read_csv(path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
    for row_idx, row in label_file.iterrows():
        label_dict[row['sentence']] = [row['label'], row['delete'], row['cons']]
    return label_dict


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