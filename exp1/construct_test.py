from func import *
import random
import pandas as pd
import csv
from argparse import ArgumentParser
import sys 
sys.path.append("..") 
from utils import *

if __name__ == '__main__':

    argp = ArgumentParser()
    argp.add_argument('output_path')
    argp.add_argument('n_subj')
    argp.add_argument('type')
    argp.add_argument('lang')

    args = argp.parse_args()
    output_path = args.output_path
    n_subj = args.n_subj
    lang = args.lang
    exp_type = args.type
    createDir(output_path)
    if exp_type == 'parallel':
        demon_path = f"stimulus/{exp_type}/demonstration.csv"
        demon_file = pd.read_csv(demon_path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
        demonstrations = demon_file[lang]
        if lang == 'chinese':
            d_label = 'zh_label'
        else:
            d_label = 'en_label'
        demon_labels = demon_file[d_label]
        
        test_path = f"stimulus/{exp_type}/test_sentence.csv"
        test_file = pd.read_csv(test_path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
        test_sentences = test_file[lang]
    elif exp_type == 'treebank':
        demon_path = f"stimulus/{exp_type}/{lang}/demonstration.csv"
        demon_file = pd.read_csv(demon_path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
        demonstrations = demon_file['demonstration']
        demon_labels = demon_file['demon_label']

        test_path = f"stimulus/{exp_type}/{lang}/test_sentence.csv"
        test_file = pd.read_csv(test_path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
        test_sentences = test_file['sentence']
    demonstrations = list(zip(demonstrations, demon_labels))
    n_trial = 24
    subj_lst = []
    for n_idx in range(n_subj):        
        demon_lst = random.sample(demonstrations, 24)
        test_lst = random.sample(list(set(test_sentences) - set([item[0] for item in demonstrations])), 24)
        subj = list(zip(demon_lst, test_lst))
        subj_lst.append(subj)
    
    for idx, lst in enumerate(subj_lst):
        with open(f"{output_path}/{idx}.csv", 'w') as f:
            f.write("demonstration\tdemon_label\ttest_sentence\n")
            for item in lst:
                f.write("\t".join([item[0][0], item[0][1], item[1]]) + '\n')