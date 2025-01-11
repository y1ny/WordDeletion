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
    argp.add_argument('lang')

    args = argp.parse_args()
    output_path = args.output_path
    n_subj = args.n_subj
    lang = args.lang
    createDir(output_path)
    if lang == 'chinese':
        demon_path = f"stimulus/demonstration_zh_toy.csv"
    else:
        demon_path = f"stimulus/demonstration_en_toy.csv"
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
    
    test_path = f"stimulus/meaningless_sentence.csv"
    test_file = pd.read_csv(test_path,
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                quotechar=None,)
    test_sentences = test_file[lang]

    demonstrations = list(zip(demonstrations, demon_labels))
    n_trial = 24
    subj_lst = []
    for n_idx in range(n_subj):        
        demon_lst = random.sample(demonstrations, 24)
        test_lst = random.sample(test_sentences, 24)
        subj = list(zip(demon_lst, test_lst))
        subj_lst.append(subj)
    
    for idx, lst in enumerate(subj_lst):
        with open(f"{output_path}/{idx}.csv", 'w') as f:
            f.write("demonstration\tdemon_label\ttest_sentence\n")
            for item in lst:
                f.write("\t".join([item[0][0], item[0][1], item[1]]) + '\n')