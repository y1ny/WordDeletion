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
    
    data_path = f"stimulus/{lang}.csv"
    data_file = pd.read_csv(data_path,
                delimiter='\t',
                quoting=csv.QUOTE_NONE,
                quotechar=None,)
    demonstrations = data_file['demonstration']
    demon_labels = data_file['demon_label']
    demon_cons = data_file['demon_cons']
    demon_cons = list(map(lambda x: x.split('-')[0], demon_cons))
    demon_parent = data_file['demon_parent']
    demon_parent = list(map(lambda x: x.split('-')[0], demon_parent))
    test_sentences = data_file['test_sentence']
    
    pool = list(zip(demonstrations, demon_labels, test_sentences, demon_cons, demon_parent))
    n_trial = 24
    subj_lst = []
    for n_idx in range(n_subj):
        # within 24 tests, no replicated sentences
        processed_subj = []
        subj = []
        for t_idx in range(n_trial):
            while True:
                choiced = random.sample(pool, 1)[0]
                if choiced[0] in processed_subj or choiced[2] in processed_subj:
                    continue
                break
            processed_subj.append(choiced[0])
            processed_subj.append(choiced[2])
            subj.append(choiced)
        subj_lst.append(subj)
    
    for idx, lst in enumerate(subj_lst):
        with open(f"{output_path}/{idx}.csv", 'w') as f:
            f.write("demonstration\tdemon_label\ttest_sentence\tdemon_cons\tdemon_parent\n")
            for item in lst:
                f.write("\t".join(item) + '\n')