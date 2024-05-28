from func import *
import random
import pandas as pd
import tqdm
import csv
from argparse import ArgumentParser
import glob
import sys 
sys.path.append("..") 
from utils import *

if __name__ == '__main__':

    argp = ArgumentParser()
    argp.add_argument('output_path')
    argp.add_argument('n_subj')

    args = argp.parse_args()
    output_path = args.output_path
    n_subj = args.n_subj
    createDir(output_path)
    
    data_lst = []
    for attachment in ['adjunct', 'pp']:
        demon_path = f"stimulus/{attachment}/demonstration.csv"
        test_path = f"stimulus/{attachment}/test_sentence.csv"
        demon_file = pd.read_csv(demon_path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
        demon_lst = list(zip(demon_file['sentence'].values.tolist(), demon_file['label'].values.tolist()))
        structure_2 = []
        structure_1 = []
        test_file = pd.read_csv(test_path,
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
        
        for idx, row in test_file.iterrows():
            if attachment == 'adjunct':
                if idx % 2 == 0:
                    structure_2.append(row['target'])
                else:
                    structure_1.append(row['target'])
            else:
                if idx in range(9) or idx in range(18,27):
                    structure_2.append(row['target'])
                else:
                    structure_1.append(row['target'])
        data_lst.append([demon_lst, structure_1, structure_2])
    
    
    n_sample = 24
    
    subj_lst = []
    for n_iter in range(int(n_subj/2)):
        adjunct = data_lst[0]
        pp = data_lst[1]
        
        adjunct_test = random.sample(list(zip(adjunct[1], adjunct[2])), int(n_sample/2))
        pp_test = random.sample(list(zip(pp[1], pp[2])), int(n_sample/2))
        
        subj_1 = []
        subj_2 = []
        
        adjunct_demon = random.sample(adjunct[0], n_sample)
        random.shuffle(adjunct_demon)
        
        pp_demon = random.sample(pp[0], n_sample)
        random.shuffle(pp_demon)
        for idx, test_pair in enumerate(adjunct_test):
            demon_1 = adjunct_demon[idx]
            demon_2 = adjunct_demon[idx+12]
            
            # avoid the replicated words in demonstrations and test sentences.
            if set(demon_1[0].split(' ')[:5]) == set(test_pair[0].split(' ')[:5]):
                demon_1, demon_2 = demon_2, demon_1
            elif set(demon_1[0].split(' ')[:5]) == set(test_pair[0].split(' ')[:5]):
                demon_1, demon_2 = demon_2, demon_1
            if idx < n_sample / 4:
                subj_1.append([demon_1, test_pair[0], 'structure_1', 'adjunct'])
                
                subj_2.append([demon_2, test_pair[1], 'structure_2', 'adjunct'])
            else:
                subj_2.append([demon_1, test_pair[0], 'structure_1', 'adjunct'])
                
                subj_1.append([demon_2, test_pair[1], 'structure_2', 'adjunct'])
        
        for idx, test_pair in enumerate(pp_test):
            demon_1 = pp_demon[idx]
            demon_2 = pp_demon[idx+12]
            if set(demon_1[0].split(' ')[:3]) == set(test_pair[0].split(' ')[:3]):
                demon_1, demon_2 = demon_2, demon_1
            elif set(demon_1[0].split(' ')[:3]) == set(test_pair[0].split(' ')[:3]):
                demon_1, demon_2 = demon_2, demon_1
            if idx < n_sample / 4:
                subj_1.append([demon_1, test_pair[0], 'structure_1', 'pp'])
                
                subj_2.append([demon_2, test_pair[1], 'structure_2', 'pp'])
            else:
                subj_2.append([demon_1, test_pair[0], 'structure_1', 'pp'])
                
                subj_1.append([demon_2, test_pair[1], 'structure_2', 'pp'])
                
        random.shuffle(subj_1)
        random.shuffle(subj_2)
        subj_lst.append(subj_1)
        subj_lst.append(subj_2)
    
    for idx, subj in enumerate(subj_lst):
        with open(f"{output_path}/{idx}.csv", 'w') as f:
            f.write("demonstration\tdemon_label\ttest_sentence\tattachment\tstructure\n")
            for item in subj:
                f.write("\t".join(item[0]) + '\t')
                f.write("\t".join(item[1:]) + '\n')