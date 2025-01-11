from func import *
import pandas as pd
import csv
from argparse import ArgumentParser
import sys 
sys.path.append("..") 
from utils import *

if __name__ == "__main__":

    argp = ArgumentParser()
    argp.add_argument('sentence')
    argp.add_argument('path')
    args = argp.parse_args()
    sentence = args.sentence
    path = args.path
    if not is_chinese(sentence):
        lang = 'en'
    else:
        lang = 'zh'
    

    resp_file = pd.read_csv(f"{path}/response.csv",
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
    result_lst = []
    for idx, resp in resp_file.iterrows():
        res = extract_response([resp['demonstration'], resp['demon_label']], 
                               resp['test_sentence'], resp['response'])
        result_lst.append(res)
    
    with open(f"{args.path}/result.csv", 'w') as f:
        f.write("demonstration\tdemon_label\ttest_sentence\tresult\n")
        for res in result_lst:
            f.write("\t".join(res) + '\n')
            
    result = [[item[2], item[2], item[-1]] for item in result_lst]
    item_lst, idx_dict = extract_ellipsis_item(result)
    cky_tree = cky(sentence, idx_dict)
    display_tree(cky_tree, sentence)
    
        
