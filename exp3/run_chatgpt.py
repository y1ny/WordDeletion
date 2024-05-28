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
    argp.add_argument('output_path')
    args = argp.parse_args()
    sentence = args.sentence
    output_path = args.output_path
    createDir(output_path)
    if not is_chinese(sentence):
        lang = 'english'
    else:
        lang = 'chinese'
    
    # # if you obtain the treebank files, run the following code to construct the demonstration set
    # cons_feature_dict = read_pkl("data/ptb_cons_feature.pkl")
    # construct_retriever_set(sentence, output_path, cons_feature_dict)
    
    # if you do not obtain the treebank files, please use the demonstration set we provided
    demon_file = pd.read_csv(f"stimulus/demonstration/{lang}.csv",
                    delimiter='\t',
                    quoting=csv.QUOTE_NONE,
                    quotechar=None,)
    demonstrations = list(zip(demon_file['demonstration'].values.tolist(),
                              demon_file['demon_label'].values.tolist()))
    resp_lst = []
    for d in demonstrations:
        input_text = construct_prompt(d, sentence, prompt_id = 1)
        resp, resp_info = send_message(input_text) 
        resp_lst.append([d, sentence, resp])

    with open(f"{output_path}/response.csv", 'w') as f:
        f.write("demonstration\tdemon_label\ttest_sentence\tresponse\n")
        for item in resp_lst:
            f.write("\t".join([item[0][0], item[0][1], item[1], item[2]]) + '\n')

        
