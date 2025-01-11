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

if __name__ == "__main__":

    argp = ArgumentParser()
    argp.add_argument('test_path')
    argp.add_argument('output_path')

    args = argp.parse_args()
    output_path = args.output_path
    test_path = args.test_path
    output_path = f"{output_path}/"
    paths = glob.glob(f"{test_path}/*.csv")
    createDir(output_path)
    
    for idx, p in tqdm.tqdm(enumerate(paths)):
        test_file = pd.read_csv(p,
                        delimiter='\t',
                        quoting=csv.QUOTE_NONE,
                        quotechar=None,)

        resp_lst = []
        for idx, row in test_file.iterrows():
            d = [row['demonstration'], row['demon_label']]
            s = row['test_sentence']
            input_text = construct_prompt(d, s, prompt_id = 1)
            resp, resp_info = send_message(input_text) 
            result = extract_response(d, s, resp)
            result = result[-1]
            resp_lst.append([d, s, result, row['demon_cons'], row['demon_parent']])

        with open(f"{output_path}/{idx}.csv", 'w') as f:
            f.write("demonstration\tdemon_label\ttest_sentence\tresponse\tdemon_cons\tdemon_parent\n")
            for item in resp_lst:
                f.write("\t".join([item[0][0], item[0][1], item[1], item[2], item[3], item[4]]) + '\n')

            
