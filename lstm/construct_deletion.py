import json
import pandas as pd
import csv
import glob
import random
import argparse
parser = argparse.ArgumentParser(description="construction deletion dataset")
parser.add_argument('--n_sample', '-n', required=False, default=800, help='training sample size')
parser.add_argument('--train_idx', '-i', required=False, default=0, help='training index')
parser.add_argument('--output_dir', '-o', required=False, default='data', help='output path')

args = parser.parse_args()
n_train = int(args.n_sample)
# print(n_train)

paths = glob.glob('samples/*.csv')
deletion_data = []
context_lst = []
deleted_span = []

for p in paths:
    data_file = pd.read_csv(p,
                        delimiter='\t',
                        quoting=csv.QUOTE_NONE,
                        quotechar=None,)
    context_lst.extend(data_file['sentence'].values.tolist())
    tmp = list(zip(data_file['delete'].values.tolist(), data_file['delete_start'].values.tolist()))
    deleted_span.extend(tmp)

data = []
for idx, (c, info) in enumerate(zip(context_lst, deleted_span)):
    d = {'title':'deletion', 'paragraphs': []}
    tmp = {'context':c, 'qas':[]}
    c_lst = c.split(' ')
    pre_ans = c_lst[:info[1]]
    pre_ans = ' '.join(pre_ans)
    start = len(pre_ans) + 1
    tmp['qas'].append({'answers':[{'answer_start':start, 'text':info[0]}], 
                    'question':c,
                    'id': idx})
    d['paragraphs'].append(tmp)
    data.append(d)
random.shuffle(data)

train_dict = {'data':data[:n_train], 'version':'v1'}
dev_dict = {'data':data[n_train:], 'version':'v1'}

with open(f'{args.output_dir}/Deletion/train.json', 'w') as f:
    json.dump(train_dict, f) 

train_lst = []
for item in train_dict['data']:
    demon = item['paragraphs'][0]['context']
    demon_label = item['paragraphs'][0]['qas'][0]['answers'][0]['text']
    demon_idx = str(item['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'])
    train_lst.append([demon, demon_label, demon_idx])

with open(f'{args.output_dir}/Deletion/wp/train_{args.n_sample}_{args.train_idx}.csv', 'w') as f:
    f.write("demonstration\tdeleted\tdeleted_idx\n")
    for item in train_lst:
        f.write("\t".join(item) + '\n')
with open(f'{args.output_dir}/Deletion/dev.json', 'w') as f:
    json.dump(dev_dict, f) 



