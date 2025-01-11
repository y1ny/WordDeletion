import pickle
import os
import pandas as pd
import csv
import re
import numpy as np
import copy
import openai
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

suffix_lst = ["John developed a very special way of speaking. ",
                "A man speaks in a really weird way after a stroke. ",
                "An alien is learning our language, but he tends to omit some words in his sentences. ",
                "A young child is facing difficulties in language acquisition, often forgetting to say certain words. ",
                "A scientist has created an AI program that interprets sentences uniquely. ",
                "In a parallel universe, the language construct is slightly different. ",
                "A robot has been programmed to communicate in a unique way. "]

postfix_lst = ["Please carefully examine John's speaking style, and guess what he would say for the sentence: ",
                "What will he say for the sentence: ",
                'Please guess how the alien would express the following sentence: ',
                'Can you anticipate what he would say for the sentence: ',
                'Predict the output for the following input: ',
                'How would the sentence ',
                'Can you predict how the robot would express the sentence: ']

instruction_lst = ['?',
                    '?',
                    '?',
                    '?',
                    '.',
                    ' be conveyed in this universe?',
                    '?']

example_lst = [('For ', ', he would say '),
                ('For ', ', he would say '),
                ('For ', ', he would say '),
                ('When he tries to say ', ', he ends up saying '),
                ('When given the sentence ', ', it outputs '),
                ('The sentence ', ' would be communicated as '),
                ('When it tries to say ', ', it outputs ')]

def paired_bootstrap_test(a, b, num_bootstrap=10000):
    '''
    paired bootstrap 检验. 
    '''
    a = np.array(a)
    b = np.array(b)
    diff = a - b
    permuted_lst = []
    for i in range(num_bootstrap):
        permuted = np.random.choice(diff, size=len(diff), replace=True)
        permuted_lst.append(np.mean(permuted))

    A = min(np.sum(np.array(permuted_lst) > 0), np.sum(np.array(permuted_lst) < [0]))
    p_value = (2*(A + 1)) / (num_bootstrap+1)
    
    return p_value

def unpaired_bootstrap_test(a, b, num_bootstrap=10000):
    '''
    unpaired bootstrap 检验. 
    '''
    a = np.array(a)
    b = np.array(b)
    permuted_lst = []
    for i in range(num_bootstrap):
        permuted_a = np.random.choice(a, size = len(a), replace=True)
        permuted_b = np.random.choice(b, size = len(b), replace=True)
        permuted_diff = np.mean(permuted_a) - np.mean(permuted_b)
        permuted_lst.append(permuted_diff)
        
    A = min(np.sum(np.array(permuted_lst) > 0), np.sum(np.array(permuted_lst) < [0]))
    p_value = (2*(A + 1)) / (num_bootstrap+1)
    return p_value

def is_chinese(word):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    if pattern.search(word):
        return False
    return True

def save_pkl(path,data):
    with open(path,'wb') as f:
        pickle.dump(data,f)

def read_pkl(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def createDir(filePath):
    if os.path.exists(filePath):
        return
    else:
        try:
            os.mkdir(filePath)
        except Exception as e:
            os.makedirs(filePath)

def minDeletionops(A, B):
    # A: original sequence
    # B: prediction sequence
    res = 0
    opera_lst = []
    a_idx = 0
    b_idx = 0
    
    while a_idx < len(A):
        if b_idx >= len(B):
            opera_lst.append([a_idx, A[a_idx]])
            a_idx += 1
            res += 1
            continue
        if A[a_idx] == B[b_idx]:
            a_idx += 1
            b_idx += 1
        else:
            opera_lst.append([a_idx, A[a_idx]])
            a_idx += 1
            res += 1
    # this part we check whether the ops are correct
    tmp_B = copy.deepcopy(B)
    for op in opera_lst:
        tmp_B.insert(op[0], op[1])
    if tmp_B != A:
        return -1, []
    return res, opera_lst

def construct_prompt(demonstration, test_sentence, prompt_id):
    suffix_sent = suffix_lst[prompt_id]
    input_text = ""
    
    input_text += f"{example_lst[prompt_id][0]}'{demonstration[0]}'" + f"{example_lst[prompt_id][1]}'{demonstration[1]}'.\n "
    input_text += postfix_lst[prompt_id]
    suffix_sent += input_text
    postfix_sent = instruction_lst[prompt_id]

    input_text = suffix_sent + f' \'{test_sentence}\' ' + postfix_sent
    return input_text

def send_message(sentence):
    history = []
    msg = {"role":'user', "content": sentence}
    history.append(msg)
    response = openai.ChatCompletion.create(
        engine="chatgpt", 
        messages=history,
        # temperature=0.7,
        temperature=0,
        max_tokens=400,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

    n_tokens = str(response['usage']['prompt_tokens']) + '+' + str(response['usage']['completion_tokens'])
    model_type_info = response['model']
    content = response['choices'][0]['message']['content']
    content = content.replace("\n", ' ')
    stop_type = response['choices'][0]['finish_reason']
    history.append({"role":'assistant', "content": content})
    resp_info = '\t'.join([model_type_info, n_tokens, stop_type])

    return content, resp_info

def clean_str(raw_str):
    raw_str = raw_str.split(' ')
    raw_str = list(filter(lambda x: x and x.strip(), raw_str))
    return ' '.join(raw_str)

def is_substring(string_B, string_A):
    pointer_A = 0
    pointer_B = 0
    
    while pointer_A < len(string_A) and pointer_B < len(string_B):
        if string_A[pointer_A] == string_B[pointer_B]:
            pointer_B += 1
        pointer_A += 1
    
    return pointer_B == len(string_B)

def extract_response(demonstration, test_sentence, resp):
    sent_raw = test_sentence
    test_sentence = re.sub(r'\W', ' ', test_sentence)
    test_sentence = clean_str(test_sentence).lower()
    
    resp = resp.replace('.', ' . ')
    resp = resp.replace(',', ' ')

    # process 's, 'd 
    tmp = []
    for idx, ch in enumerate(resp):
        if ch in ['\'', '\"']:
            if idx == 0 or idx == len(resp) - 1:
                tmp.append(ch)
            elif ch == "\'" and resp[idx+1] in ['s', 'r', 'l','m','t','v'] and resp[idx-1] != ' ':
                continue
            else:
                tmp.append(ch)
        else:
            tmp.append(ch)
    resp = ''.join(tmp)
    
    p1 = re.compile(r'["](.*?)["]', re.S) 
    p2 = re.compile(r"['](.*?)[']", re.S) 
    
    candidate_pred = re.findall(p1, resp)
    candidate_pred.extend(re.findall(p2, resp))

    candidate_pred.append(resp)
    tmp = []
    for pred in candidate_pred:
        t = ''.join(list(filter(lambda x: x.isalpha() or x.isspace(), pred)))
        
        tmp.append(clean_str(t))
    candidate_pred = tmp
    candidate_pred = list(set(candidate_pred))
    candidate_pred = list(map(lambda x: x.lower(), candidate_pred))
    candidate_pred = list(filter(lambda x: is_substring(x, test_sentence), candidate_pred))
    candidate_pred = list(map(lambda x: x.split(' '), candidate_pred))
    
    candidate_pred = list(filter(lambda x: len(x)!=1, candidate_pred))
    candidate_pred = list(filter(lambda x: ' '.join(x) != test_sentence.lower(), candidate_pred))
    if candidate_pred == []:        
        return [demonstration[0], demonstration[1], sent_raw, 'fail to follow']
    pred = ['']
    candidate_pred = sorted(candidate_pred, key = lambda x: len(''.join(x)))
    candidate_pred = sorted(candidate_pred, key = lambda x: len(x))
    target_lst = test_sentence.split(' ')
    for i in range(len(candidate_pred[-1])-1, -1, -1):
        for j in range(len(test_sentence.split(' '))-1, i-1, -1):
            candidate_pred = sorted(candidate_pred, key = lambda x: int(x[min(i, len(x)-1)].lower() == target_lst[j].lower()))

    for candidate in candidate_pred:
        if ' '.join(candidate) == test_sentence.lower():
            continue
        if len(candidate) == 1:
            continue
        pred = candidate
    if pred == [] :
        return [demonstration[0], demonstration[1], sent_raw, 'fail to follow']
    else:
        return [demonstration[0], demonstration[1], sent_raw, " ".join(pred)]
    
def extract_response_zh(demonstration, test_sentence, resp):
    sent_raw = test_sentence
    test_sentence = re.sub(r'\W', ' ', test_sentence)
    test_sentence = clean_str(test_sentence).lower()
    
    if pd.isna(resp):
        return [demonstration[0], demonstration[1], sent_raw, 'fail to follow']
    # process 
    resp = resp.replace('.', ' . ')
    resp = resp.replace(',', ' . ')

    # comman pattern of response
    p1 = re.compile(r'["](.*?)["]', re.S) 
    p2 = re.compile(r"['](.*?)[']", re.S) 
    p3 = re.compile(r'[‘](.*?)[’]', re.S) 
    p4 = re.compile(r"[“](.*?)[”]", re.S) 
    p5 = re.compile(r"^(.*?)[。]", re.S) 
    p6 = re.compile(r"^(.*?)[，]", re.S) 
    p7 = re.compile(r"^.*?[，](.*?)[。]", re.S) 

    candidate_pred = re.findall(p1, resp)
    candidate_pred.extend(re.findall(p2, resp))
    candidate_pred.extend(re.findall(p3, resp))
    candidate_pred.extend(re.findall(p4, resp))
    candidate_pred.extend(re.findall(p5, resp))
    candidate_pred.extend(re.findall(p6, resp))
    candidate_pred.extend(re.findall(p7, resp))
    candidate_pred.append(resp)
    
    # remove naive response
    tmp = []
    for pred in candidate_pred:
        t = ''.join(list(filter(lambda x: u'\u4e00' <= x <= u'\u9fff', pred)))
        tmp.append(t)
    candidate_pred = tmp
    candidate_pred = list(set(candidate_pred))
    candidate_pred = list(map(lambda x: list(x), candidate_pred))
    candidate_pred = list(filter(lambda x: x, candidate_pred))
    candidate_pred = list(filter(lambda x: is_substring(''.join(x), test_sentence), candidate_pred))
    candidate_pred = list(filter(lambda x: len(x)!=1, candidate_pred))
    candidate_pred = list(filter(lambda x: ''.join(x).lower() != test_sentence, candidate_pred))
            
    if candidate_pred == []:
        return [demonstration[0], demonstration[1], sent_raw, 'fail to follow']
    pred = ['']
    candidate_pred = sorted(candidate_pred, key = lambda x: len(''.join(x)))
    candidate_pred = sorted(candidate_pred, key = lambda x: len(x))
    target_lst = list(test_sentence)
    for i in range(len(candidate_pred[-1])-1, -1, -1):
        for j in range(len(test_sentence)-1, i-1, -1):
            candidate_pred = sorted(candidate_pred, key = lambda x: int(x[min(i, len(x)-1)] == target_lst[j]))

    for candidate in candidate_pred:
        if ''.join(candidate) == test_sentence:
            continue
        if len(candidate) == 1:
            continue
        pred = candidate
        pred = candidate
    if pred == [] :
        return [demonstration[0], demonstration[1], sent_raw, 'fail to follow']
    else:
        return [demonstration[0], demonstration[1], sent_raw, "".join(pred)]
    