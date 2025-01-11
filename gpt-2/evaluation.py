from transformers import GPT2TokenizerFast
from transformers import GPT2ForQuestionAnswering
import torch
import argparse
import pandas as pd
import csv
import os
import tqdm
def createDir(filePath):
    if os.path.exists(filePath):
        return
    else:
        try:
            os.mkdir(filePath)
        except Exception as e:
            os.makedirs(filePath)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate on the model")
    parser.add_argument('--n_training_sample', '-t', dest='n_training_sample', default=1000)
    parser.add_argument('--n_iter', '-i', dest='n_iter', default=0)
    parser.add_argument('--model_path', '-m', dest='model_path', default="/path/to/model")
    parser.add_argument('--test_path', '-p', dest='test_path', default="data/exp1-en/test_sentence.csv")
    parser.add_argument('--output_path', '-o', dest='output_path', default="result/exp1-en/")

    args = parser.parse_args()
    model_path = f'{args.model_path}-{args.n_training_sample}-{args.n_iter}'
    print(f"============evaluation for: {model_path} ============")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2ForQuestionAnswering.from_pretrained(model_path)
    
    test_file = pd.read_csv(args.test_path,
                        delimiter='\t',
                        quoting=csv.QUOTE_NONE,
                        quotechar=None,)
    # sent_lst = test_file['sentence'].values.tolist()
    data_lst = list(zip(test_file['demon'].values.tolist(), 
                        test_file['d_deleted'].values.tolist(),
                        test_file['test'].values.tolist()))
    pred_lst = []
    for idx, item in enumerate(tqdm.tqdm(data_lst)):
        inputs = tokenizer(item[0] + '[SEP]' + item[1] + '[SEP]', 
                        item[2], 
                        return_tensors="pt",
                        add_special_tokens=True,
                        return_attention_mask=True,
                        )

        with torch.no_grad():
            outputs = model(**inputs)
            
        answer_start_index = int(outputs.start_logits.argmax())
        answer_end_index = int(outputs.end_logits.argmax())
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        pred = tokenizer.decode(predict_answer_tokens).strip()
        pred_lst.append([item[0], item[1], item[2], str(answer_start_index), str(answer_end_index), pred])
    createDir(f"{args.output_path}/")
    save_path = f"{args.output_path}/{args.n_training_sample}-{args.n_iter}.csv"
    with open(save_path, 'w') as f:
        f.write("demon\td_label\ttest\tstart\tend\tprediction\n")
        for item in pred_lst:
            f.write("\t".join(item) + '\n')