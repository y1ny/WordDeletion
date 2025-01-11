# Experiment 2: Delete an NP Embedded in a VP in Chinese-English parallel sentences

We provide the scripts and results of Experiment 2, since Experiment 2 only requires the users to process the parallel sentences.

## Result in Paper
The results of humans and ChatGPT are included in `result/` directory.

To replicate the results in our paper, please refer to `analysis.ipynb`, which needs about 3 mins to excute. The expected output is showed in the jupyter notebook.

## Delete an NP Embedded in a VP

The examples of demonstrations and test sentences are included in `stimulus/` directory. 

And run the script in `exp2/` to construct the tests for participants:
```bash
python construct_test.py --output_path test/chatgpt --lang en --n_subj 300
```

To run the experiment of ChatGPT, please run:
```bash
python run_chatgpt.py --test_path test/chatgpt --output_path response/chatgpt
# run_chatgpt.py requires the api of ChatGPT, which need to be set in ../utils.py
```