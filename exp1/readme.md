# Experiment 1: Delete an NP Embedded in a VP in treebank sentences

Due to the nonpublic of treebanks, the scripts provided here requires the users to preprocess and extract the constituency structures of treebanks first.

However, we also provide the scripts and results of Experiment 1. You can replicate the results in our paper by using our processed data, or you can extract the annotations of treebanks and process by yourself.

## Result in Paper
The results of humans and ChatGPT are included in `result/` directory.

To replicate the results in our paper, please refer to `analysis.ipynb`, which needs about 3 mins to excute. The expected output is showed in the jupyter notebook.

## Delete an NP Embedded in a VP

The toy examples of demonstrations and test sentences are included in `stimulus/` directory, which is the subset of the stimulus employed in our paper. 

Due to the non-public of PTB and CTB, you need to download the treebank files and execute the scripts in `../script` to obtain the complete stimulus:
```bash
python process_ptb.py --process vpnp --output_path exp1/stimulus/english
python process_ctb.py --process vpnp --output_path exp1/stimulus/chinese
```

And run the script in `exp1/` to construct the tests for participants:
```bash
python construct_test.py --output_path test/chatgpt --lang en --n_subj 300
```

To run the experiment of ChatGPT, please run:
```bash
python run_chatgpt.py --test_path test/chatgpt --output_path response/chatgpt
# run_chatgpt.py requires the api of ChatGPT, which need to be set in ../utils.py
```