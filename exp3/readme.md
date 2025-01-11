# Experiment 3: Delete an Arbitrary Constituent

Due to the nonpublic of treebanks, the scripts provided here requires the users to preprocess and extract the constituency structures of treebanks first.

However, we also provide the scripts and results of Experiment 3. You can replicate the results in our paper by using our processed data, or you can extract the annotations of treebanks and process by yourself.

## Result in Paper
The results of humans and ChatGPT are included in `result/` directory.

To replicate the results in our paper, please refer to `analysis.ipynb`, which needs about 3 mins to excute. The expected output is showed in the jupyter notebook.

## Delete an Arbitrary Constituent

The toy examples of demonstrations and test sentences are included in `stimulus/` directory, which is the subset of the stimulus employed in our paper. 

Due to the non-public of PTB and CTB, you need to download the treebank files and execute the scripts in `../script` to obtain the whole stimulus:
```bash
python process_ptb.py --process random --output_path exp3/stimulus/
python process_ctb.py --process random --output_path exp3/stimulus/
```

And run the script in `exp3/` to construct the splitted tests for participants:
```bash
python construct_test.py --lang english --output_path test/chatgpt --n_subj 300
```

To run the experiment of ChatGPT, please run:
```bash
python run_chatgpt.py --test_path test/chatgpt --output_path response/chatgpt
# run_chatgpt.py requires the api of ChatGPT, which need to be set in ../utils.py
```