# Experiment 6: Semantic Constraints on Constituency

All scripts provided here do not involve the annotations from treebanks, and can be executed directly.
## Result in Paper
The results of humans and ChatGPT are included in `result/` directory.

To replicate the results in our paper, please refer to `analysis.ipynb`, which needs about 3 mins to excute. The expected output is showed in the jupyter notebook.

## Syntactically Ambiguous Sentence

The syntactically ambiguous sentences are included in `stimulus/` directory. You can construct your own tests by running:
```bash
python construct_test.py --output_path test/chatgpt --n_subj 300
```

To run the experiment of ChatGPT, please run:

```bash
python run_chatgpt.py --test_path test/chatgpt --output_path response/chatgpt
# run_chatgpt.py requires the api of ChatGPT, which need to be set in ../utils.py
```