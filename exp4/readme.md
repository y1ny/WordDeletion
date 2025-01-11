# Experiment 4: Reconstruction of Constituency Tree via Deletion

All scripts provided here do not involve the annotations from treebanks, and can be executed directly.
## Result in Paper
The results of humans and ChatGPT are included in `result/` directory.

To replicate the results in our paper, please refer to `analysis.ipynb`, which needs about 20 mins to excute. The expected output is showed in the jupyter notebook.

Due to the nonpublic of treebanks, we only provide the deletion-based trees of ChatGPT and humans, as well as the chance.

## Reconstruct Your Own Tree
To reconstruct your own tree, please run:

```bash
python run_chatgpt.py --sentence 'your own sentence' --output_path ./output
# run_chatgpt.py requires the api of ChatGPT, which need to be set in ../utils.py
python tree_reconstruction.py --sentence 'your own sentence' --path ./output
```

The `run_chatgpt.py` will pair your sentence with multiple distinct demonstrations, resulting in multiple tests (N = 833 for Chinese and N = 354). The created tests will be tested on ChatGPT, which will create the response file in 'output/response.csv'.

The `tree_reconstruction.py` will use these responses to reconstruct a constituency tree, and plot its constituency structure in the terminal. 

If you want to reconstruct a tree based on other participants (e.g., humans), please change the `path` argument in `tree_reconstruction.py`.

We also provide the demonstration set for human participants, which is sampled based on less combinations of syntactic category (see Methods)
