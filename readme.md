# Word Deletion Task
Code and data for the paper "Active Use of Latent Constituency Representation in both Humans and Large Language Models". See [the paper](https://arxiv.org/abs/2405.18241) and [the supplementary information](https://y1ny.github.io/assets/word_deletion_SI.pdf)
## Prerequisites
Python 3.9.12. No non-standard hardware is required.

## Install & Getting Started

1. Clone the repository

2. Construct a virtual environment for this project

```bash
conda env create -f environment.yml
# change the prefix to your own anaconda path, about 30 mins
```

3. Download the treebank library to treebank/ptb and treebank/ctb

**Notice**: Experiment 1 & 2 require the annotation from [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) and [Chinese Treebank](https://catalog.ldc.upenn.edu/LDC2013T21). Due to the nonpublic of treebanks, we can not release the annotation file. Experiment 3 & 4 can be replicated without treebanks (see readme file in corresponding directories).

## Preparing Treebanks

The preprocessing of treebanks relies on the [TreebankProcessing](https://github.com/hankcs/TreebankPreprocessing) library.

To replicate the preprocessing procedure in our work, you need to download the TreebankProcessing and run:

```bash
python ptb.py --output treebank/ptb/extract --task par
python ctb.py --output treebank/ctb/extract --task par
```

then, run:

```bash
python scripts/process_ptb.py --process filter --input_path treebank/ptb/extract --output_path treebank/ptb/processed
python scripts/process_ctb.py --process filter --input_path treebank/ctb/extract --output_path treebank/ptb/processed
```

## Run Experiment for ChatGPT

Run `run_gpt.py` in each directory

## Exp 1&2: Analysis Constituent Rate & Explaiend Ratio of Rules

Follow and run the code in `analysis.ipynb` in each directory (including visualization)

## Exp 1&2: Naive LSTM

All experiments about LSTM are in `lstm` directory

## Exp 3: Tree Reconstruction 

To reconstruct a tree based on the deletion task, please switch to the `exp3` directory.

To reconstruct trees of the sentences employed in our study, please refer to `exp3/analysis.ipynb`.

If you want to reconstruct your own sentence, please provide your API key:

```bash
python run_chatgpt.py --sentence 'your own sentence' --output_path ./output
python tree_reconstruction.py --sentence 'your own sentence' --response ./output/response.csv
```

This script would construct multiple tests for your sentence, run these tests on ChatGPT, and reconstruct the tree.

## Exp 4: Syntactically Ambiguous Sentence

The syntactically ambiguous sentences with adjunct or PP attachment is provided in the `exp4/stimulus` directory.

## Citation

If you make use of the code in this repository, please cite the following papers:

```
@article{liu2024active,
  title={Active Use of Latent Constituency Representation in both Humans and Large Language Models},
  author={Liu, Wei and Xiang, Ming and Ding, Nai},
  journal={arXiv preprint arXiv:2405.18241},
  year={2024}
}
```