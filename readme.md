# Word Deletion Task
Code and data for the paper "Active Use of Latent Tree-structured Sentence Representation in both Humans and Large Language Models". See [the paper](https://arxiv.org/abs/2405.18241) and [the supplementary information](https://y1ny.github.io/assets/word_deletion_SI.pdf)
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

**Notice**: Experiments 1 & 3 & 4 require the annotation from [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) and [Chinese Treebank](https://catalog.ldc.upenn.edu/LDC2013T21). Due to the nonpublic of treebanks, we can not release the annotation file. Experiments 2 & 5 & 6 can be replicated without treebanks (see readme file in corresponding directories).

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
python scripts/process_ctb.py --process filter --input_path treebank/ctb/extract --output_path treebank/ctb/processed
```

To convert the constituency tree to dependency tree, please follow the instruction in the TreebankProcessing (using `tb_to_stanfo.py`), then, run:

```bash
python scripts/process_ptb.py --process dependency --input_path path/to/dependency_tree 
python scripts/process_ctb.py --process filter --input_path path/to/dependency_tree 
```

## Run Experiment for ChatGPT

Run `run_gpt.py` in each directory

## Exp 1&2&3&5: Constituent Rate & Explaiend Ratio of Rules

Follow and run the code in `analysis.ipynb` in each directory (including visualization)


## Exp 2: Chinese-English Parallel Sentences

The Chinese-English parallel sentences are provided in the `exp2/stimulus` directory.

## Exp 1&3: Naive GPT-2

All experiments about GPT-2 are in `gpt-2` directory

## Exp 4: Tree Reconstruction 

To reconstruct a tree based on the deletion task, please switch to the `exp4` directory.

To reconstruct trees of the sentences employed in our study, please refer to `exp4/analysis.ipynb`.

If you want to reconstruct your own sentence, please provide your API key:

```bash
python run_chatgpt.py --sentence 'your own sentence' --output_path ./output
python tree_reconstruction.py --sentence 'your own sentence' --response ./output/response.csv
```

This script would construct multiple tests for your sentence, run these tests on ChatGPT, and reconstruct the tree.

## Exp 5: Syntactically Correct Meaningless Sentence

The syntactically correct meaningless sentences are provided in the `exp5/stimulus` directory.

## Exp 6: Syntactically Ambiguous Sentence

The syntactically ambiguous sentences with adjunct or PP attachment are provided in the `exp5/stimulus` directory.

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