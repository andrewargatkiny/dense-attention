# DenseAttention: No-Compromise Exact All *N* × *N* Interactions Algorithm with *O(N)* Space and Time Complexity

This repository hosts the code of the official implementation and the experiments for
the paper "DenseAttention: No-Compromise Exact All *N* × *N* 
Interactions Algorithm with *O(N)* Space and Time 
Complexity".

## Introduction

This research goes towards achieving several goals:
> 1. *To create hardware efficient yet hardware-agnostic architecture with the arithmetic intensity ratio as high as possible. <...>*
> 2. *To create an algorithm which would efficiently process long sequences, preferrably with $O(N)$ time and space complexity.*
> 3. *To make the resulting architecture as simple as possible, and closely resembling original 
> Transformer architecture as well so it can serve as a drop-in replacement for the former and be easily
> adopted by both research and practitioners communities.*


<details>
<summary><h3>The main contributions</h3></summary>

From the abstract:

"We propose a novel DenseAttention Network architecture, a straightforward simplification of the standard Transformer block that addresses these issues and serves as a drop-in replacement for language modeling tasks. We eliminate memory-bound components in DenseAttention, including Softmax, masking, one skip connection, and both LayerNorms, as well as key, value, and output projection matrices, as they become redundant. Despite these removals, it maintains exact $N \times N$ pairwise interactions between tokens. By exploiting the associativity of matrix multiplications, DenseAttention can be computed with $O(N^2d)$ or $O(Nd^2)$ time and space complexity, depending on the context. To handle the absence of Softmax and prevent numerical instability, we introduce MaxNormActivation at both ends of the Transformer block. We also devise Cosine Relative Positional Embeddings as a computationally efficient replacement for RoPE, and simple LocalAttention variations of the block to help the model focus on details in extremely long contexts.
DenseAttention competes with FlashAttention in speed on small sequences and outperforms it by orders of magnitude on large contexts. We pre-train encoder language models on sequences up to 16K in length, which perform similarly or better than baseline BERT-large, while significantly improving speed and efficiency.  Finally, we achieve state-of-the-art on the LRA benchmark among the Transformer-based architectures."
</details>



### Disclaimer

The paper is undergoing  a peer review at a major ML conference. However, the code and select parts of the manuscript are subject to changes. 
> The most recent draft of the paper (October 2024) is available at: [DenseAttention.pdf](assets/DenseAttention_paper.pdf)

## Implementation and experiments

The code for DenseAttention Network, its constituents and models built upon 
it can be found in [src/](./src) directory.

For reproducing the experiments, it's recommended to create Docker 
container with preinstalled Cuda or a dedicated conda environment. To start, run:
```commandline
git clone https://github.com/andrewargatkiny/dense-attention.git
pip install -r requirements.txt
export BASE_DATA_DIR="path/to/desired/data/directory"
```
To **prepare** data, use scripts in the [data_preparation/](./data_preparation) 
directory. It can take a long time.

To **train** any of the  models, launch corresponding `ds_train_*.sh` script from the [configs/](./configs) directory 

To **evaluate** BERT models, use either `eval_all_seqs.sh` for bulk evaluations 
or individual scripts in the [bert_evals](./bert_evals) directory.

It’s recommended to use ClearML open-source ML experiments tracking system for
the training. Here are the instructions on [how to install it on your system](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/) 
and [how to use it](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/). 
ClearML is enabled  by default, if you’d prefer not to use it, disable it by typing `--no_clearml` in `deepspeed_train.py` params of the training script.

