# MatMuls are Enough for Linear-Time Dense Attention

This repository hosts the code of the official implementation and the experiments for
the paper "MatMuls are Enough for Efficient and Performant Linear-Time Attention".
It also holds the code of the framework for training and evaluation of 
efficient  linear-time architectures, which is under active development and will be used in future research.

> ICML 2025 Workshops version of the paper: 
> * LCFM (short format) https://openreview.net/forum?id=yLh58rr4JX
> * ES-FoMo III (extended version) https://openreview.net/forum?id=RttNumxC1t
> 
> (Also available [locally](assets/Matmuls_are_Enough_ICML_Jun2025_%20CameraReady.pdf))
> 
> DANet-BERT models on HuggingFace: *to be added soon* 
## About DenseAttention

In this research, we propose **DenseAttention** and **DenseAttention Network (DANet)** 
as a replacement for softmax self-attention and Transformer block, respectively. 
Key advantages of the new architecture include:
* $O(N)$ time and space complexity with respect to sequence length $N$;
* Speed vs Transformers: DANet is faster than low-level FlashAttention implementation
even on small sequences despite being written in plain PyTorch;
* Faster speed over other linear-time sequence processing algorithms;
* Compatibility and accessibility: DenseAttention does not require low-level CUDA code,
runs on every device where PyTorch can be installed, and works well with both 
bf16 and older fp16 half-precision formats.

<details>
<summary><h3>Speed Comparisons</h3></summary>

| Model (Hardware) / Ctx Size | 128 | 1024 | 4096 | 16384 | 65536 | 131072 |
|---------------------------|-------|-------|-------|--------|--------|---------|
| Transformer (H100) | 736.05 | 571.39 | 318.46 | 116.74 | 33.29 | 16.87 |
| Linear Attention (H100) | 563.37 | 568.19 | 568.07 | 566.95 | 566.62 | 565.84 |
| **DANet (H100)** | **772.03** | **699.60** | **701.93** | **700.73** | **697.89** | **690.36** |
| Transformer (A100) | 303.62 | 257.54 | 165.46 | 68.04 | 20.27 | 10.47 |
| Linear Attention (A100) | 243.72 | 241.66 | 242.81 | 241.65 | 243.39 | 242.73 |
| **DANet (A100)** | **313.25** | **277.52** | **277.71** | **277.92** | **273.71** | **272.96** |
| Transformer (CPU) | 7.99 | 2.21 | 0.62 | 0.16 | OOM | OOM |
| Linear Attention (CPU) | 7.67 | 7.75 | 7.67 | 7.73 | 7.75 | 7.82 |
| **DANet (CPU)** | **14.97** | **13.60** | **13.21** | **12.94** | **13.46** | **12.83** |
Throughput (thousands tokens per second) comparison for 330M–parameters encoder models.

</details>


DenseAttention achieves better speed, linear complexity, and computational efficiency 
without compromising the modeling performance. This is exemplified by, among other 
experiments, DANet-BERT-Large LM 
pre-training on approximately 500B tokens and fine-tuning on the GLUE benchmarks, which yields best results 
out of all models of comparable size trained on sub-trillion token count data. It's also supported by 
validation on the LRA suite of benchmarks where it outperforms all previous 
Transformer-based architectures at least by 5%.

### The architecture 

[DenseAttention](./src/dense_attention.py)  is a novel self-attention mechanism which eliminates softmax and does 
not introduce any replacements. It also merges $W_Q$ and $W_K$ projection matrices into a single parameter.
Remarkably, it is composed *entirely* of dense MatMuls.

The whole [DANet](./src/danet_layers.py) further simplifies Transformer module by reducing the number of other element-wise 
operations and merging projection matrices. To ensure numerical stability, We replace 
standard `LayerNorm` with [MaxNormActivation](./src/activations.py), and for adding sharp 
focus on nearby tokens in extremely long contexts, we introduce `local` and `shifted local` 
DenseAttention layers designed to complement standard layers with global receptive field.
More details are provided [in the paper](assets/Matmuls_are_Enough_ICML_Jun2025_%20CameraReady.pdf).






## Implementation and Experiments

### Disclaimer

> This is an ongoing project. Some parts of the code and configs are subject to change. 
For the most current version and new features, please checkout `dev` branch.
---

The code for DenseAttention Network, its constituents and models built upon 
it can be found in [src/](./src) directory.

All settings for DenseAttention and DANet are documented in HuggingFace-like [model config](./src/model_config.py) file.

DANet can serve as a drop-in replacement for Transformer modules. To start building your own models using DANet, copy `src` directory into your project
and import required layers:

```python
import torch
from src.danet_layers import DANetLayer, DANetLayerWithLocalAttention
from src.model_config import ModelConfig

SIZE=128
config = ModelConfig(hidden_size=SIZE)
danet_layer = DANetLayer(config) # or DANetLayerWithLocalAttention(config)
inputs = torch.randn(8, 1024, SIZE) # batch size, sequence length, model dimension
outputs = danet_layer(inputs)

```

For examples on how to incorporate various types of positional embeddings into DANet
with and without local attention, see `DANetEncoder` in [modeling](./src/modeling.py).

### Experiments 
For reproducing the experiments from the paper, it's recommended to create Docker 
container with preinstalled Cuda or a dedicated conda environment. To start, run:
```commandline
git clone https://github.com/andrewargatkiny/dense-attention.git
cd dense-attention && pip install -r requirements.txt
export BASE_DATA_DIR="path/to/desired/data/directory"
```
To **prepare** data, use scripts in the [data_preparation/](./data_preparation) 
directory. It can take a long time. TODO: add c4 preparation script for DANet-BERT pretraining.

To reproduce an experiment, or simply **train** / **evaluate** a model, launch corresponding `ds_train_*.sh` 
script from the [configs/](./configs) directory which contains exact configurations 
for the experiments in the paper. Under the hood, it launches [deepspeed_train.py](deepspeed_train.py) 
with a preconfigured set of arguments. 
Additionally, you can **override any configuration parameter** directly
[from the command line](configs/README.md#overriding-config-params-via-cli-at-launch-time) 
using the `--override` flag, without modifying the original config files. 
To construct your own pipeline or to learn  about each argument's effect, please refer to [train_arguments.py](train_arguments.py) script and 
[Using Configs to Run Experiments](configs/README.md).


It’s recommended to use `ClearML` open-source ML experiments tracking system for
the training. You can use it in the cloud up to a small storage limit or install on
your server, for free. Here are the instructions on [how to install it on your system](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/) 
and [how to use it](https://clear.ml/docs/latest/docs/). 
ClearML is enabled  by default, but it's also possible to use `Weights & Biases` or `Tensorboard` by providing 
`--wandb` or `--tensorboard` argument in `deepspeed_train.py` params of the training script.




## Citation

If you use DenseAttention in research or production, or otherwise find it useful, please cite it as:

```
@inproceedings{
argatkiny2025matmuls,
title={MatMuls are Enough for Efficient and Performant Linear-Time Attention},
author={Andrew Argatkiny and Ilya Makarov},
booktitle={ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models},
year={2025},
url={https://openreview.net/forum?id=RttNumxC1t}
}
```

