

# Using Configs to Run Experiments

Each of the `ds_train_*.sh` scripts is preconfigured to run with some default values.
Some of them (arguments for `deepspeed_train.py` script, see [train_arguments.py](../train_arguments.py)) can be changed only by 
manually editing the code, and other can be supplied as CLI parameters:

* `SEED` sets a single training seed for all frameworks and libraries (Python, NumPy, PyTorch, etc.);
* `NODE` sets the GPU(s) to run experiments on. E.g.: `0` or `0,1,2`;
* `MASTER_PORT` should hold different values if you run more than one experiment simultaneously;
* `BASE_DATA_DIR` is a root path to your training data directory;
* `BASE_OUT_DIR` is a root path to save model checkpoints;
* `TASK_TYPE` see [tasks.py](../utils/tasks.py);
* `DS_CONFIG` is mainly used for setting global/local batch size,
and gradient accumulation steps, optimizer parameters such as $\beta_1$ and $\beta_2$, 
and floating point number format (bf16/fp16/fp32/amp). NOTE: values for weight decay 
and learning rate do not influence the training process at all.
* `CONFIG` is a bundled config file which includes
  - `MODEL_CONFIG` (config for DANet or 
  other model which should conform to the interface of
  [DANet config](../src/model_config.py) or [Transformer config](../src/other_models/modeling.py));
  - `DATA_CONFIG` (holds base paths and init paremeters for experiments' datasets);
  - `TRAINING_CONFIG` (sets learning rate and scheduler parameters)

Currently, not all parameters are supported for each config.

## Running on CPU

If you want to run an experiment on CPU, change this line

```commandline
NCCL_TREE_THRESHOLD=0 deepspeed --master_port "$MASTER_PORT" ${base_dir}/deepspeed_train.py \
```
in the corresponding `ds_train_*.sh` file to this:
```commandline
DS_ACCELERATOR="cpu" deepspeed --num_accelerators 1 --bind_cores_to_rank --bind_core_list 112-139 ${base_dir}/deepspeed_train.py \
```
`--num_accelerators, --bind_cores_to_rank, --bind_core_list` are optional parameters for multi-socket systems.

You also should set `--dist_backend` argument of `deepspeed_train` to some other value 
aside from `'nccl'`. See https://pytorch.org/docs/stable/distributed.html#backends for available options.

## Overriding config params via CLI at launch time

In addition, you can **override any configuration parameter** directly from the command line using the `--override` flag, without modifying the original config files. This allows quick and flexible experimentation while keeping base configs unchanged.
Each override must include the full path to the parameter, reflecting its nesting in the configuration structure.

**Overriding parameters from different configuration sources is distinguished using the following prefixes:**

- `cf.` - for parameters from the **Model config**
- `ds.` - for parameters from the **DeepSpeed config**

**Supported Override Types**

| Type                        | Example                                                                 |
|-----------------------------|-------------------------------------------------------------------------|
| Integer                     | `cf.model_config.num_attention_heads=1`                                |
| Boolean                     | `cf.model_config.local_attention=false`                                |
| Float                       | `ds.optimizer.params.eps=1e-16`, `ds.optimizer.params.weight_decay=0.011`|
| String                      | `cf.data.validation.inputs=input/train.src`                            |
| Quoted String               | `cf.data.validation.labels="label/train.label"`                        |
| List                        | `ds.optimizer.params.betas=[0.9,0.9]`                                   |
| Dictionary                  | `cf.data.training='{"inputs":"input/valid.src","labels":"label/valid.label"}'` |


**Notes:**

- String values can be written as: "text" or text.
- For dictionaries, use single quotes around the full JSON object.


**Example Usage**

```bash
ds_train_*.sh \
  --override \
  cf.model_config.num_attention_heads=1 \
  cf.model_config.local_attention=false \
  cf.data.validation.inputs=input/valid.src \
  cf.data.validation.labels="label/valid.label" \
  ds.train_batch_size=32 \
  ds.train_micro_batch_size_per_gpu=32 \
  ds.optimizer.params.eps=1e-16 \
  ds.optimizer.params.betas=[0.9,0.9] \
  ds.optimizer.params.weight_decay=0.011 \
  cf.data.training='{"inputs":"input/valid.src","labels":"label/valid.label"}'
  ```


## Examples

Here are some usage examples:

### DANet-BERT and GPT/LLAMA pre-training

From scratch:

```commandline
export BASE_DATA_DIR="path/to/your/root/data/directory"

configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh
```

You can also use different bundled model/data/training config, such as `bert_large_rope_swiglu_seq128.json`:

```commandline
CONFIG=configs/bert_relpe/bert_large_rope_swiglu_seq128.json configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh
```
Continue from checkpoint:

```commandline
CONFIG=configs/bert_relpe/bert_large_rope_swiglu_seq128.json configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh --resume 241 bert_pretraining_YYYY-MM-DD_hh-mm
```

Here, `bert_pretraining_YYYY-MM-DD_hh-mm` is a directory with checkpoints for a previous experiment.

GPT, using non-default path to save checkpoints:

```commandline
CONFIG=configs/gpt/gpt_hybrid_360m.json BASE_DATA_DIR="/path/to/your/empty/storage/" configs/gpt/ds_train_dense_attn_gpt_360m_bf16.sh
```


### LRA experiments

```commandline
SEED=100 configs/lra/ds_train_dense_attn_pathfinder32.sh
```

If you have several GPUs on one node, you can manage which experiment(s) to launch on 
each of the card, even running several experiments on the same device:

```commandline
# Each command should be executed in different Terminal (screen / tmux) window

SEED=0 NODE=0 MASTER_PORT=29500 configs/lra/ds_train_dense_attn_listops.sh
SEED=1 NODE=0 MASTER_PORT=29501 configs/lra/ds_train_dense_attn_listops.sh
SEED=2 NODE=1 MASTER_PORT=29502 configs/lra/ds_train_dense_attn_listops.sh
SEED=3 NODE=1 MASTER_PORT=29503 configs/lra/ds_train_dense_attn_listops.sh
```
### GLUE fine-tuning

```commandline
configs/glue/ds_train_rte.sh --resume 4 bert_large_128_rope_loc32_swiglu_mnli
```

### Speed Evaluation

In this example, we evaluate Linear Transformer (MODEL_CONFIG=configs/speed_eval/linear_attn_bert.json) 
on MLM task (TASK_TYPE=transformer_bert_mlm) using custom data config
(DATA_CONFIG=configs/speed_eval/data_seq131k.json) for sequence with 131K-tokens length
and custom batch size (DS_CONFIG=configs/speed_eval/deepspeed_config_bs2_bf16.json)

```commandline
MODEL_CONFIG=configs/speed_eval/linear_attn_bert.json TASK_TYPE=transformer_bert_mlm DATA_CONFIG=configs/speed_eval/data_seq131k.json DS_CONFIG=configs/speed_eval/deepspeed_config_bs2_bf16.json configs/speed_eval/eval_bert.sh
```