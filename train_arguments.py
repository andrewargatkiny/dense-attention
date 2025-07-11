import sys
import argparse
import json

   
def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument(
        "--config-file",
        "--cf",
        type=str,
        required=True,
        help="pointer to the main configuration file of the experiment",
    )
    parser.add_argument(
        "--model_config_file",
        default="",
        type=str,
        help="Path to optional model config. If set, overrides `model_config` "
             "section of the main config."
    )
    parser.add_argument(
        "--data_config_file",
        default="",
        type=str,
        help="Path to optional data config. If set, overrides `data` "
             "section of the main config."
    )
    parser.add_argument(
        "--train_config_file",
        default="",
        type=str,
        help="Path to optional training config. If set, overrides `training` "
             "section of the main config."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )
    # Optional Params
    parser.add_argument(
        "--task_type",
        default="sequence_classification",
        type=str,
        help="Type of the task which controls dataset class, model, etc. to "
             "be used."
    )
    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="Only evaluate the model without training"
    )
    parser.add_argument(
        "--only_mlm_task",
        default=False,
        action="store_true",
        help="When running *_classification_mlm type of task, perform only MLM. "
             "Models weights for classification task are still preserved and "
             "can be used in successive runs."
    )
    parser.add_argument(
        "--only_cls_task",
        default=False,
        action="store_true",
        help="When running *_classification_mlm type of task, perform only cls. "
             "Models weights for MLM task are still preserved and can be used "
             "in successive runs."
    )
    parser.add_argument(
        "--no_decay_embeddings",
        default=False,
        action="store_true",
        help="If True, don't apply weight decay to embeddings even if it's not"
             " 0 in general."
    )
    parser.add_argument(
        "--no_decay_pooler",
        default=False,
        action="store_true",
        help="If True, don't apply weight decay to pooler even if it's not"
             " 0 in general."
    )
    parser.add_argument(
        "--scale_ffn_weights",
        default=False,
        action="store_true",
        help="Scale weights of FFN so their norm is less than a predefined value."
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=2**16,
        type=int,
        help="The maximum number of masked tokens in a sequence to be predicted."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random seed for initialization"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        '--materialize_ffn_weights',
        default=False,
        action='store_true',
        help="Whether to multiply ffn weights to its norm ratio to unite them"
             " for fine-tuning."
    )
    parser.add_argument(
        '--load_training_checkpoint',
        '--load_cp',
        type=str,
        nargs='?',
        const=None,
        default=None,
        help=
        "This is the path to the TAR file which contains model+opt state_dict() checkpointed."
    )
    parser.add_argument(
        '--load_checkpoint_id',
        '--load_cp_id',
        type=str,
        nargs='?',
        const=None,
        default=None,
        help='Checkpoint identifier to load from checkpoint path'
    )
    parser.add_argument(
        "--load_only_weights",
        default=False,
        action="store_true",
        help="When restarting from a DeepSpeed checkpoint, load only weights "
             "without optimizer states."
    )
    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help="This is the path to store the output and TensorBoard results."
    )
    parser.add_argument(
        '--rewarmup',
        default=False,
        action='store_true',
        help='Rewarmup learning rate after resuming from a checkpoint'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size to complete.'
    )
    parser.add_argument(
        '--max_steps_per_epoch',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size within an epoch to complete.'
    )
    parser.add_argument(
        '--log_diagnostic_freq',
        type=int,
        default=100,
        help='Interval in epochs to log model weights and, possibly, activations.'
    )
    parser.add_argument(
        '--log_weight_norms', 
        action='store_true',
        help='Log user-chosen norm of parameters (weights) grouped by parameter type'
    )
    parser.add_argument(
        '--logging_norm_type', 
        type=str,
        default="L1",
        help='Vector norm of parameters (weights) to be logged. Valid options are: L1, L2, Linf'
    )
    parser.add_argument(
        '--data_path_prefix',
        type=str,
        default="",
        help="Path to prefix data loading, helpful for AML and other environments"
    )
    parser.add_argument(
        '--validation_data_path_prefix',
        type=str,
        default=None,
        help="Path to prefix validation data loading, helpful if pretraining "
             "dataset path is different"
    )
    parser.add_argument(
        "--use_sharded_dataset",
        default=False,
        action="store_true",
        help="Indicates that the training dataset is composed of many files "
             "and invokes mechanisms to process them with distributed training."
    )
    parser.add_argument(
        "--no_eval_val_data",
        default=False,
        action="store_true",
        help="Don't evaluate on validation data"
    )
    parser.add_argument(
        "--eval_train_data",
        default=False,
        action="store_true",
        help="Evaluate on train data"
    )
    parser.add_argument(
        "--eval_test_data",
        default=False,
        action="store_true",
        help="Evaluate on test data"
    )
    parser.add_argument(
        '--eval_bs_ratio',
        default=8,
        type=int,
        help='Ratio indicating how many times the eval batch size is '
             'greater than training one.'
    )
    parser.add_argument(
        "--max_validation_samples",
        default=-1,
        type=int,
        help="Max samples in an evaluation dataset to be used at eval time."
    )
    parser.add_argument(
        '--tracking_system',
        default='clearml',
        type=str,
        help="ML tracking system to log the experiment details."
        "Valid options are: 'clearml', 'tensorboard', or 'wandb'. Default is 'clearml'."
    )
    parser.add_argument(
        '--ckpt_to_save',
        default=20,
        type=int,
        help=
        'Indicates how often to save checkpoints, i. e. each 5th epoch. Default is 20.'
    )
    parser.add_argument(
        '--dense_attention',
        default=False,
        action='store_true',
        help="Currently affects how params are grouped in the optimizer."
    )
    parser.add_argument(
    '--resize_posit_embeds',
    default=False,
    action='store_true',
    help= 'If this option is invoked, model is loaded from a checkpoint, and '
          'current config has larger `max_position_embeddings` dimension than '
          'checkpointed model, then weights from its posit embeddings get '
          'copied to appropriate positions of current model ones. Then '
          'deepspeed.initialize() gets called.'
    )
    parser.add_argument(
        '--throughput_logging_samples',
        type=int,
        default=3000,
        help=
        "Minimal number of data samples to calculate and log "
        "various metrics such as training speed and last batch's loss"
    )
    parser.add_argument(
        '--inputs_logging_ratio',
        type=float,
        default=1.,
        help=
        "Which portion of a one-device microbatch inputs to use when "
        "calculating forward pass to log model's activations."
    )
    parser.add_argument(
        '--log_activations',
        default=False,
        action='store_true',
        help='Log activation distributions for each parameter of the model every .'
    )

    parser.add_argument("--keep_last_ckpts", type=int, default=3,
                        help="Keep this many most-recent checkpoints (this run) "
                             "from deletion.")
    parser.add_argument("--keep_ckpt_every", type=int, default=50,
                        help="Protect checkpoint from deletion if "
                             "(epoch-1) %% keep_ckpt_every == 0")
    parser.add_argument("--keep_ckpt_epochs",
                        type=lambda s: [int(x) for x in s.split(',') if s],
                        default=[],
                        help="Comma-separated epochs (e.g. 100,180,200) that "
                             "must never be deleted")
    # TODO: implement offset functionality for saving checkpoints and keeping them from deletion
    

    parser.add_argument(
        '--unpad_inputs',
        default=False,
        action='store_true',
        help='Whether to unpad inputs for efficient training in case'
             ' of uneven seq lengths')
    parser.add_argument(
        '--use_torch_compile',
        default=False,
        action='store_true',
        help='Use torch.compile() to speed up training'
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default="new-attention-lra",
        help='Name of the project for ClearML'
    )
    parser.add_argument(
        '--num_labels',
        type=int,
        default=2,
        help='Number of labels for classification tasks'
    )
    parser.add_argument(
        "--lm_prob",
        type=float,
        default=0.15,
        help="Masking probability in dynamic masked language modeling "
             "datasets."
    )
    parser.add_argument(
        "--variable_mask_rate",
        default=False,
        action="store_true",
        help="In dynamic masked language modeling datasets, mask each sequence "
             "with variable masking probability/rate from 0 to --lm_prob value."
    )
    parser.add_argument(
        "--mlm_use_rtc_task",
        default=False,
        action="store_true",
        help="In MLM family of tasks instead of plain MLM use Replaced Tokens "
             "Correction task which replaces a portion of a sequence's tokens "
             "with random tokens and asks the model to predict true labels for "
             "ALL tokens without giving hints which were corrupted and which "
             "are correct."
    )
    parser.add_argument(
        "--mask_token_id", "--vocab_length",
        type=int,
        default=31,
        help="Id of the `MASK` token for replacement in dynamic language "
             "modeling datasets which should be equal to number of vocab's "
             "entries."
    )
    parser.add_argument(
        '--zero_init_pooler',
        default=False,
        action='store_true',
        help='Reinitialize pooler weights to all 0s before training.'
    )
    parser.add_argument(
        '--dict_backend',
        type=str,
        default="nccl",
        help='Backend for distributed training.'
    )
    parser.add_argument(
        '--override',
        type=str,
        default=[],
        nargs='*',
        help='Override parameters for config files. Accepts values in the form '
             '"cf.key=val" or "ds.key=val" to override nested values in the model (cf) '
             'or deepspeed (ds) configs. For nested keys, use dot notation: '
             'key = key1.key2.key3.'
        
    )

    return parser


"""
Configuration override implementation.
Inspired by the implementation in the flame project:
https://github.com/fla-org/flame
"""
def parse_override_string(override: str):
    """Parse a single override string in the format 'cf.key=value' or 'ds.key=value'.
    """
    first_dot = override.find('.')
    end_eq = override.find('=')
    
    if first_dot == -1 or end_eq == -1:
        raise ValueError(f"Invalid override format: '{override}' (expected: cf.key=value or ds.key=value)")
    
    cfg_name = override[:first_dot]
    key_path = override[first_dot + 1:end_eq]
    value = override[end_eq + 1:]
    
    print(f"Parsed override: key_path={key_path}, value={value}")
    return (cfg_name, key_path, value)

def set_nested_checked(config_dict: dict, key_path: str, value):
    """Safely set a nested value in a configuration dictionary, validating the path exists.
    """
    keys = key_path.split('.')
    target_dict = config_dict

    for key in keys[:-1]:
        if key not in target_dict or not isinstance(target_dict[key], dict):
            raise KeyError(f"The path '{key_path}' does not exist under the key '{key}'")
        target_dict = target_dict[key]

    final_key = keys[-1]
    if final_key not in target_dict:
        raise KeyError(f"The final key '{final_key}' does not exist in '{key_path}'")

    target_dict[final_key] = value
    print(f"Set {key_path} = {value}")
    
def parse_value(val: str):
    """
    Parses a string into the appropriate JSON-compatible type.
    If parsing fails, returns the original string.

    Supported types:
    - int:        "1", "32"
    - float:      "0.011", "1e-16"
    - bool:       "true", "false"
    - list:       "[0.9, 0.9]"
    - dict:       '{"inputs":"input/train.src","labels":"label/train.label"}'
    - str:        "\"input/train.src\"", "input/train.src"

    Examples:
        parse_value("1")                                → 1 (int)
        parse_value("false")                            → False (bool)
        parse_value("1e-16")                            → 1e-16 (float)
        parse_value("[0.9, 0.9]")                       → [0.9, 0.9] (list)
        parse_value("\"input/train.src\"")              → "input/train.src" (str, JSON string)
        parse_value("input/train.src")                  → "input/train.src" (raw string)
        parse_value('{"inputs":"input/train.src","labels":"label/train.label"}')
                                                        → {"inputs": "input/train.src",
                                                        "labels": "label/train.label"} (dict)

    Notes:
    - String values can be written as: "text" or text (no need for escape sequences).
    - For dictionaries, use single quotes around the full JSON object.
    """
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return val

def apply_overrides(configs: dict[str, dict], overrides: list[tuple[str, str, str]]):
    """Apply a list of parsed overrides to their respective configuration dictionaries.
    """
    for cfg_name, key_path, raw_value in overrides:
        if cfg_name not in configs:
            raise KeyError(f"Use --override cf. or --override ds.")
        value = parse_value(raw_value)
        set_nested_checked(configs[cfg_name], key_path, value)

def override_configs(args):
    """Load, apply overrides, and save configuration files based on CLI arguments.
    """
    configs = {
        "cf": args.config,
        "ds": args.deepspeed_config
    }
    
    overrides = [parse_override_string(ovr) for ovr in args.override]
    apply_overrides(configs, overrides)
    print("Args updated: args.config and args.deepspeed_config now hold dicts")

    return args
    