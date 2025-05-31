#!/usr/bin/env python3
# Run with python -m utils.fuse_maxnorm_weights from project root

""" Fuse, if applicable, MaxNorm weights into queries and FFN parameters in DANet models"""
import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from src.model_config import ModelConfig
from src.modeling import DANetForPreTraining

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        "--cf",
        type=str,
        required=True,
        help="Path to the configuration .json file of the experiment which "
             "contains 'model_config' section.",

    )
    parser.add_argument(
        '--checkpoint_path',
        '--checkpoint',
        type=str,
        required=True,
        help="Path to a .pt checkpoint with model weights."
    )

    args = parser.parse_args()

    config_dict = json.load(open(args.config_file, 'r', encoding='utf-8'))
    model_config_dict = config_dict
    if 'model_config' in config_dict:
        model_config_dict = config_dict['model_config']

    config = ModelConfig(**model_config_dict)
    model = DANetForPreTraining(config, args)
    state_dict = torch.load(args.checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict['module'], strict=False)
    dtype = state_dict['module']['bert.encoder.layer.0.attention.queries'].dtype
    model.float()

    if config.pre_attn_ln_type == "scaled_max_norm":
        for layer in model.bert.encoder.layer:
            # Shape: (d, 1)
            weight = layer.activation.weight.unsqueeze(1)
            layer.attention.queries.data.mul_(weight)
            layer.attention.queries.data.mul_(weight.transpose(-1, -2))
            ffn = layer.ffn
            if hasattr(ffn, "expanding_weight"):
                ffn.expanding_weight.data.mul_(weight)
            elif hasattr(ffn, "c_up"):
                ffn.c_up.weight.data.mul_(weight.transpose(-1, -2))
            del layer.activation.weight
        print("Fused weights of scaled_max_norms")

    if config.post_attn_ln_type == "prescaled_max_norm":
        for layer in model.bert.encoder.layer:
            # Shape: (1, d)
            weight = layer.ffn_activation.weight.unsqueeze(0)
            ffn = layer.ffn
            if hasattr(ffn, "contracting_weight"):
                ffn.contracting_weight.data.mul_(weight)
            elif hasattr(ffn, "c_proj"):
                ffn.c_proj.weight.data.mul_(weight.transpose(-1, -2))
            del layer.ffn_activation.weight
        print("Fused weights of prescaled_max_norms")

    model.to(dtype)
    state_dict['module'] = model.state_dict()

    ch_path = Path(args.checkpoint_path)
    new_checkpoint = "fused_" + ch_path.name
    new_path = ch_path.with_name(new_checkpoint)
    torch.save(state_dict, new_path)
    print(f"Saved fused model weights at {new_path}")