import copy
from types import SimpleNamespace

import pytest
import torch

from src.other_models import modeling as modeling_new
from src.other_models import modeling_legacy


BASE_CONFIG = {
    "vocab_size_or_config_json_file": 128,
    "hidden_size": 32,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "attention_kernel": "softmax",
    "feature_map": None,
    "no_reweight": False,
    "no_reweight_post_norm": None,
    "hidden_act": "gelu",
    "embedding_dropout": 0.0,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "attn_proj_biases": True,
    "max_position_embeddings": 32,
    "pos_emb_type": "learned",
    "relpe_type": None,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
    "pre_attn_ln_type": "default",
    "post_attn_ln_type": "default",
    "causal": False,
    "local_attention": False,
    "local_scheme": None,
    "window_size": 8,
    "apply_relpe_after": False,
    "power": 2,
    "scaling_d_factor": False,
}

CASES = [
    {"id": "softmax_global", "override": {}},
    {
        "id": "linear_global",
        "override": {
            "attention_kernel": "linear",
            "feature_map": "identity",
        },
    },
    {
        "id": "power_global",
        "override": {"attention_kernel": "power", "power": 2},
    },
    {
        "id": "rope_global",
        "override": {
            "pos_emb_type": "relpe",
            "relpe_type": "rope",
            "attention_kernel": "softmax",
        },
    },
    {
        "id": "legacy_local_fallback",
        "override": {"local_attention": True, "local_scheme": None},
    },
    {
        "id": "explicit_local_scheme",
        "override": {
            "local_attention": False,
            "local_scheme": "g_l_sl_swa",
            "window_size": 32,
        },
    },
]


def _patch_dist_get_rank(monkeypatch):
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda: 0, raising=False
    )


def _make_args():
    return SimpleNamespace(
        num_labels=2,
        only_mlm_task=False,
        only_cls_task=False,
    )


def _build_case_config(case):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(case["override"])
    return cfg


def _build_layers_equivalent_config(old_cfg):
    cfg = copy.deepcopy(old_cfg)
    cfg["local_attention"] = False
    cfg["local_scheme"] = None

    pos_emb_type = old_cfg["pos_emb_type"]
    relpe_type = old_cfg.get("relpe_type")

    def make_layer(name, code):
        layer = {
            "layer_name": name,
            "layer_type": "transformer",
            "local_scheme": code,
            "pos_emb_type": pos_emb_type,
        }
        if relpe_type is not None:
            layer["relpe_type"] = relpe_type
        return layer

    if old_cfg.get("local_scheme"):
        codes = old_cfg["local_scheme"].split("_")
        unique_codes = []
        for code in codes:
            if code not in unique_codes:
                unique_codes.append(code)
        code_to_name = {code: f"layer{code}" for code in unique_codes}
        cfg["layers"] = [
            make_layer(code_to_name[code], code)
            for code in unique_codes
        ]
        cfg["layers_scheme"] = "_".join(code_to_name[code] for code in codes)
    elif old_cfg.get("local_attention"):
        cfg["layers"] = [
            make_layer("layerl", "l"),
            make_layer("layersl", "sl"),
            make_layer("layerg", "g"),
        ]
        cfg["layers_scheme"] = "layerl_layersl_layerg"
    else:
        cfg["layers"] = [make_layer("layerg", "g")]
        cfg["layers_scheme"] = "layerg"

    return cfg


def _instantiate_pretraining(module, cfg, args):
    kwargs = copy.deepcopy(cfg)
    config = module.TransformerConfig(**kwargs)
    return module.TransformerForPreTraining(config, args)


def _assert_state_dict_compatible(model, state_dict):
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing = sorted(model_keys - state_keys)
    unexpected = sorted(state_keys - model_keys)
    missing_suffix = " (truncated to first 10)" if len(missing) > 10 else ""
    unexpected_suffix = (
        " (truncated to first 10)" if len(unexpected) > 10 else ""
    )
    assert not missing and not unexpected, (
        f"State dict mismatch: "
        f"missing_count={len(missing)}{missing_suffix}, "
        f"first_10_missing={missing[:10]}; "
        f"unexpected_count={len(unexpected)}{unexpected_suffix}, "
        f"first_10_unexpected={unexpected[:10]}"
    )
    model.load_state_dict(state_dict, strict=True)


def _fixed_batch(config, seq_len=16, batch_size=2):
    vocab_size = config["vocab_size_or_config_json_file"]
    input_ids = torch.arange(
        batch_size * seq_len, dtype=torch.long
    ).view(batch_size, seq_len) % vocab_size
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    if seq_len > 1:
        for i in range(batch_size):
            num_pad = min(i, seq_len - 1)
            if num_pad > 0:
                attention_mask[i, -num_pad:] = 0
    base_token_type = torch.arange(seq_len, dtype=torch.long) % 2
    token_type_ids = torch.stack(
        [base_token_type.roll(shifts=i) for i in range(batch_size)],
        dim=0,
    )
    masked_lm_labels = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    positions = [1, (seq_len // 3) % seq_len, ((2 * seq_len) // 3) % seq_len]
    for i in range(batch_size):
        pos1 = positions[i % len(positions)]
        pos2 = positions[(i + 1) % len(positions)]
        masked_lm_labels[i, pos1] = (7 + 3 * i) % vocab_size
        masked_lm_labels[i, pos2] = (11 + 5 * i) % vocab_size
    label = torch.arange(batch_size, dtype=torch.long) % 2
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "masked_lm_labels": masked_lm_labels,
        "label": label,
    }


def _forward_logits(model, batch):
    model.eval()
    with torch.no_grad():
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )


def _forward_loss(model, batch):
    model.train()
    with torch.no_grad():
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            masked_lm_labels=batch["masked_lm_labels"],
            label=batch["label"],
        )


def _assert_exact_tensor_eq(a, b, msg):
    if not torch.equal(a, b):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}; max_abs_diff={max_diff}")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_transformers_old_cfg_in_legacy_vs_new_modeling_outputs(case, monkeypatch):
    _patch_dist_get_rank(monkeypatch)
    args = _make_args()
    cfg = _build_case_config(case)
    batch = _fixed_batch(cfg)

    torch.manual_seed(2026)
    legacy_ref = _instantiate_pretraining(modeling_legacy, cfg, args)
    state_dict = {
        k: v.detach().clone() for k, v in legacy_ref.state_dict().items()
    }

    legacy_model = _instantiate_pretraining(modeling_legacy, cfg, args)
    new_model = _instantiate_pretraining(modeling_new, cfg, args)
    _assert_state_dict_compatible(legacy_model, state_dict)
    _assert_state_dict_compatible(new_model, state_dict)

    legacy_logits = _forward_logits(legacy_model, batch)
    new_logits = _forward_logits(new_model, batch)
    _assert_exact_tensor_eq(
        legacy_logits[0],
        new_logits[0],
        f"{case['id']} MLM logits mismatch",
    )
    _assert_exact_tensor_eq(
        legacy_logits[1],
        new_logits[1],
        f"{case['id']} CLS logits mismatch",
    )

    legacy_loss = _forward_loss(legacy_model, batch)
    new_loss = _forward_loss(new_model, batch)
    _assert_exact_tensor_eq(
        legacy_loss, new_loss, f"{case['id']} loss mismatch"
    )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_transformers_old_vs_new_cfg_in_new_modeling_outputs(case, monkeypatch):
    _patch_dist_get_rank(monkeypatch)
    args = _make_args()
    old_cfg = _build_case_config(case)
    layers_cfg = _build_layers_equivalent_config(old_cfg)
    batch = _fixed_batch(old_cfg)

    torch.manual_seed(2026)
    old_ref = _instantiate_pretraining(modeling_new, old_cfg, args)
    state_dict = {
        k: v.detach().clone() for k, v in old_ref.state_dict().items()
    }

    old_model = _instantiate_pretraining(modeling_new, old_cfg, args)
    layers_model = _instantiate_pretraining(modeling_new, layers_cfg, args)
    _assert_state_dict_compatible(old_model, state_dict)
    _assert_state_dict_compatible(layers_model, state_dict)

    old_logits = _forward_logits(old_model, batch)
    layers_logits = _forward_logits(layers_model, batch)
    _assert_exact_tensor_eq(
        old_logits[0],
        layers_logits[0],
        f"{case['id']} old-vs-layers MLM logits mismatch",
    )
    _assert_exact_tensor_eq(
        old_logits[1],
        layers_logits[1],
        f"{case['id']} old-vs-layers CLS logits mismatch",
    )

    old_loss = _forward_loss(old_model, batch)
    layers_loss = _forward_loss(layers_model, batch)
    _assert_exact_tensor_eq(
        old_loss, layers_loss, f"{case['id']} old-vs-layers loss mismatch"
    )
