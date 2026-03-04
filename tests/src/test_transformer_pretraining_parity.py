import copy
from types import SimpleNamespace

import pytest
import torch

from src.other_models import modeling as modeling_new
from src.other_models import modeling_legacy


BASE_CONFIG = {
    "vocab_size": 128,
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
    {"id": "linear_global", "override": {"attention_kernel": "linear", "feature_map": "identity"}},
    {"id": "power_global", "override": {"attention_kernel": "power", "power": 2}},
    {
        "id": "rope_global",
        "override": {"pos_emb_type": "relpe", "relpe_type": "rope", "attention_kernel": "softmax"},
    },
    {"id": "legacy_local_fallback", "override": {"local_attention": True, "local_scheme": None}},
    {"id": "explicit_local_scheme", "override": {"local_attention": False, "local_scheme": "g_l_sl_swa"}},
]


def _patch_dist_get_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)


def _make_args():
    return SimpleNamespace(num_labels=2, only_mlm_task=False, only_cls_task=False)


def _build_case_config(case):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(case["override"])
    return cfg

#TODO: Do we really need this function? In all cases above pos_emb_type is string.
def _normalize_pos_emb_type(pos_emb_type):
    if isinstance(pos_emb_type, str):
        return pos_emb_type
    return pos_emb_type.name.lower()


def _build_layers_equivalent_config(old_cfg):
    cfg = copy.deepcopy(old_cfg)
    cfg["local_attention"] = False
    cfg["local_scheme"] = None

    pos_emb_type = _normalize_pos_emb_type(old_cfg["pos_emb_type"])
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
        code_to_name = {code: f"layer_{code}" for code in unique_codes}
        cfg["layers"] = [make_layer(code_to_name[code], code) for code in unique_codes]
        cfg["layers_scheme"] = "_".join(code_to_name[code] for code in codes)
    elif old_cfg.get("local_attention"):
        cfg["layers"] = [
            make_layer("layer_l", "l"),
            make_layer("layer_sl", "sl"),
            make_layer("layer_g", "g"),
        ]
        cfg["layers_scheme"] = "layer_l_layer_sl_layer_g"
    else:
        cfg["layers"] = [make_layer("layer_g", "g")]
        cfg["layers_scheme"] = "layer_g"

    return cfg


def _instantiate_pretraining(module, cfg, args):
    #TODO: few next lines could've been avoided if the config had "vocab_size_or_config_json_file"
    # as in real configs in `configs` dir instead of `vocab_size` parameter you used.
    kwargs = copy.deepcopy(cfg)
    vocab_size = kwargs.pop("vocab_size")
    config = module.TransformerConfig(
        vocab_size_or_config_json_file=vocab_size,
        **kwargs,
    )
    return module.TransformerForPreTraining(config, args)


def _assert_state_dict_compatible(model, state_dict):
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing = sorted(model_keys - state_keys)
    unexpected = sorted(state_keys - model_keys)
    # TODO: change the message to reflect that if we have more than 10 missing/ unexpected values, it can mislead in such cases.
    assert not missing and not unexpected, (
        f"State dict mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
    )
    model.load_state_dict(state_dict, strict=True)


def _fixed_batch(vocab_size, seq_len=16, batch_size=2):
    # TODO: batch size needs to fixed, because `attention_mask` is hardcoded to have 2 rows.
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).view(batch_size, seq_len) % vocab_size
    attention_mask = torch.tensor(
        [[1] * seq_len, [1] * (seq_len - 4) + [0] * 4], dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [[0] * (seq_len // 2) + [1] * (seq_len // 2), [1, 0] * (seq_len // 2)],
        dtype=torch.long,
    )
    masked_lm_labels = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    masked_lm_labels[0, 1] = 7
    masked_lm_labels[0, 5] = 11
    masked_lm_labels[1, 3] = 13
    label = torch.tensor([0, 1], dtype=torch.long)
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
def test_old_style_legacy_vs_new_logits_exact(case, monkeypatch):
    _patch_dist_get_rank(monkeypatch)
    args = _make_args()
    cfg = _build_case_config(case)
    batch = _fixed_batch(cfg["vocab_size"])

    torch.manual_seed(2026)
    legacy_ref = _instantiate_pretraining(modeling_legacy, cfg, args)
    state_dict = {k: v.detach().clone() for k, v in legacy_ref.state_dict().items()}

    legacy_model = _instantiate_pretraining(modeling_legacy, cfg, args)
    new_model = _instantiate_pretraining(modeling_new, cfg, args)
    _assert_state_dict_compatible(legacy_model, state_dict)
    _assert_state_dict_compatible(new_model, state_dict)

    legacy_logits = _forward_logits(legacy_model, batch)
    new_logits = _forward_logits(new_model, batch)
    _assert_exact_tensor_eq(legacy_logits[0], new_logits[0], f"{case['id']} MLM logits mismatch")
    _assert_exact_tensor_eq(legacy_logits[1], new_logits[1], f"{case['id']} CLS logits mismatch")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_old_style_legacy_vs_new_loss_exact(case, monkeypatch):
    _patch_dist_get_rank(monkeypatch)
    args = _make_args()
    cfg = _build_case_config(case)
    batch = _fixed_batch(cfg["vocab_size"])

    torch.manual_seed(2026)
    legacy_ref = _instantiate_pretraining(modeling_legacy, cfg, args)
    state_dict = {k: v.detach().clone() for k, v in legacy_ref.state_dict().items()}

    legacy_model = _instantiate_pretraining(modeling_legacy, cfg, args)
    new_model = _instantiate_pretraining(modeling_new, cfg, args)
    _assert_state_dict_compatible(legacy_model, state_dict)
    _assert_state_dict_compatible(new_model, state_dict)

    legacy_loss = _forward_loss(legacy_model, batch)
    new_loss = _forward_loss(new_model, batch)
    _assert_exact_tensor_eq(legacy_loss, new_loss, f"{case['id']} loss mismatch")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_old_style_vs_layers_logits_exact(case, monkeypatch):
    _patch_dist_get_rank(monkeypatch)
    args = _make_args()
    old_cfg = _build_case_config(case)
    layers_cfg = _build_layers_equivalent_config(old_cfg)
    batch = _fixed_batch(old_cfg["vocab_size"])

    torch.manual_seed(2026)
    old_ref = _instantiate_pretraining(modeling_new, old_cfg, args)
    state_dict = {k: v.detach().clone() for k, v in old_ref.state_dict().items()}

    old_model = _instantiate_pretraining(modeling_new, old_cfg, args)
    layers_model = _instantiate_pretraining(modeling_new, layers_cfg, args)
    _assert_state_dict_compatible(old_model, state_dict)
    _assert_state_dict_compatible(layers_model, state_dict)

    old_logits = _forward_logits(old_model, batch)
    layers_logits = _forward_logits(layers_model, batch)
    _assert_exact_tensor_eq(old_logits[0], layers_logits[0], f"{case['id']} old-vs-layers MLM logits mismatch")
    _assert_exact_tensor_eq(old_logits[1], layers_logits[1], f"{case['id']} old-vs-layers CLS logits mismatch")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_old_style_vs_layers_loss_exact(case, monkeypatch):
    _patch_dist_get_rank(monkeypatch)
    args = _make_args()
    old_cfg = _build_case_config(case)
    layers_cfg = _build_layers_equivalent_config(old_cfg)
    batch = _fixed_batch(old_cfg["vocab_size"])

    torch.manual_seed(2026)
    old_ref = _instantiate_pretraining(modeling_new, old_cfg, args)
    state_dict = {k: v.detach().clone() for k, v in old_ref.state_dict().items()}

    old_model = _instantiate_pretraining(modeling_new, old_cfg, args)
    layers_model = _instantiate_pretraining(modeling_new, layers_cfg, args)
    _assert_state_dict_compatible(old_model, state_dict)
    _assert_state_dict_compatible(layers_model, state_dict)

    old_loss = _forward_loss(old_model, batch)
    layers_loss = _forward_loss(layers_model, batch)
    _assert_exact_tensor_eq(old_loss, layers_loss, f"{case['id']} old-vs-layers loss mismatch")

