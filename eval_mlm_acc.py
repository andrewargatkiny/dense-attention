"""Intended to run on single GPU."""

import glob
import json
import os
import argparse
import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers import BertForPreTraining, BertConfig as HFBertConfig

from dataset import BertPretrainingDataset
from src.modeling import BertForPreTrainingNewAttention
from src.expanded_ffn import ExpandedFFN
from src.model_config import ModelConfig
from dataset_utils import create_dataloader


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Script for evaluation loss and Masked Language Modeling "
                    "accuracy on a validation dataset"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="path to the model's checkpoint"
    )
    parser.add_argument(
        "--config-file",
        "--cf",
        help="path to the configuration file of the model",
        type=str,
        required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path to the eval dataset"
    )
    parser.add_argument(
        '--no_eval_custom',
        action='store_true',
        default=False,
        help="Don't evaluate DenseAttention model")
    parser.add_argument(
        '--no_eval_hf',
        action='store_true',
        default=False,
        help="Don't evaluate BERT version from HuggingFace")
    parser.add_argument(
        '--use_torch_compile',
        action='store_true',
        default=False,
        help="Use torch.compile() to compare models with static graphs")
    parser.add_argument(
        '--hf_use_flash_attn_2',
        action='store_true',
        default=False,
        help="Use FlashAttention-2 via PyTorch's scaled_dot_product_attention"
             "function for HF model. Note that it will use the first version"
             "of FlashAttention if PyTorch's version < 2.2.0")
    parser.add_argument(
        "--hf_model_checkpoint",
        type=str,
        default="google-bert/bert-large-uncased",
        help="Checkpoint from HuggingFace which should store original "
             "model's weights"
    )
    parser.add_argument(
        '--unpad_inputs',
        default=False,
        action='store_true',
        help='Whether to unpad inputs for efficient inference in case'
             ' of uneven seq lengths')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to run the model on"
    )
    parser.add_argument(
        "--fp_format",
        default="fp16",
        nargs="?",
        choices=["fp16", "bf16", "fp32"],
        help="Floating point number format for converting the model weights. "
             "Default: fp16."
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch size")
    parser.add_argument(
        "--max_seq_len_hf",
        default=512,
        type=int,
        help="Maximum length of incoming sequences that a HF model should "
             "handle.")
    """
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=2**16,
        type=int,
        help="The maximum number of masked tokens in a sequence to be "
             "predicted.")
    """
    parser.add_argument(
        "--num_workers_dataloader",
        default=4,
        type=int,
        help="Num parallel workers to prepare training samples in the "
             "dataloader.")
    parser.add_argument(
        "--scale_ffn_weights",
        default=True,
        action="store_true",
        help="Scale weights of FFN so their norm is <= than a predefined value. "
             "Use only in case the model was pretrained with FFN scaling"
    )
    parser.add_argument(
        "--no_scale_ffn_weights",
        action="store_false",
        dest="scale_ffn_weights"
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
        '--num_labels',
        type=int,
        default=2,
        help='Number of labels for classification tasks'
    )

    return parser


def load_dataset(args, file_path):
    data = BertPretrainingDataset("", dict(input_file=file_path), args)
    dataloader, num_samples = create_dataloader(
        data,
        num_workers=args.num_workers_dataloader,
        train_batch_size=args.batch_size,
        data_sampler=torch.utils.data.SequentialSampler(data),
    )
    return dataloader, num_samples


def prepare_hf_model(args):
    # Use pretrained model or initialize a random one if max seq
    # len is too large.
    config = HFBertConfig.from_pretrained(args.hf_model_checkpoint)
    if args.hf_use_flash_attn_2:
        config._attn_implementation = "sdpa"
    if args.max_seq_len_hf <= 512:
        model = BertForPreTraining.from_pretrained(args.hf_model_checkpoint,
                                                   config=config)
    else:
        config.max_position_embeddings = args.max_seq_len_hf
        model = BertForPreTraining(config)
    model = model.to(args.device)
    if args.fp_format == "fp16":
        model.half()
    elif args.fp_format == "bf16":
        model.bfloat16()
    elif args.fp_format == "fp32":
        model.float()
    model.requires_grad_(False)
    model.eval()
    if args.use_torch_compile:
        model = torch.compile(model)
    print(f"HuggingFace checkpoint is {args.hf_model_checkpoint}")
    print(f"HuggingFace model datatype is {model.dtype}")
    print("PyTorch version:", torch.__version__)
    if args.hf_use_flash_attn_2:
        print("Self-attention class implementation:",
              model.bert.encoder.layer[0].attention.self.__class__.__name__)
        print("For FlashAttention-2 PyTorch version should be>= 2.2.0 and "
              "attention class should be `BertSdpaSelfAttention`.")
        # Ensure that only FlashAttention will be used by disabling other backends.
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
    return model


def evaluate_bert_hugging_face(args, model, file_path):
    device = args.device
    print(f"Data file: {file_path}")
    dataloader, num_samples = load_dataset(args, file_path)
    mlm_loss_f = CrossEntropyLoss(ignore_index=-1)
    total_loss = 0
    n_correct = 0
    n_total = 0
    validation_start = time.time()
    for i, batch in enumerate(tqdm(dataloader, smoothing=1)):
        batch = {name: t.to(device) for name, t in batch.items()} # Move to GPU
        masked_lm_labels = batch["masked_lm_labels"]
        output = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       token_type_ids=batch["token_type_ids"],
                       return_dict=True)
        total_loss += mlm_loss_f(
            output.prediction_logits.flatten(end_dim=1),
            masked_lm_labels.view(-1)
        ).item()
        n_correct += (output.prediction_logits.argmax(dim=-1) == masked_lm_labels).sum().item()
        n_total += (masked_lm_labels > -1).sum().item()
        #tqdm.write(f"{total_loss}, {i}, {n_correct}, {n_total}")
    validation_stop = time.time()
    eval_duration = validation_stop - validation_start
    inference_speed = num_samples / eval_duration

    print(f"Loss over validation dataset is {total_loss / len(dataloader)}")
    print(f"Accuracy over validation dataset is {n_correct / n_total}")
    print(f"Inference speed is {inference_speed} samples per second")


def prepare_dense_attn_model(args):
    model_class = BertForPreTrainingNewAttention
    config_class = ModelConfig
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    bert_config = config_class(**config["model_config"])

    #tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    #bert_config.vocab_size = len(tokenizer.vocab)
    # Padding for divisibility by 8
    if bert_config.vocab_size % 8 != 0:
        bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
    print("VOCAB SIZE:", bert_config.vocab_size)
    args.config = bert_config

    # Initialize custom model
    model = model_class(bert_config, args)
    # load weights from DeepSpeed checkpoint.
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["module"], strict=False)
    del checkpoint
    model.requires_grad_(False)
    model = model.to(args.device)
    if args.fp_format == "fp16":
        model.half()
    elif args.fp_format == "bf16":
        model.bfloat16()
    elif args.fp_format == "fp32":
        model.float()
    # Fuse scaling coefficients for expanding and contracting weights in FFNs.
    if args.scale_ffn_weights:
        for name, module in model.named_modules():
            if isinstance(module, ExpandedFFN):
                module.prepare_for_inference()

    model.eval()
    if args.use_torch_compile:
        model = torch.compile(model)
    return model


def evaluate_dense_attention(args, model, file_path):
    device = args.device
    print(f"Data file: {file_path}")
    dataloader, num_samples = load_dataset(args, file_path)

    mlm_loss_f = CrossEntropyLoss(ignore_index=-1)
    total_loss = 0
    n_correct = 0
    n_total = 0
    validation_start = time.time()
    for i, batch in enumerate(tqdm(dataloader, smoothing=1)):
        batch = {name: t.to(device) for name, t in batch.items()} # Move to GPU
        """
        if model.config.use_local_attention and batch["input_ids"].shape[-1] < model.config.window_size:
            pad_size = (0, model.config.window_size - batch["input_ids"].shape[-1])
            batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], pad=pad_size,
                                          mode="constant", value=0)
            batch["token_type_ids"] = torch.nn.functional.pad(batch["token_type_ids"], pad=pad_size,
                                               mode="constant", value=1)
            batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], pad=pad_size,
                                               mode="constant", value=0)
            batch["masked_lm_labels"] = torch.nn.functional.pad(batch["masked_lm_labels"], pad=pad_size,
                                                     mode="constant", value=-1)
        """
        masked_lm_labels = batch["masked_lm_labels"]
        prediction_scores, seq_relationship_score = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )

        total_loss += mlm_loss_f(
            prediction_scores.flatten(end_dim=1),
            masked_lm_labels.view(-1)
        ).item()
        n_correct += (prediction_scores.argmax(dim=-1) == masked_lm_labels).sum().item()
        n_total += (masked_lm_labels > -1).sum().item()
        #tqdm.write(f"{total_loss}, {i}, {n_correct}, {n_total}")
    validation_stop = time.time()
    eval_duration = validation_stop - validation_start
    inference_speed = num_samples / eval_duration

    print(f"Loss over validation dataset is {total_loss / len(dataloader)}")
    print(f"Accuracy over validation dataset is {n_correct / n_total}")
    print(f"Inference speed is {inference_speed} samples per second")


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    if not args.no_eval_custom:
        print("Evaluating DenseAttention model")
        model = prepare_dense_attn_model(args)
        for file_path in glob.glob(os.path.join(args.data_path, "*.hdf5")):
            evaluate_dense_attention(args, model, file_path)
    if not args.no_eval_hf:
        print("Evaluating BERT model from HuggingFace checkpoint")
        model = prepare_hf_model(args)
        for file_path in glob.glob(os.path.join(args.data_path, "*.hdf5")):
            evaluate_bert_hugging_face(args, model, file_path)


if __name__ == "__main__":
    main()
