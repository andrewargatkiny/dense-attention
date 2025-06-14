import os
import sys
import argparse
import time
import math
import json

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import numpy as np
import wandb
from typing import Optional
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm

def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
           (global_steps >= args.max_steps)

def master_process(args):
    return (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1)

class TensorBoardWriter:
    """Replacement class for the case when ClearML logging is disabled"""
    SUMMARY_WRITER_DIR_NAME = 'runs'
    def __init__(self, name, base=".."):
        self.summary_writer = SummaryWriter(
        log_dir=os.path.join(base, self.SUMMARY_WRITER_DIR_NAME, name))

    def report_scalar(self, title, series, value, iteration):
        self.summary_writer.add_scalar(tag=series, scalar_value=value,
                                       global_step=iteration,
                                       display_name=title)

    def report_histogram(self, title, series, values, iteration, xlabels):
        self.summary_writer.add_histogram(tag=title, values=values,
                                          global_step=iteration, bins=xlabels)

class WandBWriter:        
    def __init__(self, 
                 name: str, 
                 args, 
                 base: str = "..",
                 ):
        deepspeed_config = json.load(
          open(args.deepspeed_config, 'r', encoding='utf-8'))
        self.run = wandb.init(
            project=name,
            name="research",
            config={
                          **vars(args),
                          "bert_config": args.config,
                          "deepspeed_config": deepspeed_config
                      }
        )
        
    def report_scalar(self, 
                     title: str, 
                     series: str, 
                     value: float, 
                     iteration: int) -> None:
        wandb.log({
            title+'/'+series: value,
        })
        
    def report_histogram(self,
                        title: str,
                        series: str,
                        values: list,
                        iteration: int,
                        xlabels: Optional[list] = None) -> None:
        xlabels = np.array(xlabels)
        bin_centers = 0.5 * (xlabels[1:] + xlabels[:-1])
        plt.bar(bin_centers, values, width=xlabels[1] - xlabels[0], alpha=0.7, color='blue', edgecolor='black')
        plt.title(title)
        wandb.log({title:wandb.Image(plt)})
        plt.close()
        

        



"""Evaluation routines"""

def eval_classification_task(data_batches, model,
                             max_validation_samples,
                             series_name, index, args):
    world_size = dist.get_world_size()
    eval_loss = 0
    nb_eval_steps = 0
    n_correct = 0
    n_total = 0
    validation_start = time.time()
    for batch in tqdm(data_batches):
        with torch.no_grad():
            batch = {name: t.to(args.device) for name, t in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["label"]

            tmp_eval_loss, prediction_scores = model(**batch)
            dist.reduce(tmp_eval_loss, op=dist.ReduceOp.SUM, dst=0)
            # Reduce to get the loss from all the GPU's
            tmp_eval_loss = tmp_eval_loss / world_size
            eval_loss += tmp_eval_loss.mean().item()
            tmp_n_correct = (prediction_scores.argmax(dim=-1) == labels).sum()
            dist.reduce(tmp_n_correct, op=dist.ReduceOp.SUM, dst=0)
            n_correct += tmp_n_correct.item()
            n_total += input_ids.shape[0] * world_size
            nb_eval_steps += 1
            if n_total >= max_validation_samples: break

    validation_stop = time.time()
    eval_duration = validation_stop - validation_start
    inference_speed = n_total / eval_duration

    eval_loss = eval_loss / nb_eval_steps
    eval_acc = n_correct / n_total
    args.logger.info(f"{series_name} Loss for epoch {index + 1} is: {eval_loss}")
    args.logger.info(f"{series_name} Accuracy for epoch {index + 1} is: {eval_acc}")
    args.logger.info(f"{series_name} inference speed for epoch "
                     f"{index + 1} is: {inference_speed}")

    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: loss', series=series_name,
                                          value=eval_loss, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: accuracy',
                                          series=series_name,
                                          value=eval_acc, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Inference speed',
                                          series=series_name,
                                          value=inference_speed, iteration=index + 1)


def eval_mlm_classification_task(data_batches, model,
                                 max_validation_samples,
                                 series_name, index, args):
    if args.only_cls_task:
        eval_classification_task(
            data_batches, model, max_validation_samples,
            series_name, index, args
        )
        return
    world_size = dist.get_world_size()
    nb_eval_steps = 0

    total_mlm_loss = 0
    total_cls_loss = 0

    n_correct_mlm = 0
    n_total_mlm = 0
    n_correct_cls = 0
    n_total_cls = 0
    validation_start = time.time()
    for batch in tqdm(data_batches):
        with torch.no_grad():
            batch = {name: t.to(args.device) for name, t in batch.items()}
            input_ids = batch["input_ids"]
            masked_lm_labels = batch["masked_lm_labels"]
            labels = batch.get("label", None)
            outputs = model(**batch)
            if args.only_mlm_task:
                (masked_lm_loss, target_labels,
                 prediction_scores) = outputs
            else:
                (masked_lm_loss, classification_loss, target_labels,
                 prediction_scores, seq_relationship_score) = outputs

            # Reduce to get the loss from all the GPUs
            dist.reduce(masked_lm_loss, op=dist.ReduceOp.SUM, dst=0)
            masked_lm_loss = masked_lm_loss / world_size
            total_mlm_loss += masked_lm_loss.mean().item()

            tmp_n_correct_mlm = (prediction_scores.argmax(dim=-1) == target_labels).sum()
            dist.reduce(tmp_n_correct_mlm, op=dist.ReduceOp.SUM, dst=0)
            tmp_n_total_mlm = (masked_lm_labels > -1).sum()
            dist.reduce(tmp_n_total_mlm, op=dist.ReduceOp.SUM, dst=0)
            n_correct_mlm += tmp_n_correct_mlm.item()
            n_total_mlm += tmp_n_total_mlm.item()

            nb_eval_steps += 1
            n_total_cls += input_ids.shape[0] * world_size
            if n_total_cls >= max_validation_samples: break
            # Handle the case where only mlm metrics are supplied
            if args.only_mlm_task: continue

            dist.reduce(classification_loss, op=dist.ReduceOp.SUM, dst=0)
            classification_loss = classification_loss / world_size
            total_cls_loss += classification_loss.mean().item()

            tmp_n_correct_cls = (seq_relationship_score.argmax(dim=-1) == labels).sum()
            dist.reduce(tmp_n_correct_cls, op=dist.ReduceOp.SUM, dst=0)
            n_correct_cls += tmp_n_correct_cls.item()

    validation_stop = time.time()
    eval_duration = validation_stop - validation_start
    inference_speed = n_total_cls / eval_duration

    eval_loss_mlm = total_mlm_loss / nb_eval_steps
    eval_acc_mlm = n_correct_mlm / n_total_mlm

    args.logger.info(f"{series_name} MLM loss for epoch "
                     f"{index + 1} is: {eval_loss_mlm}")
    args.logger.info(f"{series_name} MLM accuracy for epoch "
                     f"{index + 1} is: {eval_acc_mlm}")
    args.logger.info(f"{series_name} inference speed for epoch "
                     f"{index + 1} is: {inference_speed}")
    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: MLM loss', series=series_name,
                                          value=eval_loss_mlm, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: MLM accuracy',
                                          series=series_name,
                                          value=eval_acc_mlm, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Inference speed',
                                          series=series_name,
                                          value=inference_speed, iteration=index + 1)

    if args.only_mlm_task: return
    eval_loss_cls = total_cls_loss / nb_eval_steps
    eval_acc_cls = n_correct_cls / n_total_cls

    args.logger.info(f"{series_name} Classification loss for epoch "
                     f"{index + 1} is: {eval_loss_cls}")
    args.logger.info(f"{series_name} Classification accuracy for epoch "
                     f"{index + 1} is: {eval_acc_cls}")
    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: loss', series=series_name,
                                          value=eval_loss_cls, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: accuracy',
                                          series=series_name,
                                          value=eval_acc_cls, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: total loss',
                                          series=series_name,
                                          value=eval_loss_mlm + eval_loss_cls,
                                          iteration=index + 1)


# Assisted by GPT o3-mini
def eval_regression_task(data_batches, model, max_validation_samples,
                         series_name, index, args):
    world_size = dist.get_world_size()
    eval_loss = 0
    nb_eval_steps = 0
    n_total = 0
    # Lists to accumulate gathered predictions and ground truth values per iteration
    all_preds = []
    all_labels = []
    validation_start = time.time()

    for batch in tqdm(data_batches):
        with torch.no_grad():
            # Move batch to device
            batch = {name: t.to(args.device) for name, t in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["label"]

            # Forward pass: model returns (loss, logits)
            tmp_eval_loss, logits = model(**batch)
            # Reduce loss from all GPUs
            dist.reduce(tmp_eval_loss, op=dist.ReduceOp.SUM, dst=0)
            tmp_eval_loss = tmp_eval_loss / world_size
            eval_loss += tmp_eval_loss.mean().item()

            # Prepare local predictions and labels as 1D tensors
            local_preds = logits.view(-1).detach()
            local_labels = labels.view(-1).detach()

            # Gather predictions and labels across GPUs at every iteration
            if world_size > 1:
                preds_list = [torch.zeros_like(local_preds) for _ in range(world_size)]
                labels_list = [torch.zeros_like(local_labels) for _ in range(world_size)]
                dist.all_gather(preds_list, local_preds)
                dist.all_gather(labels_list, local_labels)
                gathered_preds = torch.cat(preds_list)
                gathered_labels = torch.cat(labels_list)
            else:
                gathered_preds = local_preds
                gathered_labels = local_labels

            # Append the gathered tensors (moved to CPU) to the global lists
            all_preds.append(gathered_preds.cpu())
            all_labels.append(gathered_labels.cpu())

            nb_eval_steps += 1
            n_total += input_ids.shape[0] * world_size
            if n_total >= max_validation_samples:
                break

    validation_stop = time.time()
    eval_duration = validation_stop - validation_start
    inference_speed = n_total / eval_duration

    # Concatenate all predictions and labels from all iterations
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Convert tensors to numpy arrays for scipy correlation functions
    preds_np = all_preds.float().numpy()
    labels_np = all_labels.float().numpy()

    # Compute Pearson and Spearman correlation coefficients
    pearson_corr, _ = pearsonr(preds_np, labels_np)
    spearman_corr, _ = spearmanr(preds_np, labels_np)

    # Average the loss over evaluation steps
    eval_loss = eval_loss / nb_eval_steps

    # Log metrics
    args.logger.info(f"{series_name} Loss for epoch {index + 1} is: {eval_loss}")
    args.logger.info(f"{series_name} Pearson correlation for epoch {index + 1} is: {pearson_corr}")
    args.logger.info(f"{series_name} Spearman correlation for epoch {index + 1} is: {spearman_corr}")
    args.logger.info(f"{series_name} Inference speed for epoch {index + 1} is: {inference_speed}")

    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: loss', series=series_name,
                                          value=eval_loss, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: Pearson correlation',
                                          series=series_name,
                                          value=pearson_corr, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: Spearman correlation',
                                          series=series_name,
                                          value=spearman_corr, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Inference speed',
                                          series=series_name,
                                          value=inference_speed, iteration=index + 1)



def eval_glue_tasks(data_batches, model,max_validation_samples,
                    series_name, index, args):
    """A version of `eval_classification_task` which additionally calculates F1
     and Matthews Correlation."""
    world_size = dist.get_world_size()
    eval_loss = 0
    nb_eval_steps = 0
    n_correct = 0
    n_total = 0

    # Counters for confusion matrix components (for binary classification)
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    validation_start = time.time()
    for batch in tqdm(data_batches):
        with torch.no_grad():
            # Move each tensor to the designated device
            batch = {name: t.to(args.device) for name, t in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["label"]

            tmp_eval_loss, prediction_scores = model(**batch)
            # Reduce the loss from all GPUs
            dist.reduce(tmp_eval_loss, op=dist.ReduceOp.SUM, dst=0)
            tmp_eval_loss = tmp_eval_loss / world_size
            eval_loss += tmp_eval_loss.mean().item()

            # Compute predictions and reduce the correct count from all GPUs
            preds = prediction_scores.argmax(dim=-1)
            tmp_n_correct = (preds == labels).sum()
            dist.reduce(tmp_n_correct, op=dist.ReduceOp.SUM, dst=0)
            n_correct += tmp_n_correct.item()
            n_total += input_ids.shape[0] * world_size
            nb_eval_steps += 1

            # Compute batch-wise confusion matrix components
            # Assuming binary classification with labels 0 (negative) and 1 (positive)
            batch_tp = ((preds == 1) & (labels == 1)).sum()
            batch_tn = ((preds == 0) & (labels == 0)).sum()
            batch_fp = ((preds == 1) & (labels == 0)).sum()
            batch_fn = ((preds == 0) & (labels == 1)).sum()

            # Reduce each across all GPUs
            dist.reduce(batch_tp, op=dist.ReduceOp.SUM, dst=0)
            dist.reduce(batch_tn, op=dist.ReduceOp.SUM, dst=0)
            dist.reduce(batch_fp, op=dist.ReduceOp.SUM, dst=0)
            dist.reduce(batch_fn, op=dist.ReduceOp.SUM, dst=0)

            total_tp += batch_tp.item()
            total_tn += batch_tn.item()
            total_fp += batch_fp.item()
            total_fn += batch_fn.item()

            if n_total >= max_validation_samples:
                break

    validation_stop = time.time()
    eval_duration = validation_stop - validation_start
    inference_speed = n_total / eval_duration

    assert total_tp + total_fn + total_tn + total_fp == n_total, (
        "Confusion matrix total is not equal to num of samples."
    )
    eval_loss = eval_loss / nb_eval_steps
    eval_acc = n_correct / n_total

    # Compute F1 score and Matthews Correlation Coefficient "by hand"
    # F1 = 2 * TP / (2*TP + FP + FN)
    if (2 * total_tp + total_fp + total_fn) > 0:
        f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
    else:
        f1 = 0.0

    # MCC = (TP * TN - FP * FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    denom = math.sqrt((total_tp + total_fp) * (total_tp + total_fn) *
                      (total_tn + total_fp) * (total_tn + total_fn))
    if denom > 0:
        mcc = (total_tp * total_tn - total_fp * total_fn) / denom
    else:
        mcc = 0.0

    args.logger.info(f"{series_name} Loss for epoch {index + 1} is: {eval_loss}")
    args.logger.info(f"{series_name} Accuracy for epoch {index + 1} is: {eval_acc}")
    args.logger.info(f"{series_name} F1 Score for epoch {index + 1} is: {f1}")
    args.logger.info(f"{series_name} Matthews Correlation Coefficient for epoch {index + 1} is: {mcc}")
    args.logger.info(f"{series_name} inference speed for epoch {index + 1} is: {inference_speed}")

    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: loss', series=series_name,
                                          value=eval_loss, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: accuracy',
                                          series=series_name,
                                          value=eval_acc, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: F1 Score',
                                          series=series_name,
                                          value=f1, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: Matthews Corr',
                                          series=series_name,
                                          value=mcc, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Inference speed',
                                          series=series_name,
                                          value=inference_speed, iteration=index + 1)
