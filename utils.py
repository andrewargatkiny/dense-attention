import os
import sys
import argparse

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



"""Evaluation routines"""

def eval_classification_task(data_batches, model,
                             max_validation_samples,
                             series_name, index, args):
    world_size = dist.get_world_size()
    eval_loss = 0
    nb_eval_steps = 0
    n_correct = 0
    n_total = 0
    for batch in tqdm(data_batches):
        with torch.no_grad():
            batch = {name: t.to(args.device) for name, t in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["label"]

            tmp_eval_loss, prediction_scores = model(**batch)
            dist.reduce(tmp_eval_loss, op=dist.ReduceOp.SUM, dst=0)
            # Reduce to get the loss from all the GPU's
            tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
            eval_loss += tmp_eval_loss.mean().item()
            tmp_n_correct = (prediction_scores.argmax(dim=-1) == labels).sum()
            dist.reduce(tmp_n_correct, op=dist.ReduceOp.SUM, dst=0)
            n_correct += tmp_n_correct.item()
            n_total += input_ids.shape[0] * world_size
            nb_eval_steps += 1
            if n_total >= max_validation_samples: break
    eval_loss = eval_loss / nb_eval_steps
    eval_acc = n_correct / n_total
    args.logger.info(f"{series_name} Loss for epoch {index + 1} is: {eval_loss}")
    args.logger.info(f"{series_name} Accuracy for epoch {index + 1} is: {eval_acc}")
    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: loss', series=series_name,
                                          value=eval_loss, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: accuracy',
                                          series=series_name,
                                          value=eval_acc, iteration=index + 1)

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

    for batch in tqdm(data_batches):
        with torch.no_grad():
            batch = {name: t.to(args.device) for name, t in batch.items()}
            input_ids = batch["input_ids"]
            masked_lm_labels = batch["masked_lm_labels"]
            labels = batch["label"]
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


    eval_loss_mlm = total_mlm_loss / nb_eval_steps
    eval_acc_mlm = n_correct_mlm / n_total_mlm

    args.logger.info(f"{series_name} MLM loss for epoch "
                     f"{index + 1} is: {eval_loss_mlm}")
    args.logger.info(f"{series_name} MLM accuracy for epoch "
                     f"{index + 1} is: {eval_acc_mlm}")
    if master_process(args):
        args.tracker_logger.report_scalar(title='Epochs: MLM loss', series=series_name,
                                          value=eval_loss_mlm, iteration=index + 1)
        args.tracker_logger.report_scalar(title='Epochs: MLM accuracy',
                                          series=series_name,
                                          value=eval_acc_mlm, iteration=index + 1)

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
