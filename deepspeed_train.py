import os
import sys
import time
import logging

import numpy as np
import random
import json
import re
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.model_config import ModelConfig
from utils.tasks import TaskRegistry
from train_arguments import get_argument_parser
from utils.logger import Logger
from utils.optimization import warmup_exp_decay_exp, cosine_poly_warmup_decay
from train_utils import is_time_to_exit, master_process, TensorBoardWriter

from data.dataset_utils import ShardedDatasetWrapper, create_dataloader

from clearml import Task
import deepspeed

global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0
all_step_time = 0.0


def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step,
                     last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        'epoch': epoch,
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(PATH, ckpt_id,
                                            checkpoint_state_dict)
    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(args, model, PATH, ckpt_id):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    logger = args.logger
    # Workaround for learned positional embeddings if checkpoint and current
    # model's max_seq_lengths mismatch.
    if args.resize_posit_embeds:
        if not args.load_only_weights:
            raise ValueError("If you want to resize positional embeddings "
                             "when loading from checkpoint, you have to set "
                             "args.load_only_weights=True.")
        custom_load_fn = model.resize_learned_pos_embeddings
    else:
        custom_load_fn = None

    _, checkpoint_state_dict = model.load_checkpoint(
        PATH, ckpt_id,
        load_module_strict=False,
        load_module_only=args.load_only_weights,
        custom_load_fn=custom_load_fn
    )

    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict[
        'last_global_data_samples']
    del checkpoint_state_dict
    if args.load_only_weights:
        epoch = 0
        last_global_step = 0
        last_global_data_samples = 0
    return (epoch, last_global_step, last_global_data_samples)


def get_dataloader(args, dataset: Dataset):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in
            DataLoader(dataset,
                       batch_size=args.eval_bs,
                       sampler=train_sampler,
                       num_workers=args.config['training']['num_workers'],
                       drop_last=args.eval_only))


def pretrain_validation(args, dataset, series_name, index, model):

    config = args.config
    num_layers = config["model_config"]["num_hidden_layers"]
    logger = args.logger
    eval_bs = args.train_micro_batch_size_per_gpu * args.eval_bs_ratio
    max_validation_samples = args.max_validation_samples
    if max_validation_samples == -1:
        max_validation_samples= len(dataset)
        logger.info(f"length of dataset is {len(dataset)}")
    args.eval_bs = eval_bs
    logger.info(
        f"Validation micro batch size: {eval_bs}")
    if args.dense_attention:
        update_weights_scalers(model, num_layers)
    model.eval()

    data_batches = get_dataloader(args, dataset)
    eval_func = args.task.eval_func
    eval_func(data_batches, model, max_validation_samples,
              series_name, index, args)

def train(args,
          index,
          model,
          optimizer,
          pretrain_dataset_provider,
          finetune=False):
    global global_step
    global global_data_samples
    global last_global_step_from_restore
    global all_step_time

    if args.use_sharded_dataset:
        # print dataset files according to their order.
        pretrain_dataset_provider.dataset_order_info()
        dataset = pretrain_dataset_provider.get_shard(index)
        train_sampler = RandomSampler(dataset)
        worker_init = pretrain_dataset_provider.worker_init
    else:
        dataset = pretrain_dataset_provider
        worker_init = None
        if args.local_rank == -1:
            train_sampler = RandomSampler(pretrain_dataset_provider)
        else:
            train_sampler = DistributedSampler(pretrain_dataset_provider)
            train_sampler.set_epoch(index + 1)

    dataset_iterator, total_length = create_dataloader(
        train_data=dataset,
        num_workers=args.config['training']['num_workers'],
        train_batch_size=args.train_micro_batch_size_per_gpu,
        data_sampler=train_sampler, worker_init=worker_init)

    current_data_sample_count = global_data_samples
    rank = dist.get_rank()
    num_layers = args.config["model_config"]["num_hidden_layers"]

    config = args.config
    logger = args.logger
    logger.info(
        f'worker-{dist.get_rank()}: begin epoch {index+1} '
        f'current_sample_count {current_data_sample_count} '
        f'shard_length {total_length} global_data_samples {global_data_samples}'
    )

    model.train()

    epoch_step = 0
    rounds = args.throughput_logging_freq
    step_counts = 0
    lr_this_step = config["training"]["learning_rate"]
    inner_optimizer = optimizer if not args.bf16 else optimizer.optimizer
    if args.dense_attention:
        update_weights_scalers(model, num_layers)

    for group in optimizer.param_groups:
        group['lr'] = lr_this_step
        if group['name'] != 'others_with_no_wd': group['weight_decay'] = args.config["training"]["weight_decay"]

    for _, batch in enumerate(tqdm(dataset_iterator, smoothing=1)):
        try:
            step_start = time.time()
            #batch = pretrain_dataset_provider.get_batch(batch_index)
            batch = {name: t.to(args.device) for name, t in batch.items()}  # Move to GPU
            # Calculate forward pass
            loss = model(**batch)

            unscaled_loss = loss.item()
            #print(f"loss {loss}, rank {rank}")

            current_data_sample_count += (args.train_micro_batch_size_per_gpu *
                                          dist.get_world_size())

            # Prefetch training data
            #pretrain_dataset_provider.prefetch_batch()

            #if not np.isfinite(unscaled_loss): continue
            model.backward(loss)
            '''
            if not np.isfinite(unscaled_loss):
                report_model_activations(args, model,
                                         batch, global_step)
                if args.log_activations:
                    report_model_weights(args, model, global_step)
            '''

            loss = None
            del loss
            #for name, param in model.named_parameters():
            #    if param.grad is not None: print(f"Grad Extremums", name, param.grad.min(), param.grad.max())

            if model.is_gradient_accumulation_boundary():

                #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.**15)
                #for name, param in model.named_parameters():
                #    if param.grad is not None: print(f"Grad Extremums", name, param.grad.min(), param.grad.max())
                lr_this_step = update_learning_rate(
                    args, config, global_step, inner_optimizer)
                #"""
                if (global_step + 1) % args.throughput_logging_freq == 0 and master_process(args):
                    report_step_metrics(args, lr_this_step, unscaled_loss,
                                        global_step, current_data_sample_count)
                # if epoch_step % 16 == 0:
                    # For multi-node
                    # if dist.get_world_size() > 1:
                    #    dist.broadcast_object_list(list(model.parameters()), src=0)
                    # For fp16 training
                    # refresh_fp32_params(optimizer)
                    # For bf16 training
                    # optimizer._restore_from_bit16_weights()
                if epoch_step == 0 and index % args.log_diagnostic_freq == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None: print(f"Grad Extremums", name, param.grad.min(), param.grad.max())
                    for tensor, state in inner_optimizer.state.items():
                        print("step", inner_optimizer.state[tensor]['step'])
                        print("exp_avg", inner_optimizer.state[tensor]['exp_avg'])
                        print("exp_avg_sq", inner_optimizer.state[tensor]['exp_avg_sq'])
                    logger.info(
                        f"Logging model weights and activations distribution "
                        f"at the start of of epoch: {index}, step: {epoch_step}")
                    if args.log_activations:
                        report_model_activations(args, model,
                                                 batch, global_step)
                    report_model_weights(args, model, global_step)
                # for name, param in model.named_parameters():
                #     if param.grad is not None: print(f"Parameter Extremums", name, param.grad.min(), param.grad.max())
                model.step()
                if args.dense_attention:
                    update_weights_scalers(model, num_layers)
                #print(f"Finished optimization step on {rank}")
                """
                try:
                    print(f"Started optimization step on {rank}")
                    for name, param in model.named_parameters():
                        if param.grad is not None: print(f"Parameter Extremums", name, param.grad.min(), param.grad.max())
                        #if param.grad is not None: print(f"Parameter Grad", name, param.grad)
                    model.step()
                    print(f"Finished optimization step on {rank}")
                except AssertionError as ex:
                    print("Error", ex)
                    for name, param in model.named_parameters():
                        if param.grad is not None: print(f"Parameter Extremums", name, param.grad.min(), param.grad.max())
                        print(f"Parameter Grad", name, param.grad)
                    report_model_activations(args, model,
                                             batch, global_step)
                    report_model_weights(args, model, global_step)
                    import traceback
                    traceback.print_exc()
                """

                report_lamb_coefficients(args, optimizer)
                global_step += 1
                epoch_step += 1
            else:
                # Call DeepSpeed engine step on micro steps
                model.step()

        except StopIteration:
            continue

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args,
                           epoch_steps=epoch_step,
                           global_steps=current_global_step):
            print(
                f'Warning: Early epoch termination due to max steps limit, epoch step ={epoch_step}, global step = {current_global_step}, epoch = {index+1}'
            )
            break
        step_time = time.time() - step_start
        all_step_time += step_time
        if global_step % rounds == 0 and global_step != 0 and model.is_gradient_accumulation_boundary(
        ) and dist.get_rank() == 0:
            one_step_bs = args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps * dist.get_world_size(
            ) * rounds
            print(' At step {}, the throughput is {:2f} Samples/s'.format(
                global_step * args.gradient_accumulation_steps,
                one_step_bs / all_step_time),
                  flush=True)
            all_step_time = 0.0

    #pretrain_dataset_provider.release_shard(index)

    global_data_samples = current_data_sample_count


def update_weights_scalers(model, num_layers):
    """Update weights scalers of DenseAttention Model"""
    for i in range(num_layers):
        ffn = model.bert.encoder.layer[i].ffn
        ffn.adjust_norm_ratios()


def update_learning_rate(args, config, current_global_step, optimizer):
    global last_global_step_from_restore

    global_step_for_lr = current_global_step - last_global_step_from_restore
    lr_schedule = config["training"]["lr_schedule"]
    if lr_schedule == "EE":
        #print(f'LR Schedule is {args.lr_schedule} EE')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["one_cycle_steps"],
                config["training"]["warmup_proportion"])
    elif lr_schedule == "cosine":
        #print(f'LR Schedule is {args.lr_schedule} EP')
        lr_this_step = config["training"][
            "learning_rate"] * cosine_poly_warmup_decay(
                global_step_for_lr, **config["training"]["lr_scheduler_params"]
        )
    elif lr_schedule == 'constant':
        lr_this_step = config["training"]["learning_rate"]
    else:
        lr_this_step = config["training"]["learning_rate"]

    lr_this_step += config["training"]["lr_offset"]

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    return lr_this_step


def report_step_metrics(args, lr, loss, step, data_sample_count):
    ##### Record the LR against global_step on tensorboard #####

    args.tracker_logger.report_scalar(title='Train Steps: lr', series='lr',
                                      value=lr, iteration=step)
    args.tracker_logger.report_scalar(title='Train Steps: loss',
                                      series='loss',
                                      value=loss, iteration=step)
    args.tracker_logger.report_scalar(title='Train Samples: lr',
                                      series='lr',
                                      value=lr, iteration=data_sample_count)
    args.tracker_logger.report_scalar(title='Train Samples: loss',
                                      series='loss',
                                      value=loss, iteration=data_sample_count)
    ##### Recording  done. #####

    print('bing_bert_progress: step={}, loss={}, lr={}, sample_count={}'.
          format(step + 1, loss, lr, data_sample_count))


def find_layer_with_nans(args, model, data):
    none_layers = set([args.config["model_config"]["num_hidden_layers"]])
    def getActivation(layer_number):
        # the hook signature
        def hook(model, input, output):
            if not torch.all(torch.isfinite(output)):
                none_layers.add(layer_number)
                print(f"Nans discovered in layer {layer_number}")
        return hook

    hooks = []
    for name, module in model.named_modules():
        if "encoder.layer." in name:
            number = int(name.split("encoder.layer.")[1].split(".")[0])
            hooks.append(
                module.register_forward_hook(getActivation(number))
            )
    # Calculate all activations
    with torch.no_grad():
        dummy_loss = model(**data)
    for hook in hooks:
        hook.remove()

    return min(none_layers)


def report_model_gradients(args, model):
    if master_process(args):
        def getActivation(name):
            # the hook signature
            def hook(module, grad_input, grad_output):
                grad_input = grad_input[0]
                grad_output = grad_output[0]
                if grad_output is not None:
                    print(name, "grad output:", grad_output.min(), grad_output.max())
                if grad_input is not None:
                    print(name, "grad input:", grad_input.min(), grad_input.max())
            return hook
        hooks = []
        for name, module in model.named_modules():
            hooks.append(
                module.register_full_backward_hook(getActivation(name))
            )
        return hooks

def report_activations_fast(args, model):
    if master_process(args):
        def getActivation(name):
            # the hook signature
            def hook(module, input, output):
                if output is not None and not isinstance(output, (list, tuple)):
                    print(name, "layer output:", output.min(), output.max())
                if input is not None and not isinstance(input, (list, tuple)):
                    print(name, "layer input:", input.min(), input.max())
            return hook

        hooks = []
        for name, module in model.named_modules():
            hooks.append(
                module.register_forward_hook(getActivation(name))
            )
        return hooks

# TODO: handle cases when there are np.inf in reporting functions

def report_model_activations(args, model, data, step, bins=20, **kwargs):
    if master_process(args):
        args.logger.info(f"Starting to report activation for step {step}")
        activations = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                if not isinstance(output, (list, tuple)):
                    activations[name] = output.cpu().float().numpy()
            return hook
        hooks = []
        if args.use_torch_compile:
            model = model._orig_mod
        for name, module in model.named_modules():
            hooks.append(
                module.register_forward_hook(getActivation(name))
            )
        # Calculate all activations
        last_sample = int(data["input_ids"].shape[0] * args.inputs_logging_ratio)
        data = {name: t[:last_sample] for name, t in data.items()}
        with torch.no_grad():
            if kwargs.get("reg_losses", None) is not None:
                dummy_loss = model(**data, **kwargs)
            else:
                dummy_loss = model(**data)
        for hook in hooks:
            hook.remove()
        for name, values in activations.items():
            finite_values = values[np.isfinite(values)]
            # finite_values is already 1D. 
            if values.ravel().shape != finite_values.shape:
                args.tracker_logger.report_scalar(
                    title=f'nans and infs: {name}', series='n of nans',
                    value=len(values[np.isnan(values)].ravel()), iteration=step
                )
                args.tracker_logger.report_scalar(
                    title=f'nans and infs: {name}', series='n of +infs',
                    value=len(values[np.isposinf(values)].ravel()), iteration=step
                )
                args.tracker_logger.report_scalar(
                    title=f'nans and infs: {name}', series='n of -infs',
                    value=len(values[np.isneginf(values)].ravel()), iteration=step
                )

            if finite_values.size == 0: return
            try:
                vals = values#.mean(axis=-1)
                val_max, val_min = vals.max(), vals.min()
                if args.no_clearml:
                    hist, bounds = values, bins
                else:
                    hist, bounds = np.histogram(vals, bins=bins,
                                                range=(val_min, val_max))
                    bounds = list(bounds)
                args.tracker_logger.report_histogram(
                    title=name, series=name, values=hist, iteration=step,
                    xlabels=bounds
                )
            except Exception as ex:
                print(ex)


def report_model_weights(args, model, step, bins=20):
    if master_process(args):
        norm_types = {"L1": 1, "L2": 2, "Linf": float('inf')}
        if args.log_weight_norms and not args.logging_norm_type in norm_types:
            raise ValueError(f"{args.logging_norm_type} is an invalid option \
                             for the args.logging_norm_type argument. \
                             Available options are: {', '.join(norm_types)}.")

        def get_group(name):
            """
            Returns the group name under which the specified model parameter's vector norm 
            should be logged and the layer's identifier in the group.
            """

            # "module.bert.encoder.layer.0.attention.queries" -> ".layer.0."
            layer = re.search(r'\.layer\.\d+\.', name) 
            if layer:
                layer = layer.group(0)
                group_name = name.replace(layer, '.', 1)
                layer_number = re.search(r'\d+', layer).group(0)
                return group_name, f"Layer {layer_number}"

            return ("Embeddings parameters", name) if re.search(r'embed', name) else ("Other parameters", name)

        for name, param in model.named_parameters():
            p = param.detach().cpu().float()
            if args.log_weight_norms: 
                norm = torch.norm(p, p=norm_types[args.logging_norm_type]).item()
                group_name, identifier = get_group(name)
                args.tracker_logger.report_scalar(
                    title=f'{args.logging_norm_type} Norm/ {group_name}',
                    series=identifier,
                    value=norm,
                    iteration=step
                )
            values = p.numpy()
            if args.no_clearml:
                hist, bounds = values, bins
            else:
                hist, bounds = np.histogram(values, bins=bins, range=(np.nanmin(values), np.nanmax(values)))
                bounds = list(bounds)
            args.tracker_logger.report_histogram(
                title=name, series=name, values=hist, iteration=step, 
                xlabels=bounds
            )

def report_lamb_coefficients(args, optimizer):
    if master_process(args):
        if (args.fp16 and args.use_lamb):
            #print("Lamb Coeffs", optimizer.optimizer.get_lamb_coeffs())
            lamb_coeffs = optimizer.optimizer.get_lamb_coeffs()
            lamb_coeffs = np.array(lamb_coeffs)
            #if lamb_coeffs.size > 0:
            #    args.summary_writer.add_histogram(f'Train/lamb_coeffs',
            #                                      lamb_coeffs, global_step)


# Refresh fp32 master params from fp16 copies
def refresh_fp32_params(optimizer):
    # Flat fp16 track originally shaped fp16 groups which in turn track
    # real model weights in DeepSpeed fp16 Optimizer.
    for fp32, saved_fp16 in zip(optimizer.fp32_groups_flat, optimizer.fp16_groups_flat):
        fp32.data.copy_(saved_fp16.data)


def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    args.config = config
    if args.model_config_file and args.model_config_file != args.config_file:
        model_config = json.load(
            open(args.model_config_file, 'r', encoding='utf-8')
        )
        args.config["model_config"] = model_config["model_config"]
    if args.data_config_file and args.data_config_file != args.config_file:
        data_config = json.load(
            open(args.data_config_file, 'r', encoding='utf-8')
        )
        args.config["data"] = data_config["data"]
    if args.train_config_file and args.train_config_file != args.config_file:
        train_config = json.load(
            open(args.train_config_file, 'r', encoding='utf-8')
        )
        args.config["training"] = train_config["training"]
    args.task = TaskRegistry.get_task(args.task_type)

    args.job_name = config['name'] if args.job_name is None else args.job_name
    print("Running Config File: ", args.job_name)
    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                         args.job_name)

    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning(
            'Early training exit is set after {} global steps'.format(
                args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(
            args.max_steps_per_epoch))

    return args


def prepare_optimizer_parameters(args, model):
    config = args.config
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))
    params_to_optimize = list(model.named_parameters())
    #params_to_optimize = [n for n in params_to_optimize if #'pooler' not in n[0] and
    #                   'embeddings' not in n[0] and 'layer' not in n[0]]
    no_decay_list = ['bias', 'LayerNorm.bias', 'LayerNorm.weight',
                     'activation.weight', 'layer_norm.weight']
    if args.no_decay_embeddings:
        no_decay_list += ['embeddings']
    if args.no_decay_pooler:
        no_decay_list += ['pooler']
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01


    groups = [{'params': list(model.bert.embeddings.parameters()),
               'lr': 0.0,
               'weight_decay': weight_decay,
               'name': 'embeddings'}]
    for i in range(len(model.bert.encoder.layer)):
        if args.dense_attention:
            # If some kind of layer norm in attention layer has learnable
            # params, they wouldn't be updated.
            groups.append({
                'params': list(model.bert.encoder.layer[i].attention.parameters()),
                'lr': 0.0,
                'weight_decay': weight_decay,
                'name': f'layer_{i}_attention'
            })
            if hasattr(model.bert.encoder.layer[i], 'ffn'):
                groups.append({
                    'params': list(model.bert.encoder.layer[i].ffn.parameters()),
                    'lr': 0.0,
                    'weight_decay': weight_decay,
                    'name': f'layer_{i}_ffn'
                })
        else:
            groups.append({
                'params': list(model.bert.encoder.layer[i].parameters()),
                'lr': 0.0,
                'weight_decay': weight_decay,
                'name': f'layer_{i}_attention'
            })


    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in params_to_optimize
            if not any(stop_word in n for stop_word in no_decay_list)
        ],
        'lr': 0.0,
        'weight_decay': weight_decay,
        'name': 'others_with_wd'
    }, {
        'params':
        [p for n, p in params_to_optimize
         if any(stop_word in n for stop_word in no_decay_list)],
        'lr': 0.0,
        'weight_decay': 0.0,
        'name': 'others_with_no_wd'
    }]
    #optimizer_grouped_parameters.extend(groups)

    return optimizer_grouped_parameters


def prepare_model_optimizer(args):
    # Initialize torch distributed
    deepspeed.init_distributed(dist_backend=args.dict_backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    model_class = args.task.model_type
    config_class = ModelConfig
    if hasattr(args.task, "config_type"):
        config_class = args.task.config_type


    bert_config = config_class(**args.config["model_config"])
    # Padding for divisibility by 8
    if bert_config.vocab_size % 8 != 0:
        bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
    print("VOCAB SIZE:", bert_config.vocab_size)

    model = model_class(bert_config, args)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters)

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.train_micro_batch_size_per_gpu(
    )
    args.gradient_accumulation_steps = model.gradient_accumulation_steps(
    )
    args.batch_size = model.train_batch_size()
    # Number of global steps between logging sessions
    args.throughput_logging_freq = max(args.throughput_logging_samples // args.batch_size, 1)


    # Set DeepSpeed info
    args.local_rank = model.local_rank
    args.device = model.device
    args.fp16 = model.fp16_enabled()
    args.bf16 = model.bfloat16_enabled()
    args.use_lamb = (model.optimizer_name() ==
                     deepspeed.runtime.config.LAMB_OPTIMIZER
                     or model.optimizer_name() ==
                     deepspeed.runtime.config.ONEBIT_LAMB_OPTIMIZER)

    if args.use_torch_compile:
        torch.compiler.reset()
        model = torch.compile(model)
    print(model)
    return model, optimizer


def load_checkpoint(args, model):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}, CKPT_ID={args.load_checkpoint_id}"
    )
    start_epoch, global_step, global_data_samples = load_training_checkpoint(
        args=args,
        model=model,
        PATH=args.load_training_checkpoint,
        ckpt_id=args.load_checkpoint_id)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step} and global data samples at {global_data_samples}"
    )
    if args.rewarmup:
        logger.info(
            f"Rewarmup learning rate with last_global_step_from_restore = {global_step}"
        )
        last_global_step_from_restore = global_step

    return start_epoch

def evaluate_model(args, index, model):
    if args.eval_train_data and args.train_dataset is not None:
        pretrain_validation(args, args.train_dataset, "Train", index, model)
    if not args.no_eval_val_data:
        pretrain_validation(args, args.eval_dataset, "Validation", index, model)
    if args.eval_test_data:
        pretrain_validation(args, args.test_dataset, "Test", index, model)

def run(args, model, optimizer, start_epoch):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger
    task = args.task
    if args.materialize_ffn_weights:
        for layer in model.bert.encoder.layer:
            layer.ffn.rescale_weights()
    # if args.use_nvidia_dataset:
    #     pretrain_dataset_provider = NvidiaBertDatasetProvider(args)
    # else:
    #     pretrain_dataset_provider = BingBertDatasetProvider(args)
    print(model)
    print(f"Total parameters in the model: {model.get_num_params(non_embedding=False)}")
    print("Loading train dataset")

    if args.use_sharded_dataset:
        dataset = ShardedDatasetWrapper(args.data_path_prefix,
                                        config["data"]["training"], args)
    else:
        dataset = task.dataset_type(args.data_path_prefix,
                                    config["data"]["training"], args)

    dataset_val = None
    if not args.no_eval_val_data:
        print("Loading eval dataset")
        dataset_val = task.dataset_type(args.data_path_prefix,
                                        config["data"]["validation"], args)

    dataset_test = None
    if args.eval_test_data:
        print("Loading test dataset")
        dataset_test = task.dataset_type(args.data_path_prefix,
                                         config["data"]["test"], args)

    args.train_dataset = dataset if not args.use_sharded_dataset else None
    args.eval_dataset = dataset_val
    args.test_dataset = dataset_test
    #add_normalizer_preforward_hooks(args, model)
    #backward_hooks = report_model_gradients(args, model)
    #forward_hooks = report_activations_fast(args, model)

    if args.eval_only:
        max_samples = args.max_validation_samples
        eval_bs = args.train_micro_batch_size_per_gpu * args.eval_bs_ratio
        # Dry run to compile the model
        dry_run_n_samples = eval_bs * 4
        args.max_validation_samples = dry_run_n_samples
        evaluate_model(args, -1, model)
        # Full run for measuring speed and quality
        args.max_validation_samples = max_samples
        evaluate_model(args, 0, model)
        return

    for index in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"Training Epoch: {index + 1}")
        pre = time.time()
        train(args, index, model, optimizer, dataset)
        #report_model_weights(args, model, global_step)
        # Save ckpts according to "--ckpt_to_save" option,
        # e.g. "--ckpt_to_save 160 161" to save epoch 160 and 161.
        evaluate_model(args, index, model)
        if args.ckpt_to_save > 0 and index % args.ckpt_to_save == 0:
            logger.info(
                f"Saving a checkpointing of the model for epoch: {index+1}")

            success = False
            while not success:
                try:
                    checkpoint_model(
                        PATH=args.saved_model_path,
                        ckpt_id='epoch{}_step{}'.format(index + 1, global_step),
                        model=model, epoch=index + 1,
                        last_global_step=global_step,
                        last_global_data_samples=global_data_samples
                    )
                    success = True
                except Exception as ex:
                    print(ex)
                    

        post = time.time()
        logger.info(f"Time for shard {index + 1}: {post-pre} seconds")

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args, global_steps=current_global_step):
            print(
                f'Warning: Early training termination due to max steps limit, epoch={index+1}, global_step={current_global_step}'
            )
            break


def main():
    start = time.time()
    args = construct_arguments()
    model, optimizer = prepare_model_optimizer(args)
    if master_process(args):
        os.makedirs(args.saved_model_path, exist_ok=True)
        # Set experiment tracking
        if not args.no_clearml:
            task = Task.init(project_name=args.project_name,
                             task_name="research", reuse_last_task_id=False)
            Task.set_random_seed(args.seed)
            task.connect(args)
            task.connect(args.config, 'bert_config')
            task.connect_configuration(args.deepspeed_config,
                                       name='deepspeed_config')
            args.tracker_logger = task.get_logger()
        else:
            args.tracker_logger = TensorBoardWriter(name=args.job_name,
                                                    base=args.output_dir)

    start_epoch = 0
    if args.load_training_checkpoint and args.load_checkpoint_id:
        start_epoch = load_checkpoint(args, model)

    run(args, model, optimizer, start_epoch)
    elapsed = time.time() - start
    logger = args.logger
    logger.info(f"Elapsed time: {elapsed} seconds")


if __name__ == "__main__":
    main()


