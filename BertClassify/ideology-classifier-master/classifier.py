import os
import random
import pandas as pd
import time

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from data_processing import *
from classifier_parser import get_args
from classifier_logging import *

def main(): 
    args = get_args()
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    device, n_gpu = get_device_and_n_gpu(args.use_cpu, args.local_rank)
    seed_all(args.seed, n_gpu)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=(not args.cased))
    
    processor = LeftRightProcessor(
        args.max_seq_length,
        tokenizer,
        args.data_dir, 
        args.seed, 
        args.train_batch_size, 
        args.eval_batch_size
    )
    label_list = processor.get_labels()
    num_labels = args.num_labels

    create_output_dir(args.output_dir, args.force)
    processor.check_data_exists(args.skip_train)

    model = get_model(args.cache_dir, args.local_rank, args.bert_model, num_labels, device, n_gpu)

    step_tracker = StepValues()

    if not args.skip_train: 
        step_tracker = setup_and_perform_train(
            processor,
            model,
            args.num_train_examples,
            args.gradient_accumulation_steps,
            args.num_train_epochs, args.local_rank,
            args.learning_rate,
            args.warmup_proportion, 
            device, 
            n_gpu
        )
        save_model(model, args.output_dir, num_labels)
    

    if (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.skip_eval:
        setup_and_perform_eval(model, args.num_test_examples, processor, args.output_dir, device, step_tracker)

def setup_and_perform_train( processor, model, num_train_examples, gradient_accumulation_steps,
    num_train_epochs, local_rank, learning_rate, warmup_proportion, device, n_gpu ):
    """ Prepares dataloader and optimizer then performs training on the model """
    train_dataloader = processor.get_train_dataloader(local_rank, num_train_examples)

    num_train_optimization_steps = processor.get_num_train_optimization_steps(gradient_accumulation_steps, num_train_epochs, local_rank)
    log_training(num_train_examples, processor.train_batch_size, num_train_optimization_steps)
    
    optimizer = get_optimizer(model, learning_rate, warmup_proportion, num_train_optimization_steps)

    step_tracker = train_model(model, optimizer, train_dataloader, num_train_epochs, gradient_accumulation_steps, device, n_gpu)
    
    return step_tracker
    

def train_model(model, optimizer, train_dataloader, num_train_epochs, gradient_accumulation_steps, device, n_gpu):
    """ Trains the model on each example in the example set. """
    model.train()
    global_step = 0
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1 
    step_tracker = StepValues(
        global_step=global_step,
        nb_tr_steps=nb_tr_steps, 
        tr_loss=tr_loss
    )
    logger.debug("   *****  STEP TRACKER *****  ")
    logger.debug(str(step_tracker))
    return step_tracker

class StepValues:
    def __init__(self, global_step=0, nb_tr_steps=0, tr_loss=0):
        self.global_step = global_step
        self.nb_tr_steps = nb_tr_steps
        self.tr_loss = tr_loss

    def __repr__(self):
        return "global_step = %d ; nb_tr_stps = %d ; tr_loss = %d" % (self.global_step, self.nb_tr_steps, self.tr_loss)

    def loss(self):
        try:
            return (self.tr_loss/self.nb_tr_steps)
        except ZeroDivisionError:
            return None

def setup_and_perform_eval(model, num_test_examples, processor, output_dir, device, step_tracker):
    eval_dataloader = processor.get_eval_dataloader(num_test_examples)
    log_evaluating(processor.num_eval_examples, processor.eval_batch_size)
    
    result = eval_model(model, device, eval_dataloader, step_tracker)
    
    log_result(output_dir, processor.num_train_examples, processor.num_eval_examples, result)
    record_to_csv(output_dir, processor.num_train_examples, processor.num_eval_examples, result)

def eval_model(model, device, eval_dataloader, step_tracker):
    """ Determines the accuracy of the model's predictions for each example in the test set. 
     Returns as a dictionary with keys 'eval_loss', 'eval_accuracy', 'global_step', 'loss' """
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = step_tracker.loss()
    result = {
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'global_step': step_tracker.global_step,
        'loss': loss
    }
    return result

def accuracy(out, labels):
    """ returns the number of predictions that equal the correct label """
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def get_model(cache_dir, local_rank, bert_model, num_labels, device, n_gpu):
    "Returns a model with the given specifications"

    cache_path = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(local_rank))

    model = BertForSequenceClassification.from_pretrained(
        bert_model,
        cache_dir=cache_path,
        num_labels = num_labels
    )

    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model

def get_optimizer(model, learning_rate, warmup_proportion, num_train_optimization_steps):
    "Sets the given specifications for the optimizer then returns the optimizer"

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        warmup=warmup_proportion,
        t_total=num_train_optimization_steps
    )
    return optimizer

def save_model(model, output_dir, num_labels):
    """ Saves the model to the given output directory """
    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

def get_device_and_n_gpu(use_cpu, local_rank):
    """ Determines where to do calculations and how many GPUs the system has """
    if local_rank == -1 or use_cpu:
        device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device,
        n_gpu,
        bool(local_rank != -1)
    ))
    return device, n_gpu

def seed_all(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    main()
