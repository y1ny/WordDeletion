# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.

import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    GPT2ForQuestionAnswering,
    GPT2TokenizerFast,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    deepspeed,
)
import pandas as pd
import csv


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/to/gpt2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    n_training_sample: int = field(
        default=800, metadata={"help": "Path to the training data."}
    )
    n_iter: int = field(
        default=0, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False,
    from_scratch: str = field(default="all")


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""
    msg = messages[0]
    encoding = tokenizer(
        msg[0] + '[SEP]' + msg[1] + '[SEP]',
        msg[2],
        max_length=max_len,
        truncation="only_second",
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        
    )
    input_ids = encoding['input_ids']
    start_char = msg[2].split(" ")[:msg[4]]
    
    answer_start = len(" ".join(start_char))
    answer_end = answer_start + len(msg[3])
    start_positions = []
    end_positions = []
    offset_mapping = encoding.pop("offset_mapping")
    sep_count = 0
    for i, offset in enumerate(offset_mapping):

        if encoding['input_ids'][i] == 13:
            sep_count += 1
        if sep_count != 2:
            start_positions.append(0)
            end_positions.append(0)
            continue
        if answer_start in range(offset[0], offset[1]):
            start_positions.append(i)
        else:
            start_positions.append(0)
        
        if answer_end in range(offset[0], offset[1]):
            end_positions.append(i)
        else:
            end_positions.append(0)
    assert sep_count == 2
    inputs = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
        'start_positions': torch.tensor(start_positions, dtype=torch.float),  # start and end positions should be float
        'end_positions': torch.tensor(end_positions, dtype=torch.float)
    }
    

    return inputs


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.max_len)
        self.cached_data_dict[i] = ret

        return ret

import numpy as np
import random
import time
seed = int(str(time.time()).split(".")[-1])
random.seed(seed)
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset 
    rank0_print("Loading data...")

    data_file = pd.read_csv(data_args.data_path,
                        delimiter='\t',
                        quoting=csv.QUOTE_NONE,
                        quotechar=None,)
    data_lst = list(zip(data_file['demon'].values.tolist(), 
                        data_file['d_deleted'].values.tolist(),
                        data_file['test'].values.tolist(),
                        data_file['t_deleted'].values.tolist(),
                        data_file['start'].values.tolist()))

    random.shuffle(data_lst)
    n_train = data_args.n_training_sample
    train_sent = data_lst[:n_train]
    dev_sent = data_lst[n_train:]
    def load_csv_data(sent_lst):
        msgs = []
        for idx, sent in enumerate(sent_lst):
            msgs.append(sent)
        return msgs
    
    train_data = load_csv_data(train_sent)

    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)
    if data_args.eval_data_path:
        eval_data = load_csv_data(dev_sent)
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

import torch.nn as nn


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    training_args.load_best_model_at_end=True
    training_args.evaluation_strategy = IntervalStrategy.STEPS
    training_args.metric_for_best_model = 'loss'
    training_args.output_dir = f"{training_args.output_dir}-{data_args.n_training_sample}-{data_args.n_iter}"
    
    print(training_args)
    training_args.do_train = True
    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    config.use_cache = False
    if training_args.from_scratch == 'all':
        model = GPT2ForQuestionAnswering(
            config,
        )
        print("loadding from scratch")

    elif training_args.from_scratch == 'without_embedding':
        pretrained_model = GPT2ForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            quantization_config=None,
            **model_load_kwargs,
        )
        model = GPT2ForQuestionAnswering(
            config,
        )
        embedding_weights = pretrained_model.transformer.wte.weight.clone().detach()
        pos_embedding_weights = pretrained_model.transformer.wpe.weight.clone().detach()
        model.transformer.wte.weight = nn.Parameter(embedding_weights)
        model.transformer.wpe.weight = nn.Parameter(pos_embedding_weights)
        del pretrained_model
        print("loadding embedding pretrained parameters")


    elif training_args.from_scratch == 'none':
        model = GPT2ForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            quantization_config=None,
            **model_load_kwargs,
        )
        print("loadding all pretrained parameters")

    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        # use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainer
    trainer = Trainer(
        model=model, tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], 
        args=training_args, **data_module
    )

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )


if __name__ == "__main__":
    train()