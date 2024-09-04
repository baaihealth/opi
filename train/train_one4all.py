# This code is adopted from https://github.com/ymcui/Chinese-LLaMA-Alpaca.
#    Copyright 2023 Yiming Cui, Ziqing Yang
# This code is adopted from https://github.com/tatsu-lab/stanford_alpaca.
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import wandb
from loguru import logger
import torch, gc
import sys; sys.path.append(".") 
from utils import jload, rank0_log
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "Instruction: {instruction}\n"
        "Input: {input}\n"
        "Output:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "Instruction: {instruction}\n"
        "Output:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        rank0_log(f'num_new_tokens: {num_new_tokens}')
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        embedding_dim = input_embeddings.size(1)
        
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        input_embeddings[-num_new_tokens:] = input_embeddings_avg.expand(num_new_tokens, embedding_dim)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg.expand(num_new_tokens, embedding_dim)
        
        input_embeddings[-num_new_tokens:] = input_embeddings[-num_new_tokens:].float()
        output_embeddings[-num_new_tokens:] = output_embeddings[-num_new_tokens:].float()



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_log("Loading data...")
        self.list_data_dict = jload(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]: 
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        if len(sources) != 1:
            raise ValueError(f"Expected 1 source, but got {len(sources)} for index {i}.")
        
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        example = sources[0]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        ]
        targets = [f"{example['output']}{self.tokenizer.eos_token}"]

        data_dict = preprocess(sources, targets, self.tokenizer)

        self.input_ids = data_dict["input_ids"][0]
        self.labels = data_dict["labels"][0]
        return dict(input_ids=self.input_ids, labels=self.labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if any(keyword in model_args.model_name_or_path.lower() for keyword in ['llama-3', 'llama3', 'llama_3']):
        tokenizer.pad_token = tokenizer.eos_token
    elif any(keyword in model_args.model_name_or_path.lower() for keyword in ['galactica', 'galai']):
        special_tokens_dict = dict()
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"
        if tokenizer.pad_token is None:  
            special_tokens_dict["pad_token"] = "<pad>"
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if trainer.state and trainer.state.is_world_process_zero:
        try:
            wandb.init(project="OPI", 
                       name=os.path.basename(training_args.run_name), 
                       group=model_args.model_name_or_path.split('/')[-2],
                       resume = "allow",
                    )
        except Exception as e:
            print(f"wandb init failed: {e}")
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
