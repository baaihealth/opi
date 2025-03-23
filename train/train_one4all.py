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
import re
import copy
import wandb
import math
import torch
import sys; sys.path.append(".") 
from utils import jload, rank0_log
import transformers
from dataclasses import dataclass, field
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROTEIN_PLH_TOKEN = ['<SOA>','<EOA>']
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
    enable_lora: bool = field(default=False, metadata={"help": "Enable LoRA training."})
    add_plh_token: bool = field(default=False, metadata={"help": "Add placeholder token during training."})

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

def add_tokens_and_resize_embeddings(model, text_tokenizer, aa_plh):
    num_added_tokens = text_tokenizer.add_tokens(aa_plh)
    if num_added_tokens > 0:
        rank0_log(f"Added {num_added_tokens} amino acid placeholder tokens: {PROTEIN_PLH_TOKEN}")
    model.resize_token_embeddings(len(text_tokenizer))
    
    # Initialize the model's embeddings to include the new tokens
    new_tokens_tensor = torch.empty(num_added_tokens, model.config.hidden_size)
    new_tokens_embedding = torch.nn.init.kaiming_normal_(new_tokens_tensor, a=math.sqrt(5))
    # print('new_tokens_embedding:', new_tokens_embedding, new_tokens_embedding.shape)
    # Transfer the embeddings to your model's embedding layer
    model.get_input_embeddings().weight.data[-num_added_tokens:] = new_tokens_embedding
    # Resize the output embeddings if needed
    model.get_output_embeddings().weight.data[-num_added_tokens:] = new_tokens_embedding

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    text_tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = text_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(text_tokenizer))

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

# def _combined_tokenize_fn(strings: Sequence[str], 
#                  residue_tokenizer: transformers.EsmTokenizer,
#                  text_tokenizer: transformers.PreTrainedTokenizer) -> Dict:
#     """Tokenize a list of strings."""
#     combined_input_ids = []
#     combined_labels = []
#     input_ids_lens = []
#     labels_lens = []

#     for text in strings:
#         if all(aa in "LAGVSERTIDPKQNFYMHWCXBUZO" for aa in text):  # Assuming uppercase amino acids
#             # Tokenize using the appropriate tokenizer
#             residue_tokenized = residue_tokenizer(
#                 text,
#                 return_tensors="pt",
#                 padding="longest",
#                 max_length=residue_tokenizer.model_max_length,
#                 truncation=True,
#             )
#         else:
#             # Tokenize using the appropriate tokenizer
#             text_tokenized = text_tokenizer(
#                 text,
#                 return_tensors="pt",
#                 padding="longest",
#                 max_length=residue_tokenizer.model_max_length,
#                 truncation=True,
#             )

#         # Collect input_ids and length calculations
#         residue_input_ids = residue_tokenized.input_ids[0]
#         text_input_ids = text_tokenized.input_ids[0]
#         combined_ids = torch.cat((residue_input_ids, text_input_ids), dim=0)
#         combined_labels = torch.cat((residue_input_ids, text_input_ids), dim=0)
        
#         residue_input_ids_len = residue_input_ids.ne(residue_tokenizer.pad_token_id).sum().item()
#         text_input_ids_len = text_input_ids.ne(text_tokenizer.pad_token_id).sum().item()
#         combined_input_ids_len = residue_input_ids_len + text_input_ids_len
#         # input_ids = tokenized.input_ids[0]
#         # input_ids_len = input_ids.ne(tokenizer.pad_token_id).sum().item()
        
#         combined_input_ids.append(combined_ids)
#         combined_labels.append(combined_labels)
#         input_ids_lens.append(combined_input_ids_len)
#         labels_lens.append(combined_input_ids_len)

#     return dict(
#         input_ids=combined_input_ids,
#         labels=combined_labels,
#         input_ids_lens=input_ids_lens,
#         labels_lens=labels_lens,
#     )

def _tokenize_fn(strings: Sequence[str], text_tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        text_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=text_tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(text_tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
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
    text_tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, text_tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, 
                 text_tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_log("Loading data...")
        self.list_data_dict = jload(data_path)
        self.text_tokenizer = text_tokenizer

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
       
        amino_acid_pattern = re.compile("^[ACDEFGHIKLMNPQRSTVWYBZXUO]+$")
        input_sequence = example.get("input", "")

        if input_sequence and amino_acid_pattern.match(input_sequence.strip().upper()):
            if len(input_sequence.strip()) >= 3:
                modified_sequence = f"{PROTEIN_PLH_TOKEN[0]}{input_sequence}{PROTEIN_PLH_TOKEN[1]}"
                example["input"] = modified_sequence

        sources = [prompt_input.format_map(example) if example["input"] else prompt_no_input.format_map(example)]
            
        targets = [f"{example['output']}{self.text_tokenizer.eos_token}"]

        data_dict = preprocess(sources, targets, self.text_tokenizer)

        self.input_ids = data_dict["input_ids"][0]
        self.labels = data_dict["labels"][0]
        # self.input_ids = torch.tensor(data_dict["input_ids"][0], dtype=torch.long)
        # self.labels = torch.tensor(data_dict["labels"][0], dtype=torch.long)
        return dict(input_ids=self.input_ids, labels=self.labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    text_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.text_tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.text_tokenizer.pad_token_id),
        )


def make_supervised_data_module(text_tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(text_tokenizer=text_tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(text_tokenizer=text_tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    text_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if model_args.add_plh_token:
        # Add the amino acid placeholder tokens to the text tokenizer
        add_tokens_and_resize_embeddings(model, text_tokenizer, PROTEIN_PLH_TOKEN)
    
    if any(keyword in model_args.model_name_or_path.lower() for keyword in ['llama-3', 'llama3', 'llama_3']):
        text_tokenizer.pad_token = text_tokenizer.eos_token

    elif any(keyword in model_args.model_name_or_path.lower() for keyword in ['galactica', 'galai']):
        special_tokens_dict = dict()
        if text_tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"
        if text_tokenizer.pad_token is None:  
            special_tokens_dict["pad_token"] = "<pad>"
        if text_tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"
        if text_tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            text_tokenizer=text_tokenizer,
            model=model,
        )
    
    if model_args.enable_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none", 
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    # print("Trainable parameters:", trainable_params)        
    
    data_module = make_supervised_data_module(text_tokenizer=text_tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=text_tokenizer, args=training_args, **data_module)
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
