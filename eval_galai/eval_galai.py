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

import argparse
import json, os
parser = argparse.ArgumentParser()
parser.add_argument('--model_idx', default=None, type=str)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_file',default=None, type=str,help="A file that contains instructions (one instruction per line)")
parser.add_argument('--predictions_file', default=None, type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import OPTForCausalLM, AutoTokenizer
from peft import  PeftModel
import tqdm
import re
import sys
sys.path.append("..") 
import utils

os.environ['CUDA_LAUNCH_BLOCKING']='1'
generation_config = dict(
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=1000
    )


# The prompt template below is following the Stanford Alpaca format.
def generate_prompt(prompt_instructions):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "###\n Instruction: {instruction}\n Input:\n{input}\n Output:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "###\n Instruction: {instruction}\n Input:\n{input}\n Output:\n"
        ),
    }
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    (instruction, input, _) = prompt_instructions["instruction"], prompt_instructions["input"], prompt_instructions["output"]
    if input == "":
        input = '<noinput>'
        prompt = prompt_no_input.format_map({'instruction': instruction,'input': input})
    else:
        prompt = prompt_input.format_map({'instruction': instruction,'input': input})
    
    return prompt

model_dict={
            'model_idx1':'path/to/model1',
            'model_idx2':'path/to/model2',
        }

if __name__ == '__main__':
    output_dir="./eval_galai_output"
    os.makedirs(output_dir, exist_ok=True)
    test_files=[
        # "./OPI_DATA/SU/EC_number/test/CLEAN_EC_number_halogenase_test.jsonl",
        # "./OPI_DATA/SU/EC_number/test/CLEAN_EC_number_new_test.jsonl",
        # "./OPI_DATA/SU/EC_number/test/CLEAN_EC_number_price_test.jsonl",
        # "./OPI_DATA/SU/Subcellular_location/test/location_valid.jsonl",
        # "./OPI_DATA/SU/Fold_type-Remote/test/Remote_valid.jsonl",
        
        # './OPI_DATA/AP/Keywords/test/UniProtSeq_keywords_valid.jsonl',
        # './OPI_DATA/AP/Keywords/test/IDFilterSeq_keywords_valid.jsonl',
        # './OPI_DATA/AP/Keywords/test/CASPSimilarSeq_keywords_valid.jsonl',
        # './OPI_DATA/AP/GO/test/UniProtSeq_go_valid.jsonl',
        # './OPI_DATA/AP/GO/test/IDFilterSeq_go_valid.jsonl',
        # './OPI_DATA/AP/GO/test/CASPSimilarSeq_go_valid.jsonl',
        # './OPI_DATA/AP/Function/test/UniProtSeq_function_valid.jsonl',
        # './OPI_DATA/AP/Function/test/IDFilterSeq_function_valid.jsonl',
        './OPI_DATA/AP/Function/test/CASPSimilarSeq_function_valid.jsonl',
        
        # './OPI_DATA/KM/gName2Tissue/test/tissue_valid_manner2.jsonl'
        # './OPI_DATA/KM/gName2Cancer/test/gene_name_to_cancer_new_test.jsonl'
        # './OPI_DATA/KM/gID2Cancer/test/gene_ID_to_cancer_new_test.jsonl'
    ]
    
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = model_dict[args.model_idx]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,model_max_length=3072)
    
    base_model = OPTForCausalLM.from_pretrained(
        model_dict[args.model_idx], 
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    # test data
    sample_data=['']
    for test_f in test_files:
        args.data_file = test_f
        if args.data_file is None:
            examples = sample_data
        else:
            test_file_name= os.path.splitext(os.path.basename(args.data_file))[0]
            args.predictions_file = os.path.join(output_dir, f"{test_file_name}_{args.model_idx}.json")
            eval_tasks = [json.loads(l) for l in open(args.data_file, "r")]
            examples = [
                {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
                for t in eval_tasks
            ]
            print(f"Loaded {len(examples)} human-written instructions for evaluation")
        model.eval()
        
        with torch.no_grad():
            print("Start inference.")
            results = []
            for example in tqdm.tqdm(examples):
                (instruction, input, target) = example["instruction"], example["input"], example["output"]
                input_text = generate_prompt(example)
                inputs = tokenizer(input_text,return_tensors="pt",padding=True)  #add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device), 
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                decoded_output = tokenizer.decode(s,skip_special_tokens=True)
                
                splitted_data = re.split(f"\s+(Instruction|Input|Output):", decoded_output)
                
                if len(splitted_data) != 7:
                    continue    
                else:
                    inst = splitted_data[2].strip()
                    input = splitted_data[4].strip()
                    input = "" if input.lower() == "<noinput>" else input
                    output = splitted_data[6].strip()
                
                saved_dict={
                    'prompt': input_text,
                    'instruction': inst,
                    'input': input,
                    'output': output,
                    'target': target
                }
                results.append(saved_dict)
                utils.jdump(results, args.predictions_file)

            # dirname = os.path.dirname(args.predictions_file)
            # os.makedirs(dirname,exist_ok=True)
            # with open(dirname+'/generation_config.json','w') as f:
            #     json.dump(generation_config,f,ensure_ascii=False,indent=2)