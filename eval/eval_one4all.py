import os
import json
import torch
import tqdm
# import hydra
# from hydra import initialize, compose
import sys;sys.path.append(".") 
from utils import jdump
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import StoppingCriteria, GenerationConfig
from peft import PeftModel, PeftConfig

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        self.tokenizer = tokenizer

        # Convert keywords to input_ids and calculate the maximum keyword length
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Debugging: Print output_ids to check if it matches any keyword
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                print(f"Keyword id match found: {keyword_id}, {self.tokenizer.convert_ids_to_tokens(keyword_id)}")
                return True
        
        # Decode output and check for keyword presence
        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        for keyword in self.keywords:
            if len(keyword) > 1 and keyword in outputs:
                print(f"Keyword found in decoded output: {keyword}")
                return True

        return False

def generate_prompt(prompt_instructions):    
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
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    instruction, input_data = prompt_instructions["instruction"], prompt_instructions['instances'][0]["input"] if 'instances' in prompt_instructions else prompt_instructions["input"]
    
    if input_data == "":
        input_data = '<noinput>'
        prompt = prompt_no_input.format(instruction=instruction, input=input_data)
    else:
        prompt = prompt_input.format(instruction=instruction, input=input_data)
    return prompt

def main(cfg: DictConfig):
    root_dir = '/path/to/LLM_checkpoints_or_test_files'

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'lora' in cfg.model.name:
        config = PeftConfig.from_pretrained(os.path.join(root_dir, cfg.model.name))
        base_model_name_or_path = config.base_model_name_or_path
        os.makedirs(f'latest_eval_results/{os.path.basename(cfg.model.name)}', exist_ok=True)
    else:
        base_model_name_or_path = os.path.join(root_dir, cfg.model.name)
        os.makedirs(f'latest_eval_results/{os.path.basename(cfg.model.name)}', exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, 
        load_in_8bit=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_dir, cfg.model.name))
    print('base_model_tokenizer.vocab_size:', len(base_model_tokenizer.get_vocab())) #base model 151665
    print('tokenizer.vocab_size:', len(tokenizer.get_vocab())) #151667 = 151665 + 2 added tokens
    
    if 'lora' in cfg.model.name:
        lora_path = os.path.join(root_dir, cfg.model.name)
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        try:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print("LoRA weights merged successfully")
        except Exception as e:
            print(f"Error merging LoRA weights: {e}")
            print("Continuing with unmerged model")
    else:
        model = base_model
    
    # model_vocab_size = base_model.get_input_embeddings().weight.size(0) #152064
    if len(base_model_tokenizer.get_vocab()) < len(tokenizer.get_vocab()):
        print("Resizing model embeddings to fit tokenizer")
        model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    model.eval()
    
    if any(keyword in cfg.model.name.lower() for keyword in ['llama-3', 'llama3', 'llama_3', 'deepseek']):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        terminators = [t for t in terminators if t is not None]
        if not terminators:
            terminators = [tokenizer.eos_token_id]
        eos_token_id = terminators
        pad_token_id = tokenizer.eos_token_id
    elif any(keyword in cfg.model.name.lower() for keyword in ['galactica', 'galai']):
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids("</s>")  # 通常 </s> 的 ID 是 2
        if pad_token_id is None:
            pad_token_id = eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    
    print(f"Using eos_token_id: {eos_token_id}, pad_token_id: {pad_token_id}")
    
    genaration_cfg = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.75,
        num_beams=1,
        # repetition_penalty=1.3,
        max_new_tokens=400,
        use_cache=True,
        do_sample=True,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    for test_file in cfg.data.test_files:
        test_file_path = os.path.join(root_dir, test_file)
        examples = [json.loads(line) for line in open(test_file_path, "r")]
        print(f"Loaded {len(examples)} samples for evaluation of {test_file}.")

        output_file = os.path.join(f'latest_eval_results/{os.path.basename(cfg.model.name)}', f"{os.path.basename(cfg.model.name)}_{os.path.basename(test_file).split('.')[0]}_predictions.json")
        
        results = []
        with torch.no_grad():
            for example in tqdm.tqdm(examples):
                input_text = generate_prompt(example)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                input_ids=inputs["input_ids"]
                attention_mask=inputs["attention_mask"]
                if any(keyword in cfg.model.name.lower() for keyword in ['llama-3', 'llama3', 'llama_3', 'deepseek']):
                    stopping_criteria = KeywordsStoppingCriteria(['<|eot_id|>'], tokenizer, input_ids)
                elif any(keyword in cfg.model.name.lower() for keyword in ['galactica', 'galai']):
                    stopping_criteria = KeywordsStoppingCriteria(['</s>'], tokenizer, input_ids)
                else:
                    stopping_criteria = KeywordsStoppingCriteria([tokenizer.eos_token or '</s>'], tokenizer, input_ids)
                
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=genaration_cfg,
                    stopping_criteria=[stopping_criteria]
                )

                decoded_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)

                if "Output:" in decoded_output:
                    decoded_output = decoded_output.split("Output:", 1)[1].strip().replace(' ; ','; ')
                
                # Save results
                results.append({"Instruction": example["instruction"],
                                "input": example["instances"][0]['input'] if "instances" in example else example["input"], 
                                "target": example["instances"][0]['output'].replace(' ; ','; ') if "instances" in example else example["output"].replace(' ; ','; '),
                                "predict": decoded_output})
            
                # Save to output file
                jdump(results, output_file)

def replace_model_name(config, model_name):
    # Replace ${model_name} with the actual model_name
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = value.replace('${model_name}', model_name)
            elif isinstance(value, dict):
                config[key] = replace_model_name(value, model_name)
            elif isinstance(value, list):
                config[key] = [replace_model_name(item, model_name) if isinstance(item, dict) else item for item in value]
    return config

def load_task_config(model_name, task_name, cli_args=None, config_path="eval/tasks.yaml"):
    # Load the full configuration
    config = OmegaConf.load(config_path)
    
    # Ensure model_name is set if provided
    if model_name:
        # Update model_name in the configuration
        config.model_name = model_name
        
    # Check if the specified task exists in the configuration
    if task_name not in config:
        available_tasks = ', '.join(config.keys())
        raise ValueError(f"Task '{task_name}' not found in the config file. Choose from [{available_tasks}]")

    # Get the specific task configuration
    task_config = config[task_name]
    
    # Resolve interpolations, if any
    task_config = OmegaConf.create(OmegaConf.to_container(task_config, resolve=True))
    
    # Handle command-line arguments, if provided
    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        task_config = OmegaConf.merge(task_config, cli_config)

    return task_config

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError(
            "Please specify the model name and task name as command-line arguments.\n"
            "Available models can be defined in eval/eval_one4all.sh, such as:\n"
            "  - Galactica-6.7B\n"
            "  - Llama-3.1-8B-Instruct\n"
            "  - ..."
            "\n"
            "Available task names can be defined in eval/tasks.yaml, such as:\n"
            "  - all_task_Llama-3.1-8B-Instruct\n"
            "  - all_task_OPI_full_1.61M\n"
            "  - EC_number\n"
            "  - Subcellular_localization\n"
            "  - Fold_type\n"
            "  - Keywords\n"
            "  - GO\n"
            "  - Function\n"
            "  - gSymbol2Tissue\n"
            "  - gSymbol2Cancer\n"
            "  - gName2Cancer\n"
        )
    
    model_name = sys.argv[1]
    task_name = sys.argv[2]
    cli_args = sys.argv[3:] if len(sys.argv) > 3 else None
    
    task_config = load_task_config(model_name, task_name, cli_args=cli_args)
    print(OmegaConf.to_yaml(task_config))
    main(task_config)
    