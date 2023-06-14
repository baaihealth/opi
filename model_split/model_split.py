
import os
import torch
import json
import tqdm
import argparse
import pathlib
import shutil
from transformers import AutoTokenizer, OPTForCausalLM


def model_split(args):
    model_dict ={
            'model_idx1':'path/to/model1',
            'model_idx2':'path/to/model2',
    }
    original_model_path = model_dict[args.model_idx]
    
    model = OPTForCausalLM.from_pretrained(original_model_path)

    # chunks = 3
    # total_num_params = sum(p.numel() for p in model.model.parameters())  
    # total_num_params_per_shard = total_num_params // (chunks * n_layers)  
    # shard_sizes = [chunks] * n_layers 

    n_layers = len(model.model.decoder.layers)
    param_count=0
    index_dict = {"metadata":{"total_size": 0}}
    index_dict["weight_map"] = {}
    
    chunked_model_path = original_model_path + "_chunked"
    pathlib.Path(chunked_model_path).mkdir(parents=True, exist_ok=True)
    
    shutil.copy(os.path.join(original_model_path,'config.json'),chunked_model_path)
    shutil.copy(os.path.join(original_model_path,'generation_config.json'),chunked_model_path)
    shutil.copy(os.path.join(original_model_path,'special_tokens_map.json'),chunked_model_path)
    shutil.copy(os.path.join(original_model_path,'tokenizer.json'),chunked_model_path)
    shutil.copy(os.path.join(original_model_path,'tokenizer_config.json'),chunked_model_path)
    shutil.copy(os.path.join(original_model_path,'trainer_state.json'),chunked_model_path)
    shutil.copy(os.path.join(original_model_path,'training_args.bin'),chunked_model_path)

    for layer_i in tqdm.tqdm(range(n_layers)):
        chunk_name = f"pytorch_model-{layer_i + 1}-of-{n_layers}.bin"
        state_dict = {
            f"model.decoder.layers.{layer_i}.self_attn.q_proj.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.q_proj.weight"],
            f"model.decoder.layers.{layer_i}.self_attn.q_proj.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.q_proj.bias"],
            f"model.decoder.layers.{layer_i}.self_attn.k_proj.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.k_proj.weight"],
            f"model.decoder.layers.{layer_i}.self_attn.k_proj.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.k_proj.bias"],
            f"model.decoder.layers.{layer_i}.self_attn.v_proj.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.v_proj.weight"],
            f"model.decoder.layers.{layer_i}.self_attn.v_proj.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.v_proj.bias"],
            f"model.decoder.layers.{layer_i}.self_attn.out_proj.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.out_proj.weight"],
            f"model.decoder.layers.{layer_i}.self_attn.out_proj.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn.out_proj.bias"],
            f"model.decoder.layers.{layer_i}.self_attn_layer_norm.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn_layer_norm.weight"],
            f"model.decoder.layers.{layer_i}.self_attn_layer_norm.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.self_attn_layer_norm.bias"],
            f"model.decoder.layers.{layer_i}.fc1.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.fc1.weight"],
            f"model.decoder.layers.{layer_i}.fc1.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.fc1.bias"],
            f"model.decoder.layers.{layer_i}.fc2.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.fc2.weight"],
            f"model.decoder.layers.{layer_i}.fc2.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.fc2.bias"],
            f"model.decoder.layers.{layer_i}.final_layer_norm.weight": model.state_dict()[f"model.decoder.layers.{layer_i}.final_layer_norm.weight"],
            f"model.decoder.layers.{layer_i}.final_layer_norm.bias": model.state_dict()[f"model.decoder.layers.{layer_i}.final_layer_norm.bias"],
        }
        if layer_i == 0:
            state_dict["model.decoder.embed_tokens.weight"] = model.state_dict()["model.decoder.embed_tokens.weight"]
            state_dict["model.decoder.embed_positions.weight"] = model.state_dict()["model.decoder.embed_positions.weight"]
            state_dict["model.decoder.final_layer_norm.weight"] = model.state_dict()["model.decoder.final_layer_norm.weight"]
            state_dict["model.decoder.final_layer_norm.bias"] = model.state_dict()["model.decoder.final_layer_norm.bias"]
        elif layer_i == n_layers-1:
            state_dict["lm_head.weight"] = model.state_dict()["lm_head.weight"] 
            
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = chunk_name
            param_count += v.numel()
        torch.save(state_dict, os.path.join(chunked_model_path, chunk_name))

    index_dict["metadata"]["total_size"] = param_count * 2
    
    # 将模型分成多个子集并保存为 JSON 文件
    with open(os.path.join(chunked_model_path,"pytorch_model.bin.index.json"), "w") as outfile:
        json.dump(index_dict, outfile)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_idx",
        choices=[
            'model_idx1',
            'model_idx2',
            ],
    )
    args = parser.parse_args()

    model_split(args)

if __name__ == "__main__":
    main()