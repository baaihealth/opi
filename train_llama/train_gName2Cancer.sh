#!/bin/bash



OMP_NUM_THREADS=1 torchrun --nnodes=$1 --node_rank=$2 --nproc_per_node=3 train_llama/train.py \
    --model_name_or_path path/to/llama_base_model/hf_version/llama-$3 \
    --data_path  ./OPI_DATA/KM/gName2Cancer/train/gene_name_to_cancer_new_train.json \
    --bf16 True \
    --output_dir path/to/output/llama_ft_opi/llama_ft_gene_name_to_cancer_new_$3_e$4 \
    --num_train_epochs $4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
