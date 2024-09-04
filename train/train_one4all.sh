#!/bin/bash

### Define an associative array (dictionary) mapping model names to their corresponding model paths
declare -A model_paths=(
    ["Llama-3.1-8B-Instruct"]="/path/to/LLM_checkpoints/Llama3.1/Llama-3.1-8B-Instruct"
    ["Galactica-6.7B"]="/path/to/LLM_checkpoints/Galactica/Galactica-6.7B"
)

### Define an associative array (dictionary) mapping task names to their corresponding data paths
declare -A data_paths=(
    ["EC_number100"]="/path/to/OPI_DATAv1/SU/EC_number/train/CLEAN_EC_number_split100_train.json"
    ["Fold_type"]="/path/to/OPI_DATAv1/SU/Fold_type/train/fold_type_train.json"
    ["Subcellular_localization"]="/path/to/OPI_DATAv1/SU/Subcellular_localization/train/subcell_loc_train.json"
    ["Function"]="/path/to/OPI_DATAv1/AP/Function/train/function_train.json"
    ["Go_terms"]="/path/to/OPI_DATAv1/AP/GO_terms/train/go_terms_train.json"
    ["Keywords"]="/path/to/OPI_DATAv1/AP/Keywords/train/keywords_train.json"
    ["gName2Cancer"]="/path/to/OPI_DATAv1/KM/gName2Cancer/train/gene_name_to_cancer_train.json"
    ["gSymbol2Cancer"]="/path/to/OPI_DATAv1/KM/gSymbol2Cancer/train/gene_symbol_to_cancer_train.json"
    ["gSymbol2Tissue"]="/path/to/OPI_DATAv1/KM/gSymbol2Tissue/train/gene_symbol_to_tissue_train.json"
    ["OPI_full_1.46M_v2"]="/path/to/OPI_DATAv1/OPI_full_1.46M_v2_train.json"
    ["OPI_full_1.61M_v2"]="/path/to/OPI_DATAv1/OPI_full_1.61M_v2_train.json"
    ["OPI_full_1.74M_v2"]="/path/to/OPI_DATAv1/OPI_full_1.74M_v2_train.json"
)
# Define the order of tasks
ordered_tasks=(
    "EC_number100"
    "Fold_type"
    "Subcellular_localization"
    "Function"
    "Go_terms"
    "Keywords"
    "gName2Cancer"
    "gSymbol2Cancer"
    "gSymbol2Tissue"
    "OPI_full_1.46M_v2"
    "OPI_full_1.61M_v2"
    "OPI_full_1.74M_v2"
)

# Display model names and handle selection
echo "Enter a model number to select a MODEL NAME:"
select model_name in "${!model_paths[@]}"; do
    if [ -n "$model_name" ]; then
        selected_model_path="${model_paths[$model_name]}"
        echo ""
        echo "MODEL NAME selected: $model_name"
        echo "MODEL PATH: $selected_model_path"
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# Display task names and handle selection
echo ""
echo "Enter a task number to select a TASK NAME:"
select task_name in "${ordered_tasks[@]}"; do
    if [ -n "$task_name" ]; then
        selected_data_path="${data_paths[$task_name]}"
        echo ""
        echo "TASK NAME selected: $task_name"
        echo "DATA PATH: $selected_data_path"
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# Set num_train_epochs based on the selected task
if [[ "$task_name" == "OPI_full_1.46M_v2" || "$task_name" == "OPI_full_1.61M_v2" || "$task_name" == "OPI_full_1.74M_v2" ]]; then
    num_train_epochs=1
else
    num_train_epochs=3
fi

echo ""
echo "EPOCHS: $num_train_epochs"

# Run the selected task
data_path_base_name=$(basename "$selected_data_path" .json)
output_dir=/path/to/LLM_checkpoints/OPI_IT/${model_name}_${data_path_base_name}_e${num_train_epochs}
echo "OUTPUT_DIR: $output_dir"

deepspeed train/train_one4all.py \
    --deepspeed configs/zero3.json \
    --model_name_or_path $selected_model_path \
    --data_path $selected_data_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True \
    --tf32 True \
    --report_to wandb