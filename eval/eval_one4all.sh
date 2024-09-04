#!/bin/bash

# 1. Predefine model names
model_names=(
    "Galactica-6.7B"
    "Llama-3.1-8B-Instruct"
)

# Storing model names as an array
model_array=("${model_names[@]}")

# 2. Model display and selection
echo "Enter a model number to select a MODEL NAME:"
select model_name in "${model_array[@]}"; do
    if [ -n "$model_name" ]; then
        echo "MODEL NAME selected: $model_name"
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# 3. Parse to get task name list from YAML file
task_names=$(grep '^[^[:space:]][^:]*:' eval/tasks.yaml | sed 's/://g' | grep -v '^model_name')
# Storing task names as an array
IFS=$'\n' read -rd '' -a task_array <<<"$task_names"

# 4. Task display and selection
echo ""
echo "Enter a task number to select a TASK NAME:"
select task_name in "${task_array[@]}"; do
    if [ -n "$task_name" ]; then
        echo "TASK NAME selected: $task_name"
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# # 5. Starting evaluation
echo ""
echo "Running: python eval/eval_one4all.py $model_name $task_name"
python eval/eval_one4all.py "$model_name" "$task_name"
