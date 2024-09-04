## Evaluation with OPI testing data

Start evaluation with [eval/eval_one4all.sh](./eval/eval_one4all.sh)
```bash
bash eval/eval_one4all.sh
```

Following the prompts in the terminal, enter a NUMBER to select a model and a task to run.

**Example:**
```
Enter a model number to select a MODEL NAME:
1) Galactica-6.7B
2) Llama-3.1-8B-Instruct
#? 2
MODEL NAME selected: Llama-3.1-8B-Instruct

Enter a task number to select a TASK NAME:
1) all_task_Llama-3.1-8B-Instruct
2) all_task_OPI_full_1.61M
#? 2
TASK NAME selected: Subcellular_localization

Running: python eval/eval_one4all.py Llama-3.1-8B-Instruct all_task_OPI_full_1.61M
```
Note: 
1. If you select the task "all_task_Llama-3.1-8B-Instruct", ensure you have selected the model "Llama-3.1-8B-Instruct" but not "Galactica-6.7B". This means you evaluate original Llama-3.1-8B-Instruct model on all tasks of OPI. 
2. If you select the task "all_task_OPI_full_1.61M", you can select either "Llama-3.1-8B-Instruct" or "Galactica-6.7B" to evaluate. 
3. When you want to evaluate the original Galactica-6.7B, please refer to [eval/eval_original_galactica/eval_original_galactica.sh](./eval/eval_original_galactica/eval_original_galactica.sh).