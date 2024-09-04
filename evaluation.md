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
1) all_task_Llama-3.1-8B-Instruct    4) all_task_OPI_full_1.74M_v2       7) Subcellular_localization        10) Function                        13) gName2Cancer
2) all_task_OPI_full_1.46M_v2        5) EC_number100                     8) Keywords                        11) gSymbol2Tissue
3) all_task_OPI_full_1.61M_v2        6) Fold_type                        9) GO                              12) gSymbol2Cancer
#? 7
TASK NAME selected: Subcellular_localization

Running: python eval/eval_one4all.py Llama-3.1-8B-Instruct Subcellular_localization
```