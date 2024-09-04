## Environment setup
```
pip install -r requirements.txt
```

## Instruction tuning with OPI training data
Start training with [train/one4all.sh](./train/train_one4all.sh)
```bash
bash train/train_one4all.sh
```

Following the prompts in the terminal, enter a NUMBER to select a model and a task to run.

**Example:**
```
Enter a model number to select a MODEL NAME:
1) Llama3.1-8B-Instruct
2) Galactica-6.7B
#? 1

MODEL NAME selected: Llama3.1-8B-Instruct
MODEL PATH: /path/to/LLM_checkpoints/Llama3.1/Llama-3.1-8B-Instruct

Enter a task number to select a TASK NAME:
1) OPI_full_1.61M              6) Go_terms
2) EC_number                   7) Function
3) Fold_type                   8) gSymbol2Tissue
4) Subcellular_localization    9) gSymbol2Cancer
5) Keywords                   10) gName2Cancer
#? 4

TASK NAME selected: Subcellular_localization
DATA PATH: /path/to/OPI_DATA/SU/Subcellular_localization/train/subcell_loc_train.json

EPOCHS: 3
OUTPUT_DIR: /path/to/LLM_checkpoints/OPI_IT/Llama3.1-8B-Instruct_subcell_loc_train_e3
```