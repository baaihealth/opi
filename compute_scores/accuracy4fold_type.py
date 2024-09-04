
import os
import json
import tqdm

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_accuracy4fold_type(eval_file, test_files):
    """Compute accuracy for predictions against test datasets."""
    # Load evaluation data
    eval_data = load_json(eval_file)
    acc_dict={}
    # Iterate over each test file
    for test_file in test_files:
        # Load test data
        test_data = load_json(test_file)

        # Create a set of test sequences
        test_seq_set = {item["primary"] for item in test_data}

        # Initialize counters
        correct_predictions = 0
        total_predictions = 0

        # Evaluate each item in the evaluation data
        for item in tqdm.tqdm(eval_data):
            if item['input'] not in test_seq_set:
                continue
            predict = item.get("output", item.get("predict", []))
            label = item["target"]
            if predict == label:
                correct_predictions += 1
            total_predictions += 1

        # Calculate and print accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        acc_dict[os.path.basename(test_file).split('.')[0][21:]] = round(accuracy, 4)
    return acc_dict

# Example usage (assuming this script is run directly)
if __name__ == "__main__":
    # eval_file_path="eval/eval_galai/eval_galai_opi_output/Remote_valid_opi_full_6.7b_e3_0.2_0.9_1.json"
    # eval_file_path="eval/eval_galai/eval_galai_opi_output/Remote_valid_opi_full_6.7b_e10_0.2_0.9_1.json"
    # eval_file_path="eval/eval_galai/eval_galai_opi_output/Remote_valid_remote_6.7_e30.json"
    # eval_file_path="eval/eval_galai/eval_galai_opi_output/Remote_valid_remote_6.7.json"
    eval_file_path="latest_eval_results/Galactica-6.7B_OPI_full_train_1.46M_e3/fold_type_test_opi_full_6.7b_e3_0.2_0.9_1.json"
    # eval_file_path = 'latest_eval_results/Llama-3.1-8B-Instruct_fold_type_train_e3/Llama-3.1-8B-Instruct_fold_type_train_e3_fold_type_test_predictions.json'
    test_file_paths = [
        'compute_scores/remote_homology_test_fold_holdout.json',
        'compute_scores/remote_homology_test_superfamily_holdout.json',
        'compute_scores/remote_homology_test_family_holdout.json'
    ]
    print(compute_accuracy4fold_type(eval_file_path, test_file_paths))