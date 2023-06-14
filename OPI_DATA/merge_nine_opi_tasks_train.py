import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="merge instruction training data of nine tasks")
    
    parser.add_argument(
        "--output", default=None, help="file name of the merged json file", type=str
    )
    
    args = parser.parse_args()
    
    file_list = [
        "./AP/Function/train/function_description_train.json",
        "./AP/GO/train/go_terms_train.json",
        "./AP/Keywords/train/keywords_train.json",
        
        "./KM/gSymbol2Cancer/train/gene_symbol_to_cancer_new_train.json",
        "./KM/gName2Cancer/train/gene_name_to_cancer_new_train.json",
        "./KM/gSymbol2Tissue/train/tissue_train_manner2.json",
        
        "./SU/EC_number/train/CLEAN_EC_number_split100_train.json",
        "./SU/Fold_type-Remote/train/Remote_train.json",
        "./SU/Subcellular_location/train/location_train.json"
    ]
    
    merge_json_content = []
    for json_file in file_list:
        
        with open(json_file) as infile:
            json_content = json.load(infile)
        
        for item in json_content:
            merge_json_content.append(item)
    
    
    with open(args.output,"w") as outfile:
        json.dump(merge_json_content,outfile,indent=2)

if __name__ == '__main__':  
    main()
