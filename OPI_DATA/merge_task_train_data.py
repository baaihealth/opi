import os
import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="merge instruction training data of nine tasks")
    
    parser.add_argument(
        "--output", default=None, help="file name of the merged json file", type=str
    )
    
    args = parser.parse_args()
    
    file_list = [
        "./OPI_DATA/SU/EC_number/train/CLEAN_EC_number_train.json",
        "./OPI_DATA/SU/Fold_type/train/fold_type_train.json",
        "./OPI_DATA/SU/Subcellular_localization/train/subcell_loc_train.json",
     
        "./OPI_DATA/AP/GO/train/go_terms_train.json",
        "./OPI_DATA/AP/Keywords/train/keywords_train.json",
        "./OPI_DATA/AP/Function/train/function_train.json",
        
        "./OPI_DATA/KM/gSymbol2Tissue/train/gene_symbol_to_tissue_train.json"
        "./OPI_DATA/KM/gSymbol2Cancer/train/gene_symbol_to_cancer_train.json",
        "./OPI_DATA/KM/gName2Cancer/train/gene_name_to_cancer_train.json",
    ]
        
    
    merge_json_content = []
    for json_file in file_list:
        with open(os.path.join('./OPI_DATA', json_file)) as infile:
            json_content = json.load(infile)
        
        for item in json_content:
            merge_json_content.append(item)
    
    random.shuffle(merge_json_content) 
    with open(args.output,"w") as outfile:
        json.dump(merge_json_content,outfile,indent=2)

if __name__ == '__main__':  
    main()
