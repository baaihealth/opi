from transformers import AutoTokenizer, OPTForCausalLM
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="original galactica infer pipeline")
    
    parser.add_argument("--input", default=None, help="file path of input file", type=str)
    parser.add_argument("--output", default=None, help="file path of result file", type=str)
    parser.add_argument("--task", default=None, help="the name of selected task", type=str)
    
    args = parser.parse_args()
    
    with open(args.input) as infile:
        content = infile.readlines()

    for line in content:
        if line.endswith("\n"):
            record = json.loads(line[:-1])
        else:
            record = json.loads(line)
        
        seq = record["instances"][0]["input"]

        if args.task == "keyword":
            input_text = "[START_AMINO]"+seq+"[END_AMINO] ### Keywords"
        elif args.task == "function_description":
            input_text = "[START_AMINO]"+seq+"[END_AMINO] ## Function"
        elif args.task == "GO":
            input_text = "[START_AMINO]"+seq+"[END_AMINO] ## Gene Ontology"
        elif args.task == "EC":
            input_text = "[START_AMINO]"+seq+"[END_AMINO], EC (Enzyme commission) number use four digits to classify enzymes into thousands of classes based on their catalytic function. What is the EC number of the given seqence?"
        elif args.task == "symbol_to_cancer":
            input_text = "What is the cancer name that is associated with the given gene symbol {}?".format(seq)
        elif args.task == "name_to_cancer":
            input_text = "What is the cancer name that is associated with the given gene name {}?".format(seq)
        elif args.task == "tissue":
            input_text = "You can according to the protein Gene name to predict the protein expressed in normal human body what organization? All organs of the human body including: appendix, breast, bronchus, cerebral cortex, cervix, colon, duodenum, endometrium, epididymis, esophagus, fallopian tube, gallbladder, kidney, liver, lung, nasopharynx, oral mucosa, pancreas, parathyroid gland, placenta, prostate, rectum, salivary gland, seminal vesicle, skin, small intestine, stomach, testis, thyroid gland, tonsil, urinary bladder, vagina, adipose tissue, adrenal gland, bone marrow, caudate, cerebellum, heart muscle, hippocampus, lymph node, ovary, skeletal muscle, smooth muscle, soft tissue, spleen. Return the organs based on the gene {}".format(seq)
            
        else:
            print("invalid task name.")
            input_text = "[START_AMINO]"+seq+"[END_AMINO]"

        #path to galactica_base_mopdel: galactica-6.7b
        tokenizer = AutoTokenizer.from_pretrained("/path/to/galactica_base_mopdel/galactica-6.7b")
        model = OPTForCausalLM.from_pretrained("/path/to/galactica_base_mopdel/galactica-6.7b", device_map="auto")

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids,max_new_tokens=100) 
        record["instances"][0]["galactica"] = tokenizer.decode(outputs[0])

        with open(args.output,"a") as outfile:
            json.dump(record, outfile)
            outfile.write('\n')

if __name__ == '__main__':  
    main()
