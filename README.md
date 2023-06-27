<div align="center">
<img src=./OPI_logo.png />

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/WEIGHT_DIFF_LICENSE)

# <center> OPI: Exploring and Benchmarking Large Language Models for Protein Modeling

</div>

**VISION & ROADMAP.** *Open Protein Instructions(OPI) is the initial part of Open Biology Instructions(OBI) project, together with the subsequent Open Molecule Instructions(OMI), Open DNA Instructions(ODI) and Open RNA Instructions(ORI). OBI is a project which aims to fully leverage the potential ability of Large Language Models(LLMs), especially the scientific LLMs like Galactica, to facilitate research in AI for Life Science community. While OBI is still in an early stage, we hope to provide a starting point for the community to facilitate research, by bridging LLMs and biological domain konwledge.*


## Overview
This repo is for the **Open Protein Instructions (OPI)** project, aiming to build and release a protein instruction dataset as well as propose to explore and benckmark LLMs for protein modeling in protein biology.
<!-- ![Overview](./Overview.png) -->
<div align="center">
<img src=./Overview.png />
</div>

**Usage and License Notices:** [LLaMA](https://github.com/facebookresearch/llama) and [Galactica](https://github.com/paperswithcode/galai) are intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The weight diff for [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) is also CC BY NC 4.0 (allowing only non-commercial use).


## OPI dataset construction pipeline
The OPI dataset is curated on our own by extracting key informatoin from [Swiss-Prot](https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue&query=%2A) database. The detailed construction pipeline is depicted in the supplenmentary material of our manuscript which has been submitted to NeurIPS 2023 Datasets and Benchmarks. The following figure shows the general construction process.

<!-- ![OPI construction](./OPI_data.png#pic_center) -->
<div align="center">
<img src=./OPI_data.png />
</div>


- **An example of OPI training data:**
```
instruction: 
    What is the EC classification of the input protein sequence based on its biological function?
input:                         
    MGLVSSKKPDKEKPIKEKDKGQWSPLKVSAQDKDAPPLPPLVVFNHLTPPPPDEHLDEDKHFVVALYDYTAMNDRDLQMLKGEKLQVLKGTGDWWLARS
    LVTGREGYVPSNFVARVESLEMERWFFRSQGRKEAERQLLAPINKAGSFLIRESETNKGAFSLSVKDVTTQGELIKHYKIRCLDEGGYYISPRITFPSL
    QALVQHYSKKGDGLCQRLTLPCVRPAPQNPWAQDEWEIPRQSLRLVRKLGSGQFGEVWMGYYKNNMKVAIKTLKEGTMSPEAFLGEANVMKALQHERLV
    RLYAVVTKEPIYIVTEYMARGCLLDFLKTDEGSRLSLPRLIDMSAQIAEGMAYIERMNSIHRDLRAANILVSEALCCKIADFGLARIIDSEYTAQEGAK
    FPIKWTAPEAIHFGVFTIKADVWSFGVLLMEVVTYGRVPYPGMSNPEVIRNLERGYRMPRPDTCPPELYRGVIAECWRSRPEERPTFEFLQSVLEDFYT
    ATERQYELQP
output: 
    2.7.10.2
```

## OPI dataset release
We release the first curated OPI dataset corresponding to the 9 evaluation tasks to facilitate further research in protein biology domain. We are warmly welcome further improvenmet or supplement to this dataset.

**How to access the OPI dataset?** The OPI dataset can be accessed via this link [OPI_DATA](https://drive.google.com/drive/folders/1l04jJSOb7BrlbtE9Sy9VzUHCQRtOBGiq?usp=drive_link) from Google Drive. 
Once finished downloading the **OPI_DATA** from Google Drive, please put the three subfolders, i.e., AP, KM and SU, into the **OPI_DATA** folder in this repo. 

The **OPI dataset folder structure** is as follows:
```
./OPI_DATA/
├── AP
│   ├── Function
│   │   ├── test
│   │   │   ├── CASPSimilarSeq_function_valid.jsonl
│   │   │   ├── IDFilterSeq_function_valid.jsonl
│   │   │   └── UniProtSeq_function_valid.jsonl
│   │   └── train
│   │       ├── function_description_train.json
│   │       └── function_description_train_0.01.json
│   ├── GO
│   │   ├── test
│   │   │   ├── CASPSimilarSeq_go_valid.jsonl
│   │   │   ├── IDFilterSeq_go_valid.jsonl
│   │   │   └── UniProtSeq_go_valid.jsonl
│   │   └── train
│   │       ├── go_terms_train.json
│   │       └── go_terms_train_0.01.json
│   └── Keywords
│       ├── test
│       │   ├── CASPSimilarSeq_keywords_valid.jsonl
│       │   ├── IDFilterSeq_keywords_valid.jsonl
│       │   └── UniProtSeq_keywords_valid.jsonl
│       └── train
│           ├── keywords_train.json
│           └── keywords_train_0.01.json
├── KM
│   ├── gSymbol2Cancer
│   │   ├── test
│   │   │   └── gene_symbol_to_cancer_test.jsonl
│   │   └── train
│   │       └── gene_symbol_to_cancer_train.json
│   ├── gName2Cancer
│   │   ├── test
│   │   │   └── gene_name_to_cancer_test.jsonl
│   │   └── train
│   │       └── gene_name_to_cancer_train.json
│   └── gSymbol2Tissue
│       ├── test
│       │   └── gene_symbol_to_tissue_valid.jsonl
│       └── train
│           └── gene_symbol_to_tissue_train.json
└── SU
    ├── EC_number
    │   ├── test
    │   │   ├── CLEAN_EC_number_new_test.jsonl
    │   │   └── CLEAN_EC_number_price_test.jsonl
    │   └── train
    │       ├── CLEAN_EC_number_train.json
    ├── Fold_type-Remote
    │   ├── test
    │   │   └── Remote_valid.jsonl
    │   └── train
    │       └── Remote_train.json
    └── Subcellular_location
        ├── test
        │   ├── location_valid.jsonl
        └── train
            └── location_train.json
```

The **OPI_DATA** folder contains 9 protein tasks seperately. If you want to merge all or several 'train.json' files of the nine tasks into one single file, please do like this:
```
cd OPI_DATA
python merge_nine_opi_tasks_train.py --output OPI_merged.json
```
You can access the whole dataset file [OPI_full_1.46M.json](https://drive.google.com/file/d/1lKh69Bxxu3cusm9oSpDr27bLr0HjXxRn/view?usp=drive_link) via Google Drive, which contain 1.46 million examples.

## OPI-instruction tuning from original Galactica-6.7B model and LLaMA-7B model
For OPI-instruction tuning, we adopt the training script of [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). 

### 1. Galactica instruction-tuning with OPI

[Example: train_keywords.sh](./train_galai/train_keywords.sh)
```
#!/bin/bash

OMP_NUM_THREADS=1 torchrun --nnodes=$1 --node_rank=$2 --nproc_per_node=3 train_galai/train.py \
    --model_name_or_path path/to/galactica_base_model/galactica-$3 \
    --data_path  ./OPI_DATA/AP/Keywords/train/keywords_train.json \
    --bf16 True \
    --output_dir path/to/output/galai_ft_opi/galai_ft_keywords_$3_e$4 \
    --num_train_epochs $4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
```

To start, please do like this:
```
bash train_galai/train_keywords.sh 1 0 6.7b 3 
```

Explanation of such bash arguments:
```
1: nnodes \
0: node_rank \
6.7b: model size of Galactica \
3: total training epochs
```

### 2. LLaMA instruction-tuning with OPI

[Example: train_EC_number.sh](./train_llama/train_EC_number.sh)
```
#!/bin/bash

OMP_NUM_THREADS=1 torchrun --nnodes=$1 --node_rank=$2 --nproc_per_node=3 train_llama/train.py \
    --model_name_or_path path/to/llama_base_model/hf_version/llama-$3 \
    --data_path  ./OPI_DATA/SU/EC_number/train/CLEAN_EC_number_train.json \
    --bf16 True \
    --output_dir path/to/output/llama_ft_CLEAN_EC_number_$3_e$4 \
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
```

To start, please do like this:
```
bash train_llama/train_EC_number.sh 1 0 7b 3 
```

Explanation of such bash arguments:
```
1: nnodes \
0: node_rank \
7b: model size of LLaMA \
3: total training epochs
```

**Note**: As for the training, we take the suggestion to address out-of-memory issue from [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), using DeepSpeed ZeRO stage-3 with offload.

### 3. Convert DeepSpeed-format weights
Once finished instruction tuning, the DeepSpeed-format weights should be converted to **pytorch_model.bin**, using the following script:
```
cd output_dir
python zero_to_fp32.py . pytorch_model.bin
```

### 4. Split pytorch_model.bin into chunks to speedup loading for inference
After step 3, you will get the **pytorch_model.bin** file. You can further split it to small chunks, e.g., pytorch_model-00001-of-00004.bin
pytorch_model-00002-of-00004.bin, pytorch_model-00003-of-00004.bin, pytorch_model-00004-of-00004.bin, in order to speedup loading it when inferenceing. However, it is not a must, if you don't want. If you would like to split it, please do like this:
```
cd model_split
python model_split.py --model_idx OPI-instruction-tuned-model-name
```
Then you will get a checkpoint folder suffixed with "**chunked**", which you can take as the **pretrained model path** for later evaluation job.

### 5. How to access OPI-instruction-tuned Galactica-6.7B model?
In this repo, we release the OPI_full_Galactica-6.7B model which is fine-funed on OPI full dataset, which can be accessed from [HuggingFace](...). Please feel free to contact us if there is any question.

## Nine Evaluation tasks

For benchamarking, we design 3 types of evaluation tasks, each of which contains 3 specific ones, as shown in the following table.

|       Task Type        | Abbreviation |                  Task Name                  |
| :--------------------: | :----------: | :-----------------------------------------: |
| Sequence Understanding |      SU      |            EC Number Prediction             |
| Sequence Understanding |      SU      |            Fold Type Prediction             |
| Sequence Understanding |      SU      |     Subcellular Localization Prediction     |
| Annotation Prediction  |      AP      |        Function Keywords Prediction         |
| Annotation Prediction  |      AP      |     Gene Ontology(GO) Terms Prediction      |
| Annotation Prediction  |      AP      |       Function Description Prediction       |
|    Knowledge Mining    |      KM      | Tissue Location Prediction from Gene Symbol |
|    Knowledge Mining    |      KM      |     Cancer Prediction from Gene Symbol      |
|    Knowledge Mining    |      KM      |      Cancer Prediction from Gene Name       |

## Evaluating various models with OPI data
For the evaluation script, we refer to the inference script from [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca).

### 1. Evaluation of Galactica
We evaluate OPI-instruction-tuned Galactica-6.7B model and origional Galactica-6.7B model.

**For OPI-instruction-tuned Galactica-6.7B model, please use the following script:**
```
cd eval_galai
python eval_galai.py --model_idx OPI-instruction-tuned-model-name --gpus=0
```

**For the original Galactica-6.7B model, please use the following script:**
```
cd eval_galai/infer_with_original_galai
bash galactica_infer.sh
```

### 2. Evaluation of Alpaca
For comparison, we evaluate Alpaca-7B model and [Galpaca-6.7B](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-6.7b) model. The Galpaca-6.7B model is contributed by Georgia Tech Research Institute on HuggingFace.

As for Alpaca-7B model, we first get [alpaca-7b-wdiff](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff) from HuggingFace, which is the weight diff for [Stanford Alpaca-7B](https://github.com/tatsu-lab/stanford_alpaca/), then recover the original Alpaca-7B weights using the conversion script provided by [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca).

The same script is used for evaluating Alpaca-7B and Galpaca-6.7B model, just by setting a different model_idx for a different model.
```
cd eval_alpaca
python eval_alpaca.py --model_idx alpaca-7b-recover --gpus=0 #original Alpaca-7B weights
```

### 3. Evaluation of LLaMA
For comparison, we evaluate OPI-instruction-tuned LLaMA-7B model and original LLaMA-7B model.

The same script is used for evaluating OPI-instruction-tuned LLaMA-7B model and original LLaMA-7B model, just by setting a different model_idx for a different model.
```
cd eval_llama
python eval_llama.py --model_idx llama_7b_hf --gpus=0  #original LLaMA-7B weights
```

### 4. The following table shows evaluation results of OPI_full_Galactica-6.7B model on 9 tasks.
| Task Type              | Task Name                                   | Testing file                  | Accuracy | Precision | Recall |  F1   | Rouge-L |
| ---------------------- | ------------------------------------------- | ----------------------------- | :------: | :-------: | :----: | :---: | :-----: |
| Sequence Understanding | EC Number Prediction                        | CLEAN_EC_number_new_test      |    -     |   0.181   | 0.174  | 0.176 |    -    |
| Sequence Understanding | EC Number Prediction                        | CLEAN_EC_number_price_test    |    -     |   0.054   | 0.054  | 0.054 |    -    |
| Sequence Understanding | Fold Type Prediction                        | Remote_valid_fold             |  0.068   |     -     |   -    |   -   |    -    |
| Sequence Understanding | Fold Type Prediction                        | Remote_valid_superfamily      |  0.090   |     -     |   -    |   -   |    -    |
| Sequence Understanding | Fold Type Prediction                        | Remote_valid_family           |  0.416   |     -     |   -    |   -   |    -    |
| Sequence Understanding | Subcellular Localization Prediction         | location_valid                |  0.678   |     -     |   -    |   -   |    -    |
| Annotation Prediction  | Function Keywords Prediction                | CASPSimilarSeq_keywords_valid |    -     |   0.716   | 0.669  | 0.674 |    -    |
| Annotation Prediction  | Function Keywords Prediction                | IDFilterSeq_keywords_valid    |    -     |   0.822   | 0.771  | 0.778 |    -    |
| Annotation Prediction  | Function Keywords Prediction                | UniProtSeq_keywords_valid     |    -     |   0.871   | 0.802  | 0.820 |    -    |
| Annotation Prediction  | Gene Ontology(GO) Terms Prediction          | CASPSimilarSeq_go_valid       |    -     |   0.710   | 0.627  | 0.647 |    -    |
| Annotation Prediction  | Gene Ontology(GO) Terms Prediction          | IDFilterSeq_go_valid          |    -     |   0.724   | 0.637  | 0.656 |    -    |
| Annotation Prediction  | Gene Ontology(GO) Terms Prediction          | UniProtSeq_go_valid           |    -     |   0.759   | 0.683  | 0.698 |    -    |
| Annotation Prediction  | Function Description Prediction             | CASPSimilarSeq_function_valid |    -     |     -     |   -    |   -   |  0.431  |
| Annotation Prediction  | Function Description Prediction             | IDFilterSeq_function_valid    |    -     |     -     |   -    |   -   |  0.624  |
| Annotation Prediction  | Function Description Prediction             | UniProtSeq_function_valid     |    -     |     -     |   -    |   -   |  0.696  |
| Knowledge Mining       | Tissue Location Prediction from Gene Symbol | gene_symbol_to_tissue_valid   |    -     |   0.377   | 0.779  | 0.468 |    -    |
| Knowledge Mining       | Cancer Prediction from Gene Symbol          | gene_symbol_to_cancer_test    |    -     |   0.554   | 0.433  | 0.465 |    -    |
| Knowledge Mining       | Cancer Prediction from Gene Name            | gene_name_to_cancer_test      |    -     |   0.507   | 0.400  | 0.429 |    -    |

## Prediction (by OPI_full_Galactica-6.7B) v.s. Target

<details>
<summary>EC Number Prediction</summary>

```
Instruction:
    What is the EC number of the input sequence?
Input:
    MSLLAYTNLLLQNGRIFRYYKKANIKKFIKKIIKLDLKSTPSEASVSRQTFLSTGLNSVKNAVQLQARKLLINNVLERVTPTLNSDLKKKAAKRLFYGDSAPFFALVGVSLASGSGLLTKDDELEGICWEIREAVSKGKWNDSESENVEQLQAANLDELDLGEPIAKGCNAVVYSAKLKNVQSNKLAHQLAVKMMFNYDVESNSTAILKAMYRETVPAMSYFFNQNLFNIENISDFKIRLPPHPNIVRMYSVFADRIPDLQCNKQLYPEALPPRINPEGSGRNMSLFLVMKRYDCTLKEYLRDKTPNMRSSILLLSQLLEAVAHMNIHNISHRDLKSDNILVDLSEGDAYPTIVITDFGCCLCDKQNGLVIPYRSEDQDKGGNRALMAPEIANAKPGTFSWLNYKKSDLWAVGAIAYEIFNIDNPFYDKTMKLLSKSYKEEDLPELPDTIPFIIRNLVSNMLSRSTNKRLDCDVAATVAQLYLWAPSSWLKENYTLPNSNEIIQWLLCLSSKVLCERDITARNKTNTMSESVSKAQYKGRRSLPEYELIASFLRRVRLHLVRKGLKWIQELHIYN
Prediction:
    2.7.11.1
Target:
    2.7.11.1
```

</details>

<details>
<summary>Fold Type Prediction</summary>

```
Instruction:
    Please predict its folding type based on the protein sequence. Here, a number is assigned to each folding type, ranging from 0 to 1194.
Input:
    GSGDSHPDFPEDADVDLKDVDKILLISEDLKNIGNTFFKSQNWEMAIKKYTKVLRYVEGSRAAAEDADGAKLQPVALSCVLNIGACKLKMSDWQGAVDSCLEALEIDPSNTKALYRRAQGWQGLKEYDQALADLKKAQEIAPEDKAIQAELLKVKQKIKAQKDKEKAAY
Prediction:
    3
Target:
    3
```

</details>

<details>
<summary>Subcellular Localization Prediction</summary>

```
Instruction:
    By scrutinizing the protein's amino acid composition and sequence motifs, forecast its intracellular localization in eukaryotic cells.
Input:
    MEDEAVLDRGASFLKHVCDEEEVEGHHTIYIGVHVPKSYRRRRRHKRKTGHREKKEKERISENYSDKSDVENADESSSSILKPLISPAAERIRFILGEEDDSPAPPQLFTELDELLAVDGQEMEWKETARWIKFEEKVEQGGERWSKPHVATLSLHSLFELRTCMEKGSIMLDREASSLPQLVEMIVDHQIETGLLKPDLKDKVTYTLLRKHRHQTKKSNLRSLADIGKTVSSASRMFTNPDNGSPAMTHRNLTSSSLNDISDKPEKDQLKNKFMKKLPRDAEASNVLVGEVDFLDSPFIAFVRLQQAVMLGALTEVPVPTRFLFILLGPKGKAKSYHEIGRAIATLMSDEVFHDIAYKAKDRQDLIAGIDEFLDEVIVLPPGEWDPAIRIEPPKSLPSSDKRKNMYSGGENVQMNGDTPPDGGHGGGGHADCEELQRTGRFCGGLIKDIKRKAPFFASDFYDALNIQALSAILFIYLATVTNAITFGGLLGDATDNMQGVLESFLGTAVSGAIFCLFAGQPLTILSSTGPVLVFERLLFNFSKDHNFDYLEFRLWIGLWSAFLCLILVATDASFLVQYFTRFTEEGFSSLISFIFIYDAFKKMIKLADYYPINSNFKVGYNTQFSCVCMPPDPVNISVSNDTTLAPEDLPTISSSNMYHNATFDWAFLTTKECLKYGGKLVGNNCGFVPDITLMSFILFLGTYTSSMALKKFKTSPYFPTTARKLISDFAIILPILIFCVIDALVGVDTPKLIVPSEFKPTSPNRGWFVAPFGGNPWWVYLAAAIPALLVTILIFMDQQITAVIVNRKEHKLKKGAGYHLDLFWVAILMVVCSFMALPWYVAATVISIAHIDSLKMETETSAPGEQPKFLGVREQRVTGTLVFILTGLSVFMAPILKFIPMPVLYGVFLYMGVASLNGVQFMDRLKLLLMPLKHQPDFIYLRHVPLRRVHLFTFLQVLCLALLWILKSTVAAIIFPVMILALVAVRKGMDYLFSQHDLSFLDDVIPEKDKKKKEDEKKKKKKKGSVDSDNDDSDCPYSEKVPSIKIPMDIMEQQPFLSDSKPSDRERSPTFLERHTSC
Prediction:
    membrane
Target:
    membrane
```

</details>

<details>
<summary>Function Keywords Prediction</summary>

```
Instruction:
    What are the UniProtKB keywords for this specific protein sequence?
Input:
    MRGSFFSRLPPQLSLLLLLLLLLSWRRVWTQEHIGTDPSKSPVAPVCPEACSCSPGGKANCSALALPAVPAGLSWQVRSLLLDRNRVSTLPPGAFADAGALLYLVLRENRLRSVHARAFWGLGVLQRLDLSSNQLETLSPGTFTPLRALSFLSLAGNRLALLEPSILGPLPLLRVLSLQDNSLSALEAGLLNSLPALDVLRLHGNPWACSCALRPLCTWLRKHPRPTSETETLLCVSPKLQTLNLLTDFPDNAFKQCTQSLAARDLAVVYALGPASFLASLAICLALGSVLTACGARRRRRRTTVRHLIRRQPDPEGPASLEDVGSPTTTAIQA
Prediction:
    Cell membrane ; Cytoplasm ; Cytoskeleton ; Disulfide bond ; Ion channel ; Ion transport ; Leucine-rich repeat ; Membrane ; Reference proteome ; Repeat ; Signal ; Transmembrane ; Transmembrane helix ; Transport
Target:
    Cell membrane ; Cytoplasm ; Cytoskeleton ; Disulfide bond ; Ion channel ; Ion transport ; Leucine-rich repeat ; Membrane ; Reference proteome ; Repeat ; Signal ; Transmembrane ; Transmembrane helix ; Transport
```

</details>

<details>
<summary>Gene Ontology(GO) Terms Prediction</summary>

```
Instruction:
    The Gene Ontology project (GO) provides a controlled vocabulary to describe gene and gene product attributes in any organism. There are 3 disjoint categories: cellular component, molecular function and biological process. Predict the GO term for a given protein sequence.
Input:
    MEFVTNYTLEELKKRFTELGLEPYRAKQVFRWVYKKFVTDFEKMTDLGKKHRELLKEHFAFHPLEKLDRVEAPDAVKYLFKTKDGHILETVLIKERDHYTLCVSSQIGCAVGCTFCATALDGLKRNLSTAEIIDQYLQVQQDLGEEKIRNVVFMGMGEPLANYENVRKAVEIMVSPEGLDLSKRRITISTSGIVAQIKRMAQDPVMKEVNLAVSLNAVSQKKREELMPLTKTNTLEELMEVLKNYPLPKYRRITLEYVLIKGVNDSPNDAERLAKLIGRHKKKFKVNLIPFNPDPNLPYERPALTDIMKFQKVLWKYGISNFVRFSKGVEVFGACGQLRTQRLQLQRV
Prediction:
    cytoplasm ; 4 iron, 4 sulfur cluster binding ; metal ion binding ; rRNA (adenine-C2-)-methyltransferase activity ; rRNA binding ; tRNA (adenine-C2-)-methyltransferase activity ; tRNA binding ; rRNA base methylation 
Target:
    cytoplasm ; 4 iron, 4 sulfur cluster binding ; metal ion binding ; rRNA (adenine-C2-)-methyltransferase activity ; rRNA binding ; tRNA (adenine-C2-)-methyltransferase activity ; tRNA binding ; rRNA base methylation ; tRNA methylation
```

</details>

<details>
<summary>Function Description Prediction</summary>

```
Instruction:
    Generate the functional description in free-text form based on the protein sequence.
Input:
    MAAPSGVHLLVRRGSHRIFSSPLNHIYLHKQSSSQQRRNFFFRRQRDISHSIVLPAAVSSAHPVPKHIKKPDYVTTGIVPDWGDSIEVKNEDQIQGLHQACQLARHVLLLAGKSLKVDMTTEEIDALVHREIISHNAYPSPLGYGGFPKSVCTSVNNVLCHGIPDSRPLQDGDIINIDVTVYYNGYHGDTSETFLVGNVDECGKKLVEVARRCRDEAIAACRAGAPFSVIGNTISHITHQNGFQVCPHFVGHGIGSYFHGHPEIWHHANDSDLPMEEGMAFTIEPIITEGSPEFKVLEDAWTVVSLDNQRSAQFEHTVLITSRGAQILTKLPHEA
Prediction:
    Removes the N-terminal methionine from nascent proteins. The N-terminal methionine is often cleaved when the second residue in the primary sequence is small and uncharged (Met-Ala-, Cys, Gly, Pro, Ser, Thr, or Val). Requires deformylation of the N(alpha)-formylated initiator methionine before it can be hydrolyzed.
Target:
    Removes the N-terminal methionine from nascent proteins. The N-terminal methionine is often cleaved when the second residue in the primary sequence is small and uncharged (Met-Ala-, Cys, Gly, Pro, Ser, Thr, or Val). Requires deformylation of the N(alpha)-formylated initiator methionine before it can be hydrolyzed (By similarity). May play a role in colon tumorigenesis.
```

</details>

<details>
<summary>Tissue Location Prediction from Gene Symbol</summary>

```
Instruction:
    According to the relevant data reported in the scientific literature, the Gene symbol of a given protein can be used to predict which tissues the protein is expressed in.
Input:
    TMEM198
Prediction:
    adrenal gland, appendix, bone marrow, breast, bronchus, caudate, cerebellum, cerebral cortex, cervix, colon, duodenum, endometrium, epididymis, esophagus, fallopian tube, gallbladder, heart muscle, hippocampus, kidney, liver, lung, lymph node, nasopharynx, oral mucosa, ovary, pancreas, parathyroid gland, placenta, prostate, rectum, salivary gland, seminal vesicle, skeletal muscle, skin, small intestine, smooth muscle, soft tissue, spleen, stomach, testis, thyroid gland, tonsil, urinary bladder
Target:
    adrenal gland, appendix, breast, bronchus, cerebral cortex, colon, fallopian tube, gallbladder, kidney, nasopharynx, prostate, rectum, salivary gland, seminal vesicle, skeletal muscle, small intestine, stomach, urinary bladder
```

</details>

<details>
<summary>Cancer Prediction from Gene Symbol</summary>

```
Instruction:
    Return the name of the cancer corresponding to the given gene symbol.
Input:
    FOXL2
Prediction:
    granulosa-cell tumour of the ovary
Target:
    granulosa-cell tumour of the ovary
```

</details>

<details>
<summary>Cancer Prediction from Gene Name</summary>

```
Instruction:
    Give back the cancer name that is associated with the provided gene name.
Input:
    immunoglobulin lambda locus
Prediction:
    Burkitt lymphoma
Target:
    Burkitt lymphoma
```

</details>

## Demo
We use the [FastChat](https://github.com/lm-sys/FastChat) platform to visually demonstrate the ability of OPI_full_Galactica-6.7B model on various evaluation tasks.

![OPI Demo](..)