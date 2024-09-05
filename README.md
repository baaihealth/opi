<div align="center">
<img src=./demo_figures/OPI_logo.png />

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/WEIGHT_DIFF_LICENSE)

# <center> OPI: An Open Instruction Dataset for Adapting Large Language Models to Protein-Related Tasks

</div>

**Vision** *Open Protein Instructions(OPI) is the initial part of Open Biology Instructions(OBI) project, together with the subsequent Open Molecule Instructions(OMI), Open DNA Instructions(ODI), Open RNA Instructions(ORI) and Open Single-cell Instructions (OSCI). OBI is a project which aims to fully leverage the potential ability of Large Language Models(LLMs), especially the scientific LLMs like Galactica, to facilitate research in AI for Life Science community. While OBI is still in an early stage, we hope to provide a starting point for the community to bridge LLMs and biological domain knowledge.*

## Contents
- [x] [Project Overview](#project-overview)
- [x] [OPI dataset construction pipeline](#opi-dataset-construction-pipeline)
- [x] [OPI dataset overview](#opi-dataset-overview)
- [x] [OPEval: Nine evaluation tasks using the OPI dataset](#opeval-nine-evaluation-tasks-using-the-opi-dataset)
- [x] [Instruction tuning with OPI training data](#instruction-tuning-with-opi-training-data)
- [x] [Evaluating with OPI testing data](#evaluating-with-opi-testing-data)
- [x] [Evaluation results](#evaluation-results)
- [x] [Prediction comparison with SOTA mdoels](#prediction-comparison-with-sota-mdoels)
- [x] [Demo](#demo)
- [x] [Acknowledgement](#acknowledgement)
- [x] [Contact Information](#contact-information)

## Project Overview
This repo is for the **Open Protein Instructions (OPI)** project, aiming to build and release a high-quality and comprehensive protein instruction dataset with which LLMs can be adapted to protein-related tasks via instruction tuning and evaluated on these tasks.

<div align="center"><img src=./demo_figures/OPI_experiment_outline.png /></div>

**Usage and license notices:** [Galactica](https://github.com/paperswithcode/galai) is intended and licensed for research use only. [Llama-3](https://github.com/meta-llama/llama3) is licensed for researchers and commercial entities, upholding the principles of openness. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The weight diff for [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) is also CC BY NC 4.0 (allowing only non-commercial use).


## OPI dataset construction pipeline
The OPI dataset is curated on our own by extracting key information from [Swiss-Prot](https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue&query=%2A) database. The detailed construction pipeline is depicted in the supplementary material of our manuscript which has been submitted to Foundation Models for Science Workshop (NeurIPS 2024). The following figure shows the overall construction process of OPI.

<div align="center"><img src=./demo_figures/OPI_data.png /></div>


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
- **An example of OPI testing data:**
```
{"id": "seed_task_0", "name": "EC number of price dataset from CLEAN", "instruction": "Return the EC number of the protein sequence.", "instances": [{"input": "MAIPPYPDFRSAAFLRQHLRATMAFYDPVATDASGGQFHFFLDDGTVYNTHTRHLVSATRFVVTHAMLYRTTGEARYQVGMRHALEFLRTAFLDPATGGYAWLIDWQDGRATVQDTTRHCYGMAFVMLAYARAYEAGVPEARVWLAEAFDTAEQHFWQPAAGLYADEASPDWQLTSYRGQNANMHACEAMISAFRATGERRYIERAEQLAQGICQRQAALSDRTHAPAAEGWVWEHFHADWSVDWDYNRHDRSNIFRPWGYQVGHQTEWAKLLLQLDALLPADWHLPCAQRLFDTAVERGWDAEHGGLYYGMAPDGSICDDGKYHWVQAESMAAAAVLAVRTGDARYWQWYDRIWAYCWAHFVDHEHGAWFRILHRDNRNTTREKSNAGKVDYHNMGACYDVLLWALDAPGFSKESRSAALGRP", "output": "5.3.1.7"}], "is_classification": false}
```

## OPI dataset overview 
We are excited to announce the release of the OPI dataset, a curated collection of instructions covering 9 tasks for adatpting LLMs to protein biology. The dataset is designed to advance LLM-driven research in the field of protein biology. We welcome contributions and enhancements to this dataset from the community.

**Accessing the OPI dataset:**
The OPI dataset is organized into the three subfolders—AP, KM, and SU—by in the [OPI_DATA](./OPI_DATA/) directory within this repository, where you can find a seubset for each specific task as well as the full dataset file: [OPI_full_1.61M_train.json](./OPI_DATA/OPI_full_1.61M_train.json). f you want to merge all or several training data files of the tasks into one single training data file, please do like this:
```
cd OPI_DATA
python merge_task_train_data.py --output OPI_merged_train.json
```

**OPI Dataset folder structure:**
```
./OPI_DATA/
└── SU
│   ├── EC_number
│   │   ├── test
│   │   │   ├── CLEAN_EC_number_new_test.jsonl
│   │   │   └── CLEAN_EC_number_price_test.jsonl
│   │   └── train
│   │       ├── CLEAN_EC_number_split_train.json
│   ├── Fold_type
│   │   ├── test
│   │   │   └── fold_type_test.jsonl
│   │   └── train
│   │       └── fold_type_train.json
│   └── Subcellular_localization
│       ├── test
│       │   ├── subcell_loc_test.jsonl
│       └── train
            └── subcell_loc_train.json
├── AP
│   └── Keywords
│   │   ├── test
│   │   │   ├── CASPSimilarSeq_keywords_test.jsonl
│   │   │   ├── IDFilterSeq_keywords_test.jsonl
│   │   │   └── UniProtSeq_keywords_test.jsonl
│   │   └── train
│   │       ├── keywords_train.json
│   ├── GO
│   │   ├── test
│   │   │   ├── CASPSimilarSeq_go_terms_test.jsonl
│   │   │   ├── IDFilterSeq_go_terms_test.jsonl
│   │   │   └── UniProtSeq_go_terms_test.jsonl
│   │   └── train
│   │       ├── go_terms_train.json
│   ├── Function
│       ├── test
│       │   ├── CASPSimilarSeq_function_test.jsonl
│       │   ├── IDFilterSeq_function_test.jsonl
│       │   └── UniProtSeq_function_test.jsonl
│       └── train
│           ├── function_train.json
├── KM
    └── gSymbol2Tissue
    │   ├── test
    │   │   └── gene_symbol_to_tissue_test.jsonl
    │   └── train
    │       └── gene_symbol_to_tissue_train.json
    ├── gSymbol2Cancer
    │   ├── test
    │   │   └── gene_symbol_to_cancer_test.jsonl
    │   └── train
    │       └── gene_symbol_to_cancer_train.json
    ├── gName2Cancer
        ├── test
        │   └── gene_name_to_cancer_test.jsonl
        └── train
            └── gene_name_to_cancer_train.json
```


## OPEval: Nine evaluation tasks using the OPI dataset

To assess the effectiveness of instruction tuning with the OPI dataset, we developed OPEval, which comprises three categories of evaluation tasks. Each category includes three specific tasks. The table below outlines the task types, names, and the corresponding sizes of the training and testing sets.

<table border="1" style="text-align:center; border-collapse:collapse;">
  <tr>
    <th style="text-align:center;">Task Type</th>
    <th style="text-align:center;">Type Abbr.</th>
    <th style="text-align:center;">Task Name</th>
    <th style="text-align:center;">Task Abbr.</th>
    <th style="text-align:center;">Training set size</th>
    <th style="text-align:center;">Testing set size</th>
  </tr>
  <tr>
    <td rowspan="3">Sequence Understanding</td>
    <td rowspan="3">SU</td>
    <td>EC Number Prediction</td>
    <td>EC_number</td>
    <td style="text-align:center;">74,487</td>
    <td style="text-align:center;">392 (NEW-392), 149 (Price-149)</td>
  </tr>
  <tr>
    <td>Fold Type Prediction</td>
    <td>Fold_type</td>
    <td style="text-align:center;">12,312</td>
    <td style="text-align:center;">718 (Fold), 1254 (Superfamily), 1272 (Family)</td>
  </tr>
  <tr>
    <td>Subcellular Localization Prediction</td>
    <td>Subcellular_localization</td>
    <td style="text-align:center;">11,230</td>
    <td style="text-align:center;">2,772</td>
  </tr>
  <tr>
    <td rowspan="3">Annotation Prediction</td>
    <td rowspan="3">AP</td>
    <td>Function Keywords Prediction</td>
    <td>Keywords</td>
    <td style="text-align:center;">451,618</td>
    <td style="text-align:center;">184 (CASPSimilarSeq), 1,112 (IDFilterSeq), 4562 (UniprotSeq)</td>
  </tr>
  <tr>
    <td>Gene Ontology(GO) Terms Prediction</td>
    <td>GO</td>
    <td style="text-align:center;">451,618</td>
    <td style="text-align:center;">184 (CASPSimilarSeq), 1,112 (IDFilterSeq), 4562 (UniprotSeq)</td>
  </tr>
  <tr>
    <td>Function Description Prediction</td>
    <td>Function</td>
    <td style="text-align:center;">451,618</td>
    <td style="text-align:center;">184 (CASPSimilarSeq), 1,112 (IDFilterSeq), 4562 (UniprotSeq)</td>
  </tr>
  <tr>
    <td rowspan="3">Knowledge Mining</td>
    <td rowspan="3">KM</td>
    <td>Tissue Location Prediction from Gene Symbol</td>
    <td>gSymbol2Tissue</td>
    <td style="text-align:center;">8,723</td>
    <td style="text-align:center;">2,181</td>
  </tr>
  <tr>
    <td>Cancer Prediction from Gene Symbol</td>
    <td>gSymbol2Cancer</td>
    <td style="text-align:center;">590</td>
    <td style="text-align:center;">148</td>
  </tr>
  <tr>
    <td>Cancer Prediction from Gene Name</td>
    <td>gName2Cancer</td>
    <td style="text-align:center;">590</td>
    <td style="text-align:center;">148</td>
  </tr>
</table>


## Instruction tuning with OPI training data
Instruction tuning procedures are available in the [instruction_tuning](./instruction_tuning.md) guide.

**Accessing the OPI-Tuned Models:**
We have released the OPI-full-1.61M-Galactica-6.7B and OPI-full-1.61M-Llama-3.1-8B-Instruct models fine-tuned on the complete OPI dataset. You can access it on [Hugging Face](...).

## Evaluating with OPI testing data
Evalution procedures are outlined in the [evaluation](./evaluation.md) guide.

## Evaluation results
Comprehensive evaluation results are detailed in th [evaluation_results](./evaluation_results.md) document.

## Prediction comparison with SOTA mdoels

Prediction by OPI-tuned model, GPT-4o, Llama-3.1-8B-Instruct, Claude 3.5 Sonnet vs. Ground Trurh Answers are shown in in the [model_compare](./model_compare.md) document.

## Demo
We use the [FastChat](https://github.com/lm-sys/FastChat) platform to visually demonstrate the ability of OPI_full_Galactica-6.7B model on various evaluation tasks.

![OPI Demo](./demo_figures/OPI_demo.gif)

## Acknowledgement
The codes are adapted from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).  
Some codes are adapted from [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca).  
Llama-3: [Llama-3](https://github.com/meta-llama/llama3)  
Galactica: [Galactica](https://github.com/paperswithcode/galai)  

## Contact Information
For help or issues using the repos, please submit a GitHub issue.  
For other communications, please contact Qiwei Ye (qwye@baai.ac.cn).