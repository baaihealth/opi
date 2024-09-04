## 1. Evaluation of OPI-Llama-3.1-8B-Instruct models on 9 tasks (individual). 
Each testing result reflects the performance of a fine-tuned model, where the model has been specifically trained on the respective task’s training set and subsequently evaluated on its corresponding testing set.
<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center;">Task Type</th>
      <th style="text-align:center;">Task Name</th>
      <th style="text-align:center;">Testing file</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Precision</th>
      <th style="text-align:center;">Recall</th>
      <th style="text-align:center;">F1</th>
      <th style="text-align:center;">Rouge-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Sequence Understanding</td>
      <td rowspan="2">EC Number Prediction (split100)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.2013</td>
      <td>0.2013</td>
      <td>0.2013</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.3903</td>
      <td>0.3615</td>
      <td>0.3688</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td>
      <td>0.1100</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td>
      <td>0.1037</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td>
      <td>0.5736</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization Prediction</td>
      <td>subcell_loc_test</td>
      <td>0.3088</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="9">Annotation Prediction</td>
      <td rowspan="3">Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td>
      <td>-</td>
      <td>0.3539</td>
      <td>0.3499</td>
      <td>0.3398</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_keywords_test</td>
      <td>-</td>
      <td>0.6851</td>
      <td>0.6886</td>
      <td>0.6691</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_keywords_test</td>
      <td>-</td>
      <td>0.7647</td>
      <td>0.7640</td>
      <td>0.7481</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Gene Ontology(GO) Terms Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td>
      <td>-</td>
      <td>0.1314</td>
      <td>0.0857</td>
      <td>0.0945</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_go_terms_test</td>
      <td>-</td>
      <td>0.6698</td>
      <td>0.6478</td>
      <td>0.6450</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7415</td>
      <td>0.7244</td>
      <td>0.7172</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Function Description Prediction</td>
      <td>CASPSimilarSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7217</td>
    </tr>
    <tr>
      <td>IDFilterSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.4691</td>
    </tr>
    <tr>
      <td>UniProtSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.5614</td>
    </tr>
    <tr>
      <td rowspan="3">Knowledge Mining</td>
      <td>Tissue Location Prediction from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td>
      <td>-</td>
      <td>0.4034</td>
      <td>0.9193</td>
      <td>0.5392</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td>
      <td>gene_symbol_to_cancer_test</td>
      <td>-</td>
      <td>0.1250</td>
      <td>0.1239</td>
      <td>0.1201</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td>
      <td>gene_name_to_cancer_test</td>
      <td>-</td>
      <td>0.1041</td>
      <td>0.1543</td>
      <td>0.1108</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 2. Evaluation of Llama-3.1-8B-Instruct_OPI_full_1.46M_v2_train_e1 model on 9 tasks (whole). 
Each testing result is derived from the model that has been fine-tuned using the complete OPI dataset (i.e.,[OPI_full_1.46M.json](https://huggingface.co/datasets/BAAI/OPI)) and subsequently evaluated on the respective testing set for each specific task.
<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center;">Task Type</th>
      <th style="text-align:center;">Task Name</th>
      <th style="text-align:center;">Testing file</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Precision</th>
      <th style="text-align:center;">Recall</th>
      <th style="text-align:center;">F1</th>
      <th style="text-align:center;">Rouge-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Sequence Understanding</td>
      <td rowspan="2">EC Number Prediction (split70)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.2768</td>
      <td>0.2673</td>
      <td>0.2697</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.0336</td>
      <td>0.0336</td>
      <td>0.0336</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td>
      <td>0.1170</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td>
      <td>0.1483</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td>
      <td>0.6577</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization Prediction</td>
      <td>subcell_loc_test</td>
      <td>0.4214</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="9">Annotation Prediction</td>
      <td rowspan="3">Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td>
      <td>-</td>
      <td>0.4288</td>
      <td>0.4887</td>
      <td>0.4341</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_keywords_test</td>
      <td>-</td>
      <td>0.6739</td>
      <td>0.6816</td>
      <td>0.6601</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_keywords_test</td>
      <td>-</td>
      <td>0.7554</td>
      <td>0.7407</td>
      <td>0.731</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Gene Ontology(GO) Terms Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td>
      <td>-</td>
      <td>0.1829</td>
      <td>0.1227</td>
      <td>0.1433</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_go_terms_test</td>
      <td>-</td>
      <td>0.6639</td>
      <td>0.6371</td>
      <td>0.6372</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7147</td>
      <td>0.6893</td>
      <td>0.6849</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Function Description Prediction</td>
      <td>CASPSimilarSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7617</td>
    </tr>
    <tr>
      <td>IDFilterSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.4559</td>
    </tr>
    <tr>
      <td>UniProtSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.5218</td>
    </tr>
    <tr>
      <td rowspan="3">Knowledge Mining</td>
      <td>Tissue Location Prediction from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td>
      <td>-</td>
      <td>0.4016</td>
      <td>0.9189</td>
      <td>0.5391</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td>
      <td>gene_symbol_to_cancer_test</td>
      <td>-</td>
      <td>0.3467</td>
      <td>0.3000</td>
      <td>0.3114</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td>
      <td>gene_name_to_cancer_test</td>
      <td>-</td>
      <td>0.2863</td>
      <td>0.2461</td>
      <td>0.2553</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 3. Evaluation of Llama-3.1-8B-Instruct_OPI_full_1.61M_v2_train_e1 model on 9 tasks (whole). 
Each testing result is derived from the model that has been fine-tuned using the complete OPI dataset (i.e.,[OPI_full_1.61M.json](https://huggingface.co/datasets/BAAI/OPI)) and subsequently evaluated on the respective testing set for each specific task.
<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center;">Task Type</th>
      <th style="text-align:center;">Task Name</th>
      <th style="text-align:center;">Testing file</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Precision</th>
      <th style="text-align:center;">Recall</th>
      <th style="text-align:center;">F1</th>
      <th style="text-align:center;">Rouge-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Sequence Understanding</td>
      <td rowspan="2">EC Number Prediction (split100)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.3724</td>
      <td>0.3374</td>
      <td>0.3468</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.0738</td>
      <td>0.0738</td>
      <td>0.0738</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td>
      <td>0.1045</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td>
      <td>0.1507</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td>
      <td>0.6145</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization Prediction</td>
      <td>subcell_loc_test</td>
      <td>0.4214</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="9">Annotation Prediction</td>
      <td rowspan="3">Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td>
      <td>-</td>
      <td>0.4202</td>
      <td>0.5057</td>
      <td>0.4385</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_keywords_test</td>
      <td>-</td>
      <td>0.6762</td>
      <td>0.6905</td>
      <td>0.6650</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_keywords_test</td>
      <td>-</td>
      <td>0.7606</td>
      <td>0.7489</td>
      <td>0.7374</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Gene Ontology(GO) Terms Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td>
      <td>-</td>
      <td>0.1113</td>
      <td>0.0936</td>
      <td>0.099</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_go_terms_test</td>
      <td>-</td>
      <td>0.6686</td>
      <td>0.6287</td>
      <td>0.6304</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7150</td>
      <td>0.6897</td>
      <td>0.6849</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Function Description Prediction</td>
      <td>CASPSimilarSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7524</td>
    </tr>
    <tr>
      <td>IDFilterSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.4786</td>
    </tr>
    <tr>
      <td>UniProtSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.5144</td>
    </tr>
    <tr>
      <td rowspan="3">Knowledge Mining</td>
      <td>Tissue Location Prediction from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td>
      <td>-</td>
      <td>0.4002</td>
      <td>0.9356</td>
      <td>0.5466</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td>
      <td>gene_symbol_to_cancer_test</td>
      <td>-</td>
      <td>0.2890</td>
      <td>0.2701</td>
      <td>0.2664</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td>
      <td>gene_name_to_cancer_test</td>
      <td>-</td>
      <td>0.2786</td>
      <td>0.2707</td>
      <td>0.2659</td>
      <td>-</td>
    </tr>
  </tbody>
</table>



## 4. Evaluation of OPI-Galactica-6.7B models on 9 tasks (individual).
Each testing result reflects the performance of a fine-tuned model, where the model has been specifically trained on the respective task’s training set and subsequently evaluated on its corresponding testing set.
<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center;">Task Type</th>
      <th style="text-align:center;">Task Name</th>
      <th style="text-align:center;">Testing file</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Precision</th>
      <th style="text-align:center;">Recall</th>
      <th style="text-align:center;">F1</th>
      <th style="text-align:center;">Rouge-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">Sequence Understanding</td>
      <td rowspan="2">EC Number Prediction (split100)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.3316</td>
      <td>0.3203</td>
      <td>0.3213</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.1007</td>
      <td>0.1007</td>
      <td>0.1007</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2">EC Number Prediction (split70)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.2377</td>
      <td>0.2337</td>
      <td>0.2332</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.0805</td>
      <td>0.0805</td>
      <td>0.0805</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td>
      <td>0.0836</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td>
      <td>0.0941</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td>
      <td>0.5303</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization Prediction</td>
      <td>subcell_loc_test</td>
      <td>0.6086</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="9">Annotation Prediction</td>
      <td rowspan="3">Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td>
      <td>-</td>
      <td>0.7592</td>
      <td>0.7839</td>
      <td>0.7563</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_keywords_test</td>
      <td>-</td>
      <td>0.8480</td>
      <td>0.8328</td>
      <td>0.8285</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_keywords_test</td>
      <td>-</td>
      <td>0.8863</td>
      <td>0.8615</td>
      <td>0.8620</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Gene Ontology(GO) Terms Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7757</td>
      <td>0.7481</td>
      <td>0.7533</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7605</td>
      <td>0.7428</td>
      <td>0.7373</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_go_terms_test</td>
      <td>-</td>
      <td>0.8058</td>
      <td>0.7839</td>
      <td>0.7810</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Function Description Prediction</td>
      <td>CASPSimilarSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.6728</td>
    </tr>
    <tr>
      <td>IDFilterSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7153</td>
    </tr>
    <tr>
      <td>UniProtSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7734</td>
    </tr>
    <tr>
      <td rowspan="3">Knowledge Mining</td>
      <td>Tissue Location Prediction from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td>
      <td>-</td>
      <td>0.3917</td>
      <td>0.8096</td>
      <td>0.4911</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td>
      <td>gene_symbol_to_cancer_test</td>
      <td>-</td>
      <td>0.6773</td>
      <td>0.6299</td>
      <td>0.6389</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td>
      <td>gene_name_to_cancer_test</td>
      <td>-</td>
      <td>0.6582</td>
      <td>0.5952</td>
      <td>0.6016</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 5. Evaluation of Galactica-6.7B-OPI_full_1.46M_train_e3 model on 9 tasks (whole). 
Each testing result is derived from the model that has been fine-tuned using the complete OPI dataset (i.e.,[OPI_full_1.46M.json](https://huggingface.co/datasets/BAAI/OPI)) and subsequently evaluated on the respective testing set for each specific task.
<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center;">Task Type</th>
      <th style="text-align:center;">Task Name</th>
      <th style="text-align:center;">Testing file</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Precision</th>
      <th style="text-align:center;">Recall</th>
      <th style="text-align:center;">F1</th>
      <th style="text-align:center;">Rouge-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Sequence Understanding</td>
      <td rowspan="2">EC Number Prediction (split70)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.1786</td>
      <td>0.1739</td>
      <td>0.1754</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.0537</td>
      <td>0.0537</td>
      <td>0.0537</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td>
      <td>0.0710</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td>
      <td>0.0901</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td>
      <td>0.4170</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization Prediction</td>
      <td>subcell_loc_test</td>
      <td>0.6631</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="9">Annotation Prediction</td>
      <td rowspan="3">Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td>
      <td>-</td>
      <td>0.6875</td>
      <td>0.6407</td>
      <td>0.6497</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_keywords_test</td>
      <td>-</td>
      <td>0.8236</td>
      <td>0.7752</td>
      <td>0.7809</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_keywords_test</td>
      <td>-</td>
      <td>0.8718</td>
      <td>0.8015</td>
      <td>0.8205</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Gene Ontology(GO) Terms Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td>
      <td>-</td>
      <td>0.6581</td>
      <td>0.6200</td>
      <td>0.6305</td>
      <td>-</td>
    </tr>
    <tr>
      <td>IDFilterSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7245</td>
      <td>0.6407</td>
      <td>0.6590</td>
      <td>-</td>
    </tr>
    <tr>
      <td>UniProtSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7589</td>
      <td>0.6831</td>
      <td>0.6985</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Function Description Prediction</td>
      <td>CASPSimilarSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.4361</td>
    </tr>
    <tr>
      <td>IDFilterSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.6178</td>
    </tr>
    <tr>
      <td>UniProtSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.6925</td>
    </tr>
    <tr>
      <td rowspan="3">Knowledge Mining</td>
      <td>Tissue Location Prediction from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td>
      <td>-</td>
      <td>0.3772</td>
      <td>0.7801</td>
      <td>0.4684</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td>
      <td>gene_symbol_to_cancer_test</td>
      <td>-</td>
      <td>0.5608</td>
      <td>0.4429</td>
      <td>0.4756</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td>
      <td>gene_name_to_cancer_test</td>
      <td>-</td>
      <td>0.4932</td>
      <td>0.3902</td>
      <td>0.4190</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 6. Evaluation of Galactica-6.7B-OPI_full_1.61M_v2_train_e1 model on 9 tasks (whole). 
Each testing result is derived from the model that has been fine-tuned using the complete OPI dataset (i.e.,[OPI_full_1.61M.json](https://huggingface.co/datasets/BAAI/OPI)) and subsequently evaluated on the respective testing set for each specific task.

<table border="1" style="text-align:center; border-collapse:collapse; width: 100%;">
  <thead>
    <tr>
      <th style="text-align:center;">Task Type</th>
      <th style="text-align:center;">Task Name</th>
      <th style="text-align:center;">Testing file</th>
      <th style="text-align:center;">Accuracy</th>
      <th style="text-align:center;">Precision</th>
      <th style="text-align:center;">Recall</th>
      <th style="text-align:center;">F1</th>
      <th style="text-align:center;">Rouge-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Sequence Understanding</td>
      <td rowspan="2">EC Number Prediction (split100)</td>
      <td>CLEAN_EC_number_new_test</td>
      <td>-</td>
      <td>0.2700</td>
      <td>0.2663</td>
      <td>0.2596</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLEAN_EC_number_price_test</td>
      <td>-</td>
      <td>0.0268</td>
      <td>0.0268</td>
      <td>0.0268</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3">Fold Type Prediction</td>
      <td>fold_type_test_Fold_Holdout</td>
      <td>0.0808</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Superfamily_Holdout</td>
      <td>0.1348</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fold_type_test_Family_Holdout</td>
      <td>0.4854</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Subcellular Localization Prediction</td>
      <td>subcell_loc_test</td>
      <td>0.7771</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="9">Annotation Prediction</td>
      <td>Function Keywords Prediction</td>
      <td>CASPSimilarSeq_keywords_test</td>
      <td>-</td>
      <td>0.8120</td>
      <td>0.7360</td>
      <td>0.7643</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Function Keywords Prediction</td>
      <td>IDFilterSeq_keywords_test</td>
      <td>-</td>
      <td>0.8377</td>
      <td>0.8019</td>
      <td>0.8070</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Function Keywords Prediction</td>
      <td>UniProtSeq_keywords_test</td>
      <td>-</td>
      <td>0.8596</td>
      <td>0.8196</td>
      <td>0.8276</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Gene Ontology (GO) Terms Prediction</td>
      <td>CASPSimilarSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7613</td>
      <td>0.7492</td>
      <td>0.7476</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Gene Ontology (GO) Terms Prediction</td>
      <td>IDFilterSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7404</td>
      <td>0.7274</td>
      <td>0.7207</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Gene Ontology (GO) Terms Prediction</td>
      <td>UniProtSeq_go_terms_test</td>
      <td>-</td>
      <td>0.7638</td>
      <td>0.7373</td>
      <td>0.7358</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Function Description Prediction</td>
      <td>CASPSimilarSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7430</td>
    </tr>
    <tr>
      <td>Function Description Prediction</td>
      <td>IDFilterSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7014</td>
    </tr>
    <tr>
      <td>Function Description Prediction</td>
      <td>UniProtSeq_function_test</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.7133</td>
    </tr>
    <tr>
      <td rowspan="3">Knowledge Mining</td>
      <td>Tissue Location Prediction from Gene Symbol</td>
      <td>gene_symbol_to_tissue_test</td>
      <td>-</td>
      <td>0.3917</td>
      <td>0.9077</td>
      <td>0.5303</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Symbol</td>
      <td>gene_symbol_to_cancer_test</td>
      <td>-</td>
      <td>0.3555</td>
      <td>0.3189</td>
      <td>0.3229</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Cancer Prediction from Gene Name</td>
      <td>gene_name_to_cancer_test</td>
      <td>-</td>
      <td>0.2728</td>
      <td>0.2554</td>
      <td>0.2533</td>
      <td>-</td>
    </tr>
  </tbody>
</table>