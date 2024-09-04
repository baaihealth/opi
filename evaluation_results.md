## 1. Evaluation of OPI-full-1.61M-Llama-3.1-8B-Instruct model on 9 tasks. 
Each testing result is derived from the model that has been fine-tuned using the complete OPI dataset (i.e.,[OPI_full_1.61M.json](...)) and subsequently evaluated on the respective testing set for each specific task.
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

## 2. Evaluation of OPI-full-1.61M-Galactica-6.7B model on 9 tasks 
Each testing result is derived from the Galactica-6.7B model that has been fine-tuned using the complete OPI dataset (i.e.,[OPI_full_1.61M.json](...)) and subsequently evaluated on the respective testing set for each specific task.

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