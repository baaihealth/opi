#Function Keywords Prediction of original galactica model
python galactica_infer.py --input CASPSimilarSeq_test.json --output infer_result.json --task keyword

#GO Terms Prediction of original galactica model
python galactica_infer.py --input CASPSimilarSeq_test.json --output infer_result.json --task GO

#Function Description Prediction of original galactica model
python galactica_infer.py --input CASPSimilarSeq_test.json --output infer_result.json --task function_description

#EC Number Prediction of original galactica model
python galactica_infer.py --input EC_test.json --output infer_result.json --task EC

#Tissue Location Prediction from Gene Symbol of original galactica model
python galactica_infer.py --input input.json --output infer_result.json --task tissue

#Cancer Name Prediction from Gene Symbol of original galactica model
python galactica_infer.py --input input.json --output infer_result.json --task symbol_to_cancer

#Cancer Name Prediction from Gene Name of original galactica model
python galactica_infer.py --input input.json --output infer_result.json --task name_to_cancer