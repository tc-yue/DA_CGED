# This repo provides the Code for DataAugmentationCGED Paper in ACL 2022.

### Pipeline
- 1：Prepare the training data for the error generation model.
  - python3 error_generation/train_data_process.py
- 2：Train the model and predict the errors
  - sh error_generation/train.sh
- 3： Filter non-error generated spans by span-level perplexity
  - sh noise_filter/start.sh
- 4: auto label the generated span by editing method (TBD)
  - python3 auto_label/auto_label.py --mlm_version=${version} --dir_name=${dir_name}
- 5: construct the final training sample (TBD)
  - python3 sample_construction/merge_multi_error.py
- 6: train the detection model with the augmented dataset (TBD) 
  - sh error_detection/train.sh
      
### Dataset
- We don't have the copyright of the dataset. Please contact with the host of the [CGED shared task](https://aclanthology.org/2020.nlptea-1.4/)
- The sample data is in ./data/train_data_process.txt.sample: 

### Requirements
- transformers2.0.0
- RoBERTa-wwm-ext, Chinese. From [repo](https://github.com/ymcui/Chinese-BERT-wwm)