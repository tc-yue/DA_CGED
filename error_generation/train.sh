version_name="transformer_v1"
dirname="./"

filename=${dirname}/logs/${version_name}.all
filename2=${filename}.multi
cat $filename >> $filename2

python3 ${dirname}/error_generation/run_mlm.py \
  --version_name ${version_name} \
  --log_file ${dirname}/logs/ \
  --config_fp ${dirname}/data/chinese_roberta_wwm_ext/ \
  --output_path ${dirname}/logs/ \
  --train_file ${dirname}/data/constrained_train_v4.txt \
  --test_file ${dirname}/data/train_processed.txt \
  --batch_size 160 \
  --val_step 100 \
  --learning_rate 5e-5 \
  --n_gpu 2 \
  --epochs 5 \
  --train_shuffle \
  --train \
  --test \
  --gradient_accumulation_steps 1 \
