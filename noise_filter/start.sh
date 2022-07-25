version_name="transformer_v1"
dirname="./"
python3 ${dirname}/noise_filter/cal_ppl.py \
  --log_file ${dirname}/logs/${version_name}.ppl.log\
  --config_fp ${dirname}/data/chinese_roberta_wwm_ext/ \
  --test_file ${dirname}/logs/${version_name}.pred \
  --batch_size 4200 \
  --jizhi 1 \
  --n_gpu 6 \

 python3 filter_ppl.py \
  --pred_file ${dirname}/logs/${version_name}.pred \
  --ppl_file ${dirname}/logs/${version_name}.pred.ppl \
  --denoise_file ${dirname}/logs/${version_name}.denoise \

