. path

hf_model_name=bert-base-uncased
data_name=$1


IFS=- read hf_model_type the_rest <<< "$hf_model_name"

pretrained_model_dir=${parent_dir}/pretrained_models/${hf_model_name}
base_dir=${parent_dir}
if [ $data_name == "dialdoc" ]; then
    max_seq_len=512
else
    max_seq_len=384
fi

cache_dir=${base_dir}/cache/cls_${hf_model_type}
[ ! -d $pretrained_model_dir ] && echo "$pretrained_model_dir does not exist!" && exit 1
mkdir -p $cache_dir

python gen_data.py \
  --pretrained_model_dir $pretrained_model_dir \
  --input_dir ${base_dir}/data \
  --data_name $data_name \
  --output_dir $cache_dir \
  --max_seq_len $max_seq_len \
  --max_history_len 128 \
  --max_num_spans_per_passage 50 \
  --use_cls_span_start
