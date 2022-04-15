. path

export WORLD_SIZE=2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="10123"
export OMP_NUM_THREADS=1

data_name=$1
checkpoint_path=$3
prediction_results_path=$4

if [ $# -ne 3 ]; then
    echo $0 [data_name] [checkpoint path] [prediction_results_path]
    exit 1
fi

base_dir=${parent_dir}/dialki/$data_name
output_dir=${base_dir}/exp
mkdir -p $output_dir

if [ $data_name == "dialdoc" ]; then
    max_seq_len=512
    passages_per_question=8
    max_answer_length=5
    hist_loss_weight=1.0
else
    max_seq_len=384
    passages_per_question=10
    max_answer_length=1 # one sentence only
    hist_loss_weight=0.5
fi

torchrun \
    --nproc_per_node $WORLD_SIZE \
    --standalone \
    --nnodes=1 \
    train_reader.py \
    --pretrained_model_cfg ${parent_dir}/pretrained_models/bert-base-uncased \
    --checkpoint_file $checkpoint_path \
    --prediction_results_file $prediction_results_path \
    --seed 42 \
    --learning_rate 3e-5 \
    --eval_step 1000 \
    --do_lower_case \
    --eval_top_docs 20 \
    --warmup_steps 1000 \
    --max_seq_len ${max_seq_len} \
    --batch_size 2 \
    --passages_per_question ${passages_per_question} \
    --num_train_epochs 20 \
    --dev_batch_size 4 \
    --max_answer_length ${max_answer_length} \
    --passages_per_question_predict 20 \
    --dev_file ${base_dir}/cache/cls_bert/train \
    --output_dir $output_dir \
    --gradient_accumulation_steps 1 \
    --ignore_token_type \
    --decision_function 1 \
    --hist_loss_weight ${hist_loss_weight} \
    --fp16 \
    --fp16_opt_level O2 \
    --data_name ${data_name} \
    --adv_loss_type js \
    --adv_loss_weight 5 \
    --use_z_attn
