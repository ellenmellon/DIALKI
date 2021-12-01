set -e

data_name=$1

python download_hf_model.py --data_name=$data_name

bash scripts/gen_${data_name}_json.sh
bash scripts/gen_data_cls.sh ${data_name}
bash scripts/train.sh ${data_name}
