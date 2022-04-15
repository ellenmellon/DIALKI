
dataname=$1
ckpt=$2
output_path=$3

bash scripts/eval.sh $dataname $ckpt $output_path
