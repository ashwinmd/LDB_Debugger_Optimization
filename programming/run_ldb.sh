dataset=$1
small_model=$2
big_model=$3
seedfile=$4
output_dir=$5
max_iters=$6
iters_to_run_small=$7
strategy="ldb"
python main.py \
  --run_name $output_dir \
  --root_dir ../output_data/$strategy/$dataset/$model/ \
  --dataset_path ../input_data/$dataset/dataset/probs.jsonl \
  --strategy $strategy \
  --small_model $small_model \
  --big_model $big_model \
  --seedfile $seedfile \
  --pass_at_k "1" \
  --max_iters $max_iters \
  --iters_to_run_small $iters_to_run_small \
  --n_proc "1" \
  --port "8000" \
  --testfile ../input_data/$dataset/test/tests.jsonl \
  --verbose
