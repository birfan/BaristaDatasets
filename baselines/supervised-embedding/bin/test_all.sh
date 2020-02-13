#!/bin/bash

source bin/utils.sh

ds="$1"
[ -z "$ds" ] && ds="barista-personalised"
task_size="$2"
order_info=$3
ds_type="$4"
ds_format="$5"
response_type="$6"

[ -z "$task_size" ] && task_size="Task1k"

if [[ $ds =~ "-order" ]]; then
  order_info=True
elif [[ "$order_info" == "True" ]]; then
  ds=$3"-order"
fi

ds_set=""
[ -n "$task_size" ] && ds_set=$ds_set$task_size
[ -n "$ds_format" ] && ds_set=$ds_set"-"$ds_format
[ -n "$response_type" ] && ds_set=$ds_set"-"$response_type
[ -n "$ds_type" ] && ds_set=$ds_set"-"$ds_type

dir_name=$ds/$ds_set
[ -z "$ds_set" ] && dir_name=$ds

if [ "$ds" == "barista" ]; then
  start_task=1
  end_task=7
else
  start_task=0
  end_task=8
fi

if [[ ! -d $dir_name/log ]]; then
  mkdir -p $dir_name/log
fi

for ((i = start_task ; i <= end_task ; i++)); do
  python3.6 test.py --test $dir_name/data/test-task$i.tsv --candidates $dir_name/data/candidates.tsv \
  --vocab $dir_name/data/vocab-task$i.tsv --checkpoint_dir $dir_name/checkpoints/task$i --emb_dim 32 --task_id $i \
  --log_predictions --dupi="incorrect_predictions-test.json" --dupc="correct_predictions-test.json" \
  --result_file=$dir_name/log/results_test.csv > $dir_name/log/task$i-test.txt
done

for ((i = start_task ; i <= end_task ; i++)); do
  python3.6 test.py --test $dir_name/data/test-OOV-task$i.tsv --candidates $dir_name/data/candidates.tsv \
  --vocab $dir_name/data/vocab-task$i.tsv --checkpoint_dir $dir_name/checkpoints/task$i --emb_dim 32 --OOV --task_id $i \
  --log_predictions --dupi="incorrect_predictions-oov.json" --dupc="correct_predictions-oov.json" \
  --result_file=$dir_name/log/results_test.csv > $dir_name/log/task$i-OOV.txt
done

