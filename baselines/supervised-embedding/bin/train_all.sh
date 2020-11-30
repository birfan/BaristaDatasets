#!/bin/bash

source bin/utils.sh

ds="$1"
[ -z "$ds" ] && ds="barista-personalised"
task_size="$2"
batch_size=$3
order_info=$4
ds_type="$5"
ds_format="$6"
response_type="$7"

[ -z "$task_size" ] && task_size="Task1k"

ds_set=""
[ -n "$task_size" ] && ds_set=$ds_set$task_size
[ -n "$ds_format" ] && ds_set=$ds_set"-"$ds_format
[ -n "$response_type" ] && ds_set=$ds_set"-"$response_type
[ -n "$ds_type" ] && ds_set=$ds_set"-"$ds_type

if [ -z "$batch_size" ]; then
  batch_size=128
fi

dir_name=/project/supervised-embedding-with-oov-dict/$ds/$ds_set

if [ "$ds" == "barista" ]; then
  start_task=1
  end_task=7
else
  start_task=0
  end_task=8
fi

for ((i = start_task ; i <= end_task ; i++)); do
  mkdirs $dir_name $i
  parse_candidates $ds $dir_name $2 $4 $5 $6 $7
  parse_dialogs $ds $i '--with_history' $dir_name $2 $4 $5 $6 $7
  train $dir_name $i $batch_size > $dir_name/log/task$i.txt
done

