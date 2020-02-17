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

ds_set=""
[ -n "$task_size" ] && ds_set=$ds_set$task_size
[ -n "$ds_format" ] && ds_set=$ds_set"-"$ds_format
[ -n "$response_type" ] && ds_set=$ds_set"-"$response_type
[ -n "$ds_type" ] && ds_set=$ds_set"-"$ds_type

dir_name=/project/supervised-embedding/$ds/$ds_set
[ -z "$ds_set" ] && dir_name=/project/supervised-embedding/$ds

if [ "$ds" == "barista" ]; then
  start_task=1
  end_task=7
else
  start_task=0
  end_task=8
fi

for ((i = start_task ; i <= end_task ; i++)); do
  mkdirs $dir_name $i
  parse_candidates $ds $dir_name $2 $3 $4 $5 $6
  parse_dialogs $ds $i '--with_history' $dir_name $2 $3 $4 $5 $6
  train $dir_name $i > $dir_name/log/task$i.txt
done

