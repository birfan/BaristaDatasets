#!/bin/bash

source bin/utils.sh

ds="$1"
task_size="$2"
hops=$3
[ -z "$ds" ] && ds="barista-personalised"
[ -z "$task_size" ] && task_size="Task1k"
[ -z "$hops" ] && hops=1

order_info=$4
ds_format="$5"
response_type="$6"
ds_type="$7"

ds_set=""
[ -n "$task_size" ] && ds_set=$ds_set$task_size
[ -n "$ds_format" ] && ds_set=$ds_set"-"$ds_format
[ -n "$response_type" ] && ds_set=$ds_set"-"$response_type
[ -n "$ds_type" ] && ds_set=$ds_set"-"$ds_type

dir_name=/project/MemN2N/$ds/$ds_set/hop$hops

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
  evaluate False $i $1 $2 $3 $4 $7 $5 $6 > $dir_name/log/task$i-test.txt
done

for ((i = start_task ; i <= end_task ; i++)); do
  evaluate True $i $1 $2 $3 $4 $7 $5 $6 > $dir_name/log/task$i-OOV.txt
done
