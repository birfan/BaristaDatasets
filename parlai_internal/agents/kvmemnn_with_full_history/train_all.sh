#!/bin/bash

ds="$1"
task_size="$2"
hops=$3
nt=$4
eps=$5

[ -z "$ds" ] && ds="barista-personalised"
[ -z "$task_size" ] && task_size="Task1k"
[ -z "$hops" ] && hops=1
[ -z "$nt" ] && nt=20
[ -z "$eps" ] && eps=25


if [ "$ds" == "barista" ]; then
  start_task=1
  end_task=7
else
  start_task=0
  end_task=8
fi
log_folder=/project/KVMemNN-extended-dict/$ds/$task_size/hop$hops/log

if [[ ! -d $log_folder ]]; then
  mkdir -p $log_folder
fi

for ((i = start_task ; i <= end_task ; i++)); do
  python3.6 parlai_internal/agents/kvmemnn/train.py -ds=$ds -ts=$task_size -tid=$i --hops=$hops -nt=$nt -eps=$eps > $log_folder/task$i.txt
done
