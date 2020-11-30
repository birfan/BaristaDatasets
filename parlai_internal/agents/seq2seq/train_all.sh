#!/bin/bash

ds="$1"
task_size="$2"

[ -z "$ds" ] && ds="barista-personalised"
[ -z "$task_size" ] && task_size="Task1k"

if [ "$ds" == "barista" ]; then
  start_task=1
  end_task=7
else
  start_task=0
  end_task=8
fi

#log_folder=/project/Seq2Seq-extended-dict-glove-no-history/$ds/$task_size/log
log_folder=/project/Seq2Seq-extended-dict/$ds/$task_size/log

if [[ ! -d $log_folder ]]; then
  mkdir -p $log_folder
fi

for ((i = start_task ; i <= end_task ; i++)); do
  python3.6 parlai_internal/agents/seq2seq/train.py -ds=$ds -ts=$task_size -tid=$i > $log_folder/task$i.txt
done
