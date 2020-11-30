#!/bin/bash

ds="$1"
task_size="$2"
hops=$3
nt=$4

[ -z "$ds" ] && ds="barista-personalised"
[ -z "$task_size" ] && task_size="Task1k"
[ -z "$hops" ] && hops=1
[ -z "$nt" ] && nt=20

if [ "$ds" == "barista" ]; then
  start_task=1
  end_task=7
else
  start_task=0
  end_task=8
fi

for ((i = start_task ; i <= end_task ; i++)); do
  python3.6 parlai_internal/agents/kvmemnn/eval.py -ds=$ds -ts=$task_size -tid=$i --hops=$hops -nt=$nt
done

for ((i = start_task ; i <= end_task ; i++)); do
  python3.6 parlai_internal/agents/kvmemnn/eval_OOV.py -ds=$ds -ts=$task_size -tid=$i --hops=$hops -nt=$nt
done
