if which gshuf >/dev/null; then
  shuf_cmd="gshuf"
else
  shuf_cmd="shuf"
fi

function mkdirs {
  dir_name=$1
  task="task"$2
  if [[ ! -d $dir_name ]]; then 
    mkdir -p $dir_name/
  fi
  if [[ ! -d $dir_name/data ]]; then
    mkdir -p $dir_name/data
  fi
  if [[ ! -d $dir_name/log ]]; then
    mkdir -p $dir_name/log
  fi
  if [[ ! -d $dir_name/checkpoints/$task/ ]]; then 
    mkdir -p $dir_name/checkpoints/$task/
  fi
}

function train {
  dir_name=$1
  task="task"$2

  python3.6 train.py --train $dir_name/data/train-$task.tsv --dev $dir_name/data/dev-$task.tsv \
  --vocab $dir_name/data/vocab-$task.tsv --save_dir $dir_name/checkpoints/$task/model \
  --candidates $dir_name/data/candidates.tsv --task $task
}

function parse_candidates {
  ds="$1"
  dir_name="$2"

  task_size="$3"
  [ -z "$task_size" ] && task_size="Task1k"

  order_info=$4
  [ -z "$order_info" ] && order_info=False

  ds_t="$5"
  [ -z "$ds_t" ]  && ds_t=""
  
  ds_format="$6"
  [ -z "$ds_format" ]  && ds_format=""

  response_type="$7"
  [ -z "$response_type" ]  && response_type=""

  if [[ $ds =~ "-order" ]]; then
    order_info=True
  elif [[ "$order_info" == "True" ]]; then
    ds=$3"-order"
  fi

  base_path="../../data/$ds/$ds-dataset/$response_type/$ds_t/$ds_format/$task_size"
  python3.6 parse_candidates.py $base_path/$ds-candidates.txt > $dir_name/data/candidates.tsv
}

function parse_dialogs {
  ds="$1"
  task="task""$2"
  additional_options="$3"
  dir_name="$4"

  task_size="$5"
  [ -z "$task_size" ] && task_size="Task1k"

  order_info=$6
  [ -z "$order_info" ] && order_info=False

  ds_t="$7"
  [ -z "$ds_t" ]  && ds_t=""
  
  ds_format="$8"
  [ -z "$ds_format" ]  && ds_format=""

  response_type="$9"
  [ -z "$response_type" ]  && response_type=""

  if [[ $ds =~ "-order" ]]; then
    order_info=True
  elif [[ "$order_info" == "True" ]]; then
    ds=$3"-order"
  fi

  base_path="../../data/$ds/$ds-dataset/$response_type/$ds_t/$ds_format/$task_size/"

  for entry in "$base_path/$task"/*
  do
    if [[ $entry == *'trn.txt' ]]; then
      python3.6 parse_dialogs.py --input $entry $additional_options > $dir_name/data/train-$task.tsv
    elif [[ $entry == *'dev.txt' ]]; then 
      python3.6 parse_dialogs.py --input $entry $additional_options > $dir_name/data/dev-$task.tsv
    elif [[ $entry == *'tst.txt' ]]; then
      python3.6 parse_dialogs.py --input $entry $additional_options > $dir_name/data/test-$task.tsv
    elif [[ $entry == *'tst-OOV.txt' ]]; then
      python3.6 parse_dialogs.py --input $entry $additional_options > $dir_name/data/test-OOV-$task.tsv
    fi
  done
  eval $shuf_cmd -n 500 $dir_name/data/dev-$task.tsv > $dir_name/data/dev-$task-500.tsv
  cat $dir_name/data/train-$task.tsv $dir_name/data/dev-$task.tsv $dir_name/data/test-$task.tsv | python3.6 build_vocabulary.py > $dir_name/data/vocab-$task.tsv
}

