
function train {
  task=$1
  ds_name="$2"
  task_size="$3"
  hops=$4
  order_info=$5
  ds_t="$6"
  ds_format="$7"
  response_type="$8"

  [ -z "$hops" ] && hops=1

  [ -z "$task_size" ] && task_size="Task1k"

  [ -z "$order_info" ] && order_info=False

  [ -z "$ds_t" ]  && ds_t=""

  [ -z "$ds_format" ]  && ds_format=""

  [ -z "$response_type" ]  && response_type=""

  if [[ $ds_name =~ "-order" ]]; then
    order_info=True
  elif [[ "$order_info" == "True" ]]; then
    ds_name=$2"-order"
  fi

  python3.6 single_dialog.py --task_id=$task --train=True --hops=$hops --ds_name=$ds_name --task_size=$task_size \
--ds_type=$ds_t --order_info=$order_info --ds_format=$ds_format --response_type=$response_type
}

function evaluate {
  OOV=$1
  task=$2
  ds_name="$3"
  task_size="$4"
  hops=$5
  order_info=$6
  ds_t="$7"
  ds_format="$8"
  response_type="$9"

  [ -z "$hops" ] && hops=1

  [ -z "$task_size" ] && task_size="Task1k"

  [ -z "$order_info" ] && order_info=False

  [ -z "$ds_t" ]  && ds_t=""

  [ -z "$ds_format" ]  && ds_format=""

  [ -z "$response_type" ]  && response_type=""

  if [[ $ds_name =~ "-order" ]]; then
    order_info=True
  elif [[ "$order_info" == "True" ]]; then
    ds_name=$3"-order"
  fi

  if [[ "$OOV" == "True" ]]; then
    dupi="incorrect_predictions-oov.json"
    dupc="correct_predictions-oov.json"
  else
    dupi="incorrect_predictions-test.json"
    dupc="correct_predictions-test.json"
  fi

  python3.6 single_dialog.py --task_id=$task --OOV=$OOV --train=False --hops=$hops --ds_name=$ds_name --task_size=$task_size \
--ds_type=$ds_t --order_info=$order_info --ds_format=$ds_format --response_type=$response_type \
--log_predictions=True --dupi=$dupi --dupc=$dupc

}


