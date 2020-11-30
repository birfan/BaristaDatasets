import numpy as np
import os
import csv
import json

def batch_iter(tensor, batch_size, shuffle=False):
    batches_count = tensor.shape[0] // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
        data = tensor[shuffle_indices]
    else:
        data = tensor

    neg_shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
    negative_data = tensor[neg_shuffle_indices]

    for batch_num in range(batches_count):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1)*batch_size, tensor.shape[0])
        yield data[start_index:end_index]


def neg_sampling_iter(tensor, batch_size, count, seed=None):
    batches_count = tensor.shape[0] // batch_size
    trials = 0
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
    data = tensor[shuffle_indices]
    for batch_num in range(batches_count):
        trials += 1
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1)*batch_size, tensor.shape[0])
        if trials > count:
            return
        else:
            yield data[start_index:end_index]

# BI:extra functions

def convert_BOW_sentence(bow, vocab_inverse):
    """Convert Bag of Words representation to sentence"""
    s = ""
    for ind in range(len(bow)):
        if bow[ind]:
            s += vocab_inverse[ind]
            if ind < len(bow)-1:
                s+= " "
    return s

def log_predictions_to_file(context, predicted, correct_response, profile="", is_pos=True, id_example=1, incorrect_pred_path=None, correct_pred_path=None, vocab=None):
    """
    Log predictions to file (.json or .txt depending on dump_incorrect_predictions_path and dump_correct_predictions_path).
    Only logs for test or validation if it is set.
    """
    write_file = None
    if not is_pos and incorrect_pred_path is not None:
        write_file = incorrect_pred_path
    elif is_pos and correct_pred_path is not None:
        write_file = correct_pred_path
   
    context_text = convert_BOW_sentence(context, vocab)
    predicted_text = convert_BOW_sentence(predicted, vocab)
    correct_text = convert_BOW_sentence(correct_response, vocab)
    
    if write_file is not None:
        if write_file.endswith(".json"):
            # write to json file. NOTE: FOR FASTER SPEED, PREFER EITHER TEXT, OR ONLY USE IN TEST
            if os.path.isfile(write_file):
                with open(write_file) as f:
                    data_log = json.load(f)
            else:
                data_log = {}
            str_id = str(id_example)
            data_log[str_id] = {"profile": profile, 
                                "context": context_text, 
                                "predicted": predicted_text,
                                "correct": correct_text
                               }
                
            with open(write_file, 'w') as json_file:
                json.dump(data_log, json_file, indent=4, sort_keys=False)

        else:
            # write to text file                    
            with open(write_file, 'a') as f:
                txt = str(id_example) + \
                      "\nprofile\n" + profile + \
                      "\ncontext\n" + context_text +  \
                      "\npredicted\n" + predicted_text + \
                      "\ncorrect\n" + correct_text + "\n"
                f.write(txt)
        id_example += 1
    return id_example

def write_result_to_csv(report, result_file, task_id, datapath=None, OOV=False, test_time=None):
    """Write evaluation results to csv."""
    
    if datapath is not None:
        result_file = modelzoo_path(datapath, result_file)
    
    if report is None or result_file is None:
        return
    
    if OOV:
        result_file = result_file.replace("_test", "_test_OOV")

    file_exists = True
    if not os.path.isfile(result_file):
        file_exists = False
        file_dir = os.path.dirname(result_file)
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

    columns = ['task']
    results = [task_id]
    for key, value in report.items():
        if value:
            columns.append(key)
            if key == "accuracy":
                results.append("%.2f" % (float(value)*100))
            else:
                results.append("%.2f" % value)
    if test_time:
        columns.append('test_time')
        results.append("%.2f" % test_time)

    with open(result_file, 'a') as outcsv:
        writer = csv.writer(outcsv)
        if not file_exists:
            writer.writerow(columns)
        writer.writerow(results)
 

