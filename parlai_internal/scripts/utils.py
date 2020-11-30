#!/usr/bin/env python3

# Copyright (c) Bahar Irfan, 2020.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import csv
import os
import importlib

def log_predictions_to_file(context, predicted, correct_response, profile="", id_example=1, incorrect_pred_path=None, correct_pred_path=None):
    """
    Log predictions to file (.json or .txt depending on dump_incorrect_predictions_path and dump_correct_predictions_path).
    Only logs for test or validation if it is set.
    """
    write_file = None
    if predicted.lower() != correct_response.lower() and incorrect_pred_path is not None:
        write_file = incorrect_pred_path
    elif predicted.lower() == correct_response.lower() and correct_pred_path is not None:
        write_file = correct_pred_path
   
    if write_file is not None:
        if write_file.endswith(".json"):
            # write to json file. NOTE: FOR FASTER SPEED, PREFER EITHER TEXT, OR ONLY USE IN TEST
            try:
                if os.path.isfile(write_file):
                    with open(write_file) as f:
                        data_log = json.load(f)
                else:
                    data_log = {}
                str_id = str(id_example)
                data_log[str_id] = {"profile": profile, 
                                "context": context, 
                                "predicted": predicted,
                                "correct": correct_response
                               }
                
                with open(write_file, 'w') as json_file:
                    json.dump(data_log, json_file, indent=4, sort_keys=False)
            except:
                pass

        else:
            try:
                # write to text file                    
                with open(write_file, 'a') as f:
                    txt = str(id_example) + \
                          "\nprofile\n" + profile + \
                          "\ncontext\n" + context +  \
                          "\npredicted\n" + predicted + \
                          "\ncorrect\n" + correct_response + "\n"
                    f.write(txt)
            except:
                pass
        id_example += 1
    return id_example
 
def update_predictions_path(opt):
    """Update predictions path with zoo path."""
    incorrect_pred_path = modelzoo_path(opt.get('datapath'), opt.get('dump_incorrect_predictions_path'))
    correct_pred_path = modelzoo_path(opt.get('datapath'), opt.get('dump_correct_predictions_path'))
    id_example=1     
    return incorrect_pred_path, correct_pred_path, id_example

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

def update_opt(opt, model_name, log_incorrect=False, log_correct=False):
    opt['task']='internal:' + opt['dataset'] + ':' + opt['task_size'] + ':' + str(opt['task_id'])

    hops_dir = ""
    if 'hops' in opt:
        if opt['hops'] is not None:
            hops_dir = "hop" + str(opt['hops'])

    opt['model_file'] = 'izoo:' + os.path.join(model_name, opt['dataset'], opt['task_size'], hops_dir, 'checkpoints', 'task'+str(opt['task_id'])) + '/model'

    task_log_folder = modelzoo_path(opt['datapath'], opt['model_file'].replace("checkpoints", "log").replace("model",""))
    if not os.path.exists(task_log_folder):
        os.makedirs(task_log_folder)

    if log_incorrect:
        opt['log_predictions']=True
        #opt['dump_incorrect_predictions_path'] = task_log_folder + "incorrect_predictions-" + opt['datatype'] + ".json"
        opt['dump_incorrect_predictions_path'] = task_log_folder + "incorrect_predictions-" + opt['datatype'] + ".txt"
        #if "KVMemNN" in model_name:
        #    opt['dump_incorrect_predictions_path'] = opt['dump_incorrect_predictions_path'].replace(".json", ".txt")
    else:
        opt['dump_incorrect_predictions_path'] = None

    if log_correct:
       opt['log_predictions']=True
       #opt['dump_correct_predictions_path'] = task_log_folder + "correct_predictions-" + opt['datatype'] + ".json"
       opt['dump_correct_predictions_path'] = task_log_folder + "correct_predictions-" + opt['datatype'] + ".txt"
       #if "KVMemNN" in model_name:
       #    opt['dump_correct_predictions_path'] = opt['dump_correct_predictions_path'].replace(".json", ".txt")
    else:
       opt['dump_correct_predictions_path'] = None

    return opt

def modelzoo_path(datapath, path):
    """
    ParlAI internal module from parlai.core.build_data
    Used here to remove the dependency of utils on ParlAI
    Map pretrain models filenames to their path on disk.

    If path starts with 'models:', then we remap it to the model zoo path within the
    data directory (default is ParlAI/data/models). We download models from the model
    zoo if they are not here yet.
    """
    if path is None:
        return None
    if (
        not path.startswith('models:')
        and not path.startswith('zoo:')
        and not path.startswith('izoo:')
    ):
        return path
    elif path.startswith('models:') or path.startswith('zoo:'):
        zoo = path.split(':')[0]
        zoo_len = len(zoo) + 1
        model_path = path[zoo_len:]
        # Check if we need to download the model
        if "/" in path:
            animal = path[zoo_len : path.rfind('/')].replace('/', '.')
        else:
            animal = path[zoo_len:]
        if '.' not in animal:
            animal += '.build'
        module_name = 'parlai.zoo.{}'.format(animal)
        try:
            my_module = importlib.import_module(module_name)
            my_module.download(datapath)
        except (ImportError, AttributeError):
            try:
                # maybe we didn't find a specific model, let's try generic .build
                animal_ = '.'.join(animal.split(".")[:-1]) + '.build'
                module_name_ = 'parlai.zoo.{}'.format(animal_)
                my_module = importlib.import_module(module_name_)
                my_module.download(datapath)
            except (ImportError, AttributeError):
                # truly give up
                raise ImportError(
                    f'Could not find pretrained model in {module_name} or {module_name_}.'
                )

        return os.path.join(datapath, 'models', model_path)
    else:
        # Internal path (starts with "izoo:") -- useful for non-public
        # projects.  Save the path to your internal model zoo in
        # parlai_internal/.internal_zoo_path
        # TODO: test the internal zoo.
        zoo_path = 'parlai_internal/zoo/.internal_zoo_path'
        if not os.path.isfile('parlai_internal/zoo/.internal_zoo_path'):
            raise RuntimeError(
                'Please specify the path to your internal zoo in the '
                'file parlai_internal/zoo/.internal_zoo_path in your '
                'internal repository.'
            )
        else:
            with open(zoo_path, 'r') as f:
                zoo = f.read().split('\n')[0]
            return os.path.join(zoo, path[5:])

