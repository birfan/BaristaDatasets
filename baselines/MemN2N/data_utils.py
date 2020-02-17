from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf
import csv
import json

stop_words = set(["a","an","the"])

def load_candidates(data_dir, task_id, dataset_name):
    """Load bot response candidates."""
    assert task_id >= 0 and task_id < 9

    candidates=[]
    candid_dic={}
    candidates_f="../" + dataset_name + "-candidates.txt"
    print("candidates_f", candidates_f)
    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    return candidates,candid_dic


def load_dialog_task(data_dir, task_id, candid_dic, isOOV, dataset_name):
    """Load the nth task.

    Returns a tuple containing the training and testing data for the task.
    """
    assert task_id >= 0 and task_id < 9

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = dataset_name + '-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and f.endswith('trn.txt')][0]
    if isOOV:
        test_file = [f for f in files if s in f and f.endswith('tst-OOV.txt')][0]
    else: 
        test_file = [f for f in files if s in f and f.endswith('tst.txt')][0]
    val_file = [f for f in files if s in f and f.endswith('dev.txt')][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def tokenize(sent):
    """Return the tokens of a sentence including punctuation.
    
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    """
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result


def parse_dialogs_per_response(lines,candid_dic):
    """Parse dialogs provided in the babi tasks format."""
    data=[]
    context=[]
    u=None
    r=None
    for line in lines:
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                # Process turn containing bot response
                u, r = line.split('\t')
                a = candid_dic[r]
                u = tokenize(u)
                r = tokenize(r)
                # Add temporal encoding, and utterance/response encoding
                data.append((context[:],u[:],a))
                u.append('$u')
                u.append('#'+str(nid))
                r.append('$r')
                r.append('#'+str(nid))
                context.append(u)
                context.append(r)
            else:
                # Process turn without bot response
                r=tokenize(line)
                r.append('$r')
                r.append('#'+str(nid))
                context.append(r)
        else:
            # Clear context
            context=[]
    return data



def get_dialogs(f,candid_dic):
    """Given a file name, read the file, retrieve the dialogs, and then convert 
    the sentences into a single dialog.
    
    If max_length is supplied, any stories longer than max_length tokens will 
    be discarded.
    """
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)


def vectorize_candidates_sparse(candidates,word_idx):
    shape=(len(candidates),len(word_idx)+1)
    indices=[]
    values=[]
    for i,candidate in enumerate(candidates):
        for w in candidate:
            indices.append([i,word_idx[w]])
            values.append(1.0)
    return tf.SparseTensor(indices,values,shape)


def vectorize_candidates(candidates,word_idx,sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    return tf.constant(C,shape=shape)


def vectorize_data(data, word_idx, sentence_size, 
                   batch_size, candidates_size, max_memory_size):
    """Vectorize stories, and queries.
    
    Vectors contain indices of the dictionary for words in a sentence, and 0 if the word is not
    in the dictinary.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (story, query, answer) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)

        # Take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq

        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))
    return S, Q, A

# BI:extra functions
def convert_word_index_to_sentence(word_idx, vocab):
    """Convert word indices to sentence."""
    s = ""
    for ind in range(len(word_idx)):
        w_idx = word_idx[ind]
        if w_idx> 0:
            s += vocab[w_idx-1]
            if ind < len(word_idx)-1:
                s+= " "
    return s

def log_predictions_to_file(context, predicted, correct_response, profile="", id_example=1, incorrect_pred_path=None, correct_pred_path=None):
    """
    Log predictions to file (.json or .txt depending on dump_incorrect_predictions_path and dump_correct_predictions_path).
    Only logs for test or validation if it is set.
    """
    write_file = None
    if predicted != correct_response and incorrect_pred_path is not None:
        write_file = incorrect_pred_path
    elif predicted == correct_response and correct_pred_path is not None:
        write_file = correct_pred_path
   
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
                                "context": context, 
                                "predicted": predicted,
                                "correct": correct_response
                               }
                
            with open(write_file, 'w') as json_file:
                json.dump(data_log, json_file, indent=4, sort_keys=False)

        else:
            # write to text file                    
            with open(write_file, 'a') as f:
                txt = str(id_example) + \
                      "\nprofile\n" + profile + \
                      "\ncontext\n" + context +  \
                      "\npredicted\n" + predicted + \
                      "\ncorrect\n" + correct_response + "\n"
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
            results.append("%.2f" % value)
    if test_time:
        columns.append('test_time')
        results.append("%.2f" % test_time)

    with open(result_file, 'a') as outcsv:
        writer = csv.writer(outcsv)
        if not file_exists:
            writer.writerow(columns)
        writer.writerow(results)
 

