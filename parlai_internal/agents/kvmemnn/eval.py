#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluate pre-trained model trained for hits@1 metric
Key-Value Memory Net model trained on barista datasets
"""

from parlai.scripts.eval_model import eval_model, setup_args
from parlai_internal.scripts.utils import write_result_to_csv, update_opt
import os
import time

if __name__ == '__main__':
    parser = setup_args()
    parser.add_argument('-ds','--dataset',
                           default='barista-personalised', type=str,
                           help='Dataset name. Choices: barista, barista-personalised, barista-personalised-order.')
    parser.add_argument('-ts', '--task-size',
                           default='Task1k', type=str,
                           help='Task size folder,'
                           'choices: SecondInteraction, Task1k, Task10k for barista-personalised and barista-personalised-order'
                           'choices: Task100, Task1k, Task10k for barista.')
    parser.add_argument('-tid', '--task-id', type=int, default=1,
                           help='Task number, default is 1. For personalised sets, 0-8, for barista 1-7.')

    parser.set_defaults(
        task='internal:barista-personalised:Task1k:1',
        model='parlai_internal.agents.kvmemnn.kvmemnn:KvmemnnAgent',
        model_file='izoo:KVMemNN/model',
        datatype='test',
    )
    opt = parser.parse_args(print_args=False)

    # add additional model args
    opt = update_opt(opt, "KVMemNN", log_incorrect=True, log_correct=True)
    
    new_parser = setup_args(parser=parser)
    new_parser.set_params(
        task=opt['task'],
        model_file=opt['model_file'],
        log_predictions=opt['log_predictions'],
        dump_incorrect_predictions_path=opt['dump_incorrect_predictions_path'],
        dump_correct_predictions_path=opt['dump_correct_predictions_path'],
        datatype='test',
        numthreads=1,
        batchsize=1,
        hide_labels=False,
        dict_lower=True,
        dict_include_valid=False,
        dict_tokenizer='split',
        rank_candidates=True,
        metrics='accuracy,f1,hits@1',
        display_examples=False,
    )
    opt = new_parser.parse_args()

    start_test = time.time()

    report = eval_model(opt)

    test_time = time.time() - start_test

    result_file = os.path.join("izoo:" + "KVMemNN", opt['dataset'], opt['task_size'], "hop"+str(opt['hops']), "log") + "/results_test.csv"
    write_result_to_csv(report, result_file, opt['task_id'], opt['datapath'], test_time=test_time)
