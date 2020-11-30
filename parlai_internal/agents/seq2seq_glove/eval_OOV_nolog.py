#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluate pre-trained model trained for hits@1 metric.
Sequence-to-sequence model trained on barista datasets
"""

from parlai.scripts.eval_model import eval_model, setup_args
from parlai_internal.scripts.utils import write_result_to_csv, update_opt
import os
import time

if __name__ == '__main__':
    model_name="Seq2Seq-extended-dict-glove-no-history"
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
        model='parlai_internal.agents.seq2seq.seq2seq_v0:Seq2seqAgent',
        model_file='izoo:Seq2Seq/model',
        datatype='oov',
    )
    opt = parser.parse_args(print_args=False)

    # add additional model args
    opt = update_opt(opt, model_name, log_incorrect=False, log_correct=False)
    
    new_parser = setup_args(parser=parser)
    new_parser.set_params(
        task=opt['task'],
        model_file=opt['model_file'],
        log_predictions=opt['log_predictions'],
        dump_incorrect_predictions_path=opt['dump_incorrect_predictions_path'],
        dump_correct_predictions_path=opt['dump_correct_predictions_path'],
        datatype='oov',
        numthreads=opt['numthreads'],
        batchsize=opt['batchsize'], #used to be 32
        hide_labels=False,
        dict_lower=True,
        dict_include_valid=True,
        dict_include_test=False,
        dict_tokenizer='split',
        rank_candidates=False,
        metrics='accuracy,f1,hits@1,ppl',
        display_examples=False,
        #log_every_n_secs=2,
    )
    opt = new_parser.parse_args()

    start_test = time.time()

    report = eval_model(opt)

    test_time = time.time() - start_test

    result_file = os.path.join("izoo:" + model_name, opt['dataset'], opt['task_size'], "log") + "/results_test.csv"
    write_result_to_csv(report, result_file, opt['task_id'], opt['datapath'], OOV=True, test_time=test_time)
