#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train model for ppl metric with pre-selected parameters.

These parameters have some variance in their final perplexity, but they were used to
achieve the pre-trained model.
"""

from parlai.scripts.train_model import setup_args, TrainLoop
from parlai_internal.scripts.utils import update_opt
import os

if __name__ == '__main__':
    model_name="ProfileMemory-extended-dict"
    parser = setup_args()

    # BI:additional arguments for barista datasets
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
        model='parlai_internal.agents.profilememory.persona_seq2seq:PersonachatSeqseqAgentSplit',
        model_file='izoo:ProfileMemory/model',
        datatype='train',
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
        datatype='train',
        batchsize=128,
        numthreads=1,
        num_epochs=100,
        dict_lower=True,
        dict_include_valid=True,
        dict_include_test=True,
        dict_maxexs=-1,
        dict_tokenizer='split',
        hiddensize=1024,
        embeddingsize=300,
        attention='general',
        numlayers=2,
        encoder='lstm',
        learningrate=0.001,
        dropout=0.2,
        #validation_every_n_secs=60,
        validation_every_n_epochs=1,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=60,
        tblog=True,
        personachat_attnsentlevel=True,
        personachat_sharelt=True,
        personachat_reweight='use',
        personachat_learnreweight=True,
        truncate=100,
        use_persona=True,
    )
    opt = new_parser.parse_args()

    TrainLoop(opt).train()

