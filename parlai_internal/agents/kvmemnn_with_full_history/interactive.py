#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Interact with a pre-trained model. Key-Value Memory Net model trained on barista datasets.
"""

from parlai.scripts.interactive import interactive, setup_args
from parlai_internal.scripts.utils import update_opt
import os

if __name__ == '__main__':
    parser = setup_args()
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
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
    )
    opt = parser.parse_args(print_args=False)

    # add additional model args
    opt = update_opt(opt, "KVMemNN", log_incorrect=False, log_correct=False)
    
    new_parser = setup_args(parser=parser)
    new_parser.set_params(
        task=opt['task'],
        model_file=opt['model_file'],
        numthreads=1,
        batchsize=1,
        dict_lower=True,
        dict_include_valid=False,
        dict_tokenizer='split',
    )
    opt = new_parser.parse_args()

    interactive(opt)
