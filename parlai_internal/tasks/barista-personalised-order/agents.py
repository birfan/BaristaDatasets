# Copyright (c) 2018-present, Bahar Irfan.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
#from parlai.core.teachers import ParlAIDialogTeacher
from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import os.path
import copy

tasks = {}
tasks[0] = "barista-personalised-order-task0-unknown"
tasks[1] = "barista-personalised-order-task1-unknown-known"
tasks[2] = "barista-personalised-order-task2-recognition-error"
tasks[3] = "barista-personalised-order-task3-incorrect-recall"
tasks[4] = "barista-personalised-order-task4-changes-to-preference"
tasks[5] = "barista-personalised-order-task5-recognition-recall"
tasks[6] = "barista-personalised-order-task6-recognition-changes"
tasks[7] = "barista-personalised-order-task7-recall-changes"
tasks[8] = "barista-personalised-order-task8-all"

fold_main = 'barista-personalised-order'
fold_ds = 'barista-personalised-order-dataset'
fold_second = 'SecondInteraction'
fold_1k = 'Task1k'
fold_6k = 'Task6k'
fold_10k = 'Task10k'
#TODO: remove 'internal:' +
def _path(exsz, task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'trn'
    elif dt == 'test':
        suffix = 'tst'
    elif dt == 'valid':
        suffix = 'dev'
    elif dt == 'oov':
        suffix = 'tst-OOV'
    return os.path.join(opt['datapath'], fold_main, fold_ds,
        '{exsz}'.format(exsz=exsz), 'task{num}'.format(num=task),
        '{tsk}-{type}.txt'.format(tsk=tasks[int(task)], type=suffix))


# python <script.py> -t barista-personalised-order:SecondInteraction:<task_id>
# Second Interaction scenario
#class ExampleTeacher(ParlAIDialogTeacher):
class SecondInteractionTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_example, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_second, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_second,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised-order:Task1k:<task_id>
# 1k dialogs
#class ExampleTeacher(ParlAIDialogTeacher):
class Task1kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_example, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_1k, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_1k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised-order:Task6k:<task_id>
# 6k dialogs
#class ExampleTeacher(ParlAIDialogTeacher):
class Task6kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_example, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_6k, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_6k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised:Task10k:<task_id>
# 10k dialogs
#class ExampleTeacher(ParlAIDialogTeacher):
class Task10kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_example, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_10k, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_10k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised-order:AllSecondInteraction
# Train on all tasks at once for SecondInteraction
class AllSecondInteractionTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_second + ':%d' % (i)
                               for i in range(len(tasks)) )
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_second,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised-order:All1k
# Train on all tasks at once for 1k dialogs
class All1kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_1k + ':%d' % (i)
                               for i in range(len(tasks)) )
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_1k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised-order:All6k
# Train on all tasks at once for 6k dialogs
class All6kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_6k + ':%d' % (i)
                               for i in range(len(tasks)) )
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_6k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista-personalised:All10k
# Train on all tasks at once for 10k dialogs
class All10kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_10k + ':%d' % (i)
                               for i in range(len(tasks)) )
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_10k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# By default train on all tasks at once for 1k dataset.
class DefaultTeacher(All1kTeacher):
    pass
