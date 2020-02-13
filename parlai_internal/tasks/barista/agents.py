# Copyright (c) 2018-present, Bahar Irfan.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
#from parlai.core.teachers import ParlAIDialogTeacher
from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os.path

tasks = {}
tasks[1] = "barista-task1-greetings"
tasks[2] = "barista-task2-order-drink-no-greetings"
tasks[3] = "barista-task3-order-drink-no-greetings-changes"
tasks[4] = "barista-task4-order-all-no-greetings"
tasks[5] = "barista-task5-order-all-no-greetings-changes"      
tasks[6] = "barista-task6-order-all-with-greetings"
tasks[7] = "barista-task7-order-all-with-greetings-changes"
fold_main = 'barista'
fold_ds = 'barista-dataset'
fold_example = 'Example'
fold_1k = 'Task1k'
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


# python <script.py> -t barista:Example:<task_id>
# 10 dialogs task (example).
#class ExampleTeacher(ParlAIDialogTeacher):
class ExampleTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_example, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_example, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_example,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t barista:Task1k:<task_id>
# Single 1k task.
#class Task1kTeacher(ParlAIDialogTeacher):
class Task1kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_1k, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_1k, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_1k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t barista:Task10k:<task_id>
# Single 10k task.
#class Task10kTeacher(ParlAIDialogTeacher):
class Task10kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        #opt['parlaidialogteacher_datafile'] = _path(fold_10k, opt['task'].split(':')[2], opt)
        splitted = opt['task'].split(':')
        task_num = splitted[len(splitted)-1]
        opt['datafile'] = _path(fold_10k, task_num, opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_10k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t barista:AllExample
# Train on all tasks at once for 10 dialogs task (example).
class AllExampleTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_example + ':%d' % (i + 1)
                               for i in range(len(tasks)))
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_example, 
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista:All1k
# Train on all tasks at once for 1k dataset.
class All1kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_1k + ':%d' % (i + 1)
                               for i in range(len(tasks)))
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_1k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)

# python <script.py> -t barista:All10k
# Train on all tasks at once for 10k dataset.
class All10kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join("internal:" + fold_main + ':' + fold_10k + ':%d' % (i + 1)
                               for i in range(len(tasks)))
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], fold_main, fold_ds, fold_10k,
            fold_main + '-candidates.txt'
        )
        super().__init__(opt, shared)


# By default train on all tasks at once for 1k dataset.
class DefaultTeacher(All1kTeacher):
    pass
