# Copyright (c) 2018-present, Bahar Irfan.
# All rights reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'barista-personalised-order')
    version = None
    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):

        if os.path.exists(dpath):
            build_data.mark_done(dpath, version_string=version)
        else:
            print("See download instructions for barista-personalised-order dataset in README file at https://github.com/birfan/BaristaDatasets/")


