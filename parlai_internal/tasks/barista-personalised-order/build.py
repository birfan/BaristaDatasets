# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'barista-personalised')
    version = None
    build_data.mark_done(dpath, version_string=version)
    # check if data had been previously built

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        build_data.mark_done(dpath, version_string=version)

        """
        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)


        # Download the data.
        fname = 'barista-personalised-dataset.tar.gz'
        url = 'https://github.com/birfan/ConvMem.git/datasets/' + fname
        build_data.download(url, dpath, fname)

        # uncompress it
        build_data.untar(dpath, fname)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)

        """

