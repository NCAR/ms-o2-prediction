#! /usr/bin/env python

import os
from subprocess import call
from glob import glob

from workflow import task_manager as tm

tm.CONDA_ENV = 'py3_geyser'

script = '/gpfs/u/home/mclong/p/o2-prediction/calcs/proc_cesm_dple.py'
for i in range(10):
    tm.submit(['./proc_cesm_dple.py','-v','O2','-i',f'{i}'],
              memory='200GB')

tm.wait()
