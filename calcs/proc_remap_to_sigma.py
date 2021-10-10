#! /usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import project as P

import os
from subprocess import call
from glob import glob

from workflow import task_manager as tm
import ncops as ops

tm.CONDA_ENV = 'py3_geyser'

clobber = False

chunk_size = 3*12

#droot = '/glade/p_old/decpred/CESM-DPLE_POPCICEhindcast'
droot = '/glade/p/cesm/community/CESM-DPLE/CESM-DPLE_POPCICEhindcast'
case = 'g.e11_LENS.GECOIAF.T62_g16.009'
datestr = '024901-031612'

#dple_root = '/glade/p_old/decpred/CESM-DPLE/monthly'
dple_root = '/glade/p/cesm/community/CESM-DPLE/monthly'

dout = '/glade/p/cgd/oce/projects/DPLE_O2'

#-------------------------------------------------------------------------------
#-- remap to sigma coordinates
#-------------------------------------------------------------------------------

# specify the sigma coordinate: computed range of PD in hindcast
dsigma = 0.05
sigma_start = 23.4-dsigma/2
sigma_stop = 27.12+dsigma/2

#-------------------------------------------------------------------------------
#-- define remap function
#-------------------------------------------------------------------------------

def remap(var_file_in,pd_file_in,file_out,include_thickness=True):

    if not os.path.exists(var_file_in):
        raise ValueError(f'{var_file_in} dne')

    if not os.path.exists(pd_file_in):
        raise ValueError(f'{pd_file_in} dne')

    if not os.path.exists(os.path.dirname(file_out)):
        call(['mkdir','-p',os.path.dirname(file_out)])

    control = {'file_in_sigma':pd_file_in,
               'file_in':var_file_in,
               'file_out':file_out,
               'dzname':'dz',
               'kmtname': 'KMT',
               'zname': 'z_t',
               'include_thickness':include_thickness,
               'sigma_start':sigma_start,
               'sigma_stop':sigma_stop,
               'sigma_delta':dsigma,
               'convert_from_pd': True,
               'sigma_varname' : 'PD'}

    if not os.path.exists(file_out) or clobber:
        jid = ops.ncop_chunktime(script='remap_z_to_sigma/calc_z_to_sigma.py',
                                 kwargs = control,
                                 chunk_size = chunk_size,
                                 clobber=clobber,
                                 cleanup=True)



for v in ['NO3','O2','SALT']:

    #-- remap hindcast
    var_file_in = f'{droot}/{case}.pop.h.{v}.{datestr}.nc'
    pd_file_in = f'{droot}/{case}.pop.h.PD.{datestr}.nc'
    file_out = f'{dout}/sigma_coord/CESM-DPLE_POPCICEhindcast/{case}.pop.h.sigma.{v}.{datestr}.nc'
    
    remap(var_file_in,pd_file_in,file_out)

    if v != 'O2': continue

    #-- remap DPLE
    files_in = sorted(glob(f'{dple_root}/{v}/b.e11.BDP.f09_g16.*.nc'))
    for var_file_in in files_in:
        pd_file_in = var_file_in.replace(f'/{v}/','/PD/').replace(f'.{v}.','.PD.')
        file_out = var_file_in.replace(f'{dple_root}',f'{dout}/sigma_coord/CESM-DPLE')
        remap(var_file_in,pd_file_in,file_out,include_thickness=False)

tm.wait()
