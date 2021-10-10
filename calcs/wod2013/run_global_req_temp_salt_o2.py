#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J wod-proc
#BSUB -o logs/wod-proc.%J
#BSUB -e logs/wod-proc.%J
#BSUB -q caldera
#BSUB -N
import os
import sys
from subprocess import call
from datetime import datetime,timedelta
import task_manager as tm
import json
import xarray as xr
import numpy as np

os.putenv('PYTHONUNBUFFERED','no')
now = datetime.now()
if not os.path.isdir('logs'):
    call(['mkdir','-p','logs'])

split_phase = False
scratch = '/glade/scratch/'+os.environ['USER']

#-----------------------------------------------------------------
#-- settings
#-----------------------------------------------------------------
from wod_info import woa_shortnames

#-- parameters of extraction
name_of_extraction = 'global_req_temp_salt_o2'
variables = ['temperature','salinity','oxygen']
max_depth = 2000.

files = [os.path.join(scratch,'wod2013','wod_'+name_of_extraction,f)
         for f in ['ocldb1501130274.26813.CTD',
                   'ocldb1501130274.26813.OSD',
                   'ocldb1501130274.26813.OSD2',
                   'ocldb1501130274.26813.OSD3',
                   'ocldb1501130274.26813.PFL',
                   'ocldb1501130274.26813.UOR']]

odir = os.path.join(os.environ['SCRATCH'],'wod2013',name_of_extraction)

#-- produce annual timeseries
yr0 = 1958
yr1 = 2017

#-----------------------------------------------------------------
#-- convert to netCDF
#-----------------------------------------------------------------

clobber = False
settings = {'name_of_extraction':name_of_extraction,
            'variables':variables,
            'output_dir' : odir,
            'max_depth':max_depth,
            'clobber' : clobber}

script = 'assemble_wod_database_nc.py'
files_out = []
for file_in in files:

    file_out = os.path.join(odir,os.path.basename(file_in)+'.nc')
    files_out.append(file_out)

    if not os.path.exists(file_out) or clobber:
        kwargs = settings.copy()
        kwargs.update({'file_in':file_in,'file_out':file_out})
        jid = tm.submit([script,'\'{0}\''.format(json.dumps(kwargs))])

tm.wait()

#-- if 2 phases, resubmit this script and exit
if split_phase and tm.total_elapsed_time() > 1:
    print('resubmitting myself')
    call('bsub < run_global_req_temp_salt_o2.py',shell=True)
    sys.exit(0)

#-----------------------------------------------------------------
#-- make gridded
#-----------------------------------------------------------------

clobber = False

script = 'make_gridded.py'
kwargs = {'variables':variables,
          'files_in': files_out,
          'max_depth':max_depth}

FILE_OUT = os.path.join(odir,'..',name_of_extraction+'.gridded.z.nc')
if not os.path.exists(FILE_OUT) or clobber:
    file_cat = []
    for yr in range(yr0,yr1+1):
        kwargs['yr0'] = yr
        kwargs['yr1'] = yr

        #--- regrid in z coordinates
        file_out = os.path.join(odir,'..',name_of_extraction+'.gridded.z.%04d.nc'%yr)
        kwargs['output_on_sigma'] = False
        kwargs['file_out'] = file_out
        file_cat.append(file_out)
        if not os.path.exists(file_out) or clobber:
            jid = tm.submit([script,'\'{0}\''.format(json.dumps(kwargs))])

    tm.wait()

    print('writing %s'%FILE_OUT)
    ds = xr.open_mfdataset(file_cat,concat_dim='time',data_vars='minimal')
    ds.to_netcdf(FILE_OUT,unlimited_dims='time')
    call('rm -f '+' '.join(file_cat),shell=True)

FILE_OUT = os.path.join(odir,'..',name_of_extraction+'.gridded.sigma.nc')
if not os.path.exists(FILE_OUT) or clobber:
    file_cat = []
    for yr in range(yr0,yr1+1):
        kwargs['yr0'] = yr
        kwargs['yr1'] = yr

        #--- regrid in sigma coordinates
        file_out = os.path.join(odir,'..',name_of_extraction+'.gridded.sigma.%04d.nc'%yr)
        kwargs['sigma_interp_method'] = 'remap_z'
        kwargs['output_on_sigma'] = True
        kwargs['file_out'] = file_out
        file_cat.append(file_out)
        if not os.path.exists(file_out) or clobber:
            jid = tm.submit([script,'\'{0}\''.format(json.dumps(kwargs))])

    tm.wait()

    print('writing %s'%FILE_OUT)
    ds = xr.open_mfdataset(file_cat,concat_dim='time',data_vars='minimal')
    ds.to_netcdf(FILE_OUT,unlimited_dims='time')
    call('rm -f '+' '.join(file_cat),shell=True)

#-----------------------------------------------------------------
#-- assemble climatology
#-----------------------------------------------------------------

import woa_tools
woa_pth = os.path.join(os.environ['SCRATCH'],'wod2013','woa2013v2')
if not os.path.exists(woa_pth):
    call(['mkdir','-p',woa_pth])
woa_files = woa_tools.make_merged_clim(woa_pth,clobber=False)

file_clm = {}
for v in variables:
    file_clm[v] = woa_files[woa_shortnames[v]]

file_sigma = woa_files['s'].replace('_s_','_PD_')
woa_tools.add_potential_density(woa_files['s'],woa_files['t'],
                                file_sigma,clobber=False)

file_clm_sigma = {}
for v,file_in in file_clm.items():
    file_clm_sigma[v] = file_in.replace('.nc','.sigma.nc')
    woa_tools.regrid_to_sigma_coords(file_in,file_clm_sigma[v],
                                     file_sigma,clobber=False)

#-----------------------------------------------------------------
#-- compute annual anomalies
#-----------------------------------------------------------------
import make_gridded

#-- read gridded monthly data
file_in = os.path.join(odir,'..',name_of_extraction+'.gridded.z.nc')
file_out = os.path.join(odir,'..',name_of_extraction+'.gridded.z.anom_wrt_woa.ann.nc')

if not os.path.exists(file_out) or clobber:
    make_gridded.compute_anom_wrt_woa(variables=variables,
                                      file_in = file_in,
                                      file_clm = file_clm,
                                      file_out = file_out,
                                      max_depth = max_depth,
                                      output_on_sigma=False)

#-- read gridded monthly data
file_in = os.path.join(odir,'..',name_of_extraction+'.gridded.sigma.nc')
file_out = os.path.join(odir,'..',name_of_extraction+'.gridded.sigma.anom_wrt_woa.ann.nc')

if not os.path.exists(file_out) or clobber:
    make_gridded.compute_anom_wrt_woa(variables=variables,
                                      file_in = file_in,
                                      file_clm = file_clm_sigma,
                                      file_out = file_out,
                                      max_depth = None,
                                      output_on_sigma=True)
