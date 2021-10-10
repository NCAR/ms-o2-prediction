#! /usr/bin/env python
import os
from subprocess import call
from glob import glob
import yaml

import logging
import time

import xarray as xr
import numpy as np

import cftime
from datetime import datetime

#filename='proc_cesm_dple.log',
logging.basicConfig(level=logging.INFO)

#-----------------------------------------------------------------------------
#-- global constants
#-----------------------------------------------------------------------------

USER = os.environ['USER']

n_ensemble_max = 10

mean = 'ann'
n_forecast_lead = 10 # related to expected freq from "mean"

sigma_coord = True
if not sigma_coord:
    diro = f'/glade/scratch/{USER}/calcs/o2-prediction'
else:
    diro = f'/glade/scratch/{USER}/calcs/o2-prediction/sigma_coord'

first_start_year, last_start_year = 1954, 2015
first_clim_year, last_clim_year = 1964, 2014

#first_start_year, last_start_year = 1954, 1956
#first_clim_year, last_clim_year = 1964, 1956

do_dask = False
mem_limit_local = 30.

file_format = 'nc' # nc or zarr for output

case_prefix = 'b.e11.BDP.f09_g16'
if not sigma_coord:
    dir_dple = '/glade/p_old/decpred/CESM-DPLE'
else:
    dir_dple = '/glade/scratch/mclong/calcs/o2-prediction/sigma_coord/CESM-DPLE'

#-----------------------------------------------------------------------------
#-- function
#-----------------------------------------------------------------------------

def write_output(ds,file_out,attrs={}):
    '''Function to write output:
       - optionally add some file-level attrs
       - switch method based on file extension
       '''

    diro = os.path.dirname(file_out)
    if not os.path.exists(diro):
        call(['mkdir','-p',diro])

    if os.path.exists(file_out):
        call(['rm','-f',file_out])

    if attrs:
        ds.attrs = attrs

    ext = os.path.splitext(file_out)[1]

    logging.info(f'writing {file_out}')
    if ext == '.nc':
        ds.to_netcdf(file_out,unlimited_dims=['L'])

    elif ext == '.zarr':
        ds.to_zarr(file_out)

    else:
        raise ValueError('Unknown output file extension: {ext}')


#-----------------------------------------------------------------------------
#-- function
#-----------------------------------------------------------------------------

def time_bound_var(ds):
    tb_name = ''
    if 'bounds' in ds['time'].attrs:
        tb_name = ds['time'].attrs['bounds']
    elif 'time_bound' in ds:
        tb_name = 'time_bound'
    else:
        raise ValueError('No time_bound variable found')
    tb_dim = ds[tb_name].dims[-1]
    return tb_name,tb_dim

#-----------------------------------------------------------------------------
#-- function
#-----------------------------------------------------------------------------

def fix_time(ds):
    tb_name,tb_dim = time_bound_var(ds)

    if 'M' in ds[tb_name].dims:
        time_bound = ds[tb_name].isel(M=0)
    else:
        time_bound = ds[tb_name]

    if 'units' in ds.time.attrs:
        units = ds.time.attrs['units']
    else:
        units = 'days since 0000-01-01 00:00:00'

    if 'calendar' in ds.time.attrs:
        calendar = ds.time.attrs['calendar']
    else:
        calendar = 'noleap'


    time = cftime.num2date(time_bound.mean(tb_dim),
                           units = units,calendar = calendar)

    ds.time.values = time
    ds = ds.drop([tb_name])
    return ds

#-----------------------------------------------------------------------------
#-- function
#-----------------------------------------------------------------------------

def proc_dple_single_member(ds,L,isel={}):

    #-- fix time and drop extraneous variables
    dsattrs = ds.attrs
    attrs = {v:da.attrs for v,da in ds.variables.items()}
    encoding = {v:{key:val for key,val in da.encoding.items()
                   if key in ['dtype','_FillValue','missing_value']}
                   for v,da in ds.variables.items()}

    ds = fix_time(ds)
    ds = ds.drop([v for v in ds.variables if v not in ds.coords and v != var])

    #-- subset?
    if isel:
        ds = ds.isel(**isel)
        if 'coordinates' in attrs[var]:
            del attrs[var]['coordinates']
        if 'coordinates' in ds[var].attrs:
            del ds[var].attrs['coordinates']

    #-- compute appropriate average TODO: ds = calc.compute_{freq}_mean(ds).rename({'time':'L'})
    ds = ds.groupby('time.year').mean('time').rename({'year':'L'})
    if len(ds.L) == n_forecast_lead+1:
        ds = ds.isel(L=slice(1,11))
    ds['L'] = L

    ds.attrs = dsattrs
    for v in ds.variables:
        if v in attrs:
            ds[v].attrs = attrs[v]
        if v in encoding:
            ds[v].encoding = encoding[v]

    return ds

#-----------------------------------------------------------------------------
#-- function
#-----------------------------------------------------------------------------

def drift_corr_dple(var,isel={},forecast_year_range=None,varout=''):
    '''Compute drift correction on CESM-DPLE'''

    logging.info('-'*80)
    logging.info(f'entering drift_corr_dple:')
    logging.info(f'var={var}')
    logging.info(f'isel={isel}')
    logging.info(f'forecast_year_range={forecast_year_range}')
    logging.info(f'varout={varout}')

    #-----------------------------------------------------
    #-- input args
    #-----------------------------------------------------

    if not varout:
        varout = var

    #-----------------------------------------------------
    #-- make coordinate axes
    #-----------------------------------------------------

    S = xr.DataArray(np.arange(first_start_year,last_start_year+1,1,dtype='int32')+1,
                 dims='S',
                 attrs={'long_name':'start year'})
    logging.info(f'{S}\n')

    L = xr.DataArray(np.arange(1,n_forecast_lead+1,1,dtype='int32'),
                 dims='L',
                 attrs={'long_name':'forecast lead'})
    logging.info(f'{L}\n')

    if forecast_year_range is not None:
        y0 = forecast_year_range[0] - 1
        y1 = forecast_year_range[1] - 1
        nx_mon0 = y0*12 + 2
        nx_mon1 = y1*12 + 2 + 12
        L = L.isel(L=slice(y0,y1+1))
        isel.update({'time':slice(nx_mon0,nx_mon1)})

    isel_str = ''
    if isel:
        isel_str = '.'.join([index_str(k,v) for k,v in isel.items()])
        isel_str = f'.{isel_str}.'

    file_out = {'mean': f'{diro}/CESM-DPLE{isel_str}{varout}.{mean}.mean.{file_format}',
                'drift': f'{diro}/CESM-DPLE{isel_str}{varout}.{mean}.mean.drift.{file_format}',
                'anom': f'{diro}/CESM-DPLE{isel_str}{varout}.{mean}.mean.anom.{file_format}'}

    logging.info(f'file_out = {file_out}')

    #-----------------------------------------------------
    #-- make verification time matrix
    #-----------------------------------------------------

    verification_time = S + 0.5 + L - 1

    #-----------------------------------------------------
    #-- find the files
    #-----------------------------------------------------

    files_by_year = {}
    n = np.zeros(len(S),dtype='int32')

    for i in range(0,len(S)):
        year = S.values[i]-1
        files_by_year[year] = sorted(glob(f'{dir_dple}/monthly/{var}/{case_prefix}.{year}*.nc'))[:n_ensemble_max]
        n[i] = len(files_by_year[year])

    #-- ensure that we have the same number of files for each start year
    n_ensemble = n[0]
    np.testing.assert_equal(n_ensemble,n)

    logging.info(f'{n_ensemble} files for {len(S)} start years.')

    #-----------------------------------------------------
    #-- assemble ensemble into `start_year x lead_time x lat x lon` array
    #-----------------------------------------------------

    grid_vars = []
    ds_list = []
    xr_open_ds = {'decode_times': False,'decode_coords': False}

    #-- loop over files for each year
    for year,files in files_by_year.items():
        logging.info(f'start year: {year}')
        tic = time.time()

        #-- open the datasets
        if do_dask:
            ds = xr.open_mfdataset(files,
                                   concat_dim='M',**xr_open_ds)
            ds = proc_dple_single_member(ds,L,isel=isel)
        else:
            ds_sublist = []
            for f in files:
                logging.info(f'loading {f}')
                with xr.open_dataset(f,**xr_open_ds) as ds:
                    dsi = proc_dple_single_member(ds,L,isel=isel)
                    dsi.load()
                ds_sublist.append(dsi)
            ds = xr.concat(ds_sublist,dim='M')

        ds_list.append(ds)
        toc = time.time() - tic

        logging.info(f'it/s: {toc:0.2f}')

    #-- assemble into single dataset
    ds = xr.concat(ds_list,dim='S')
    ds['S'] = S

    #-----------------------------------------------------
    #-- rechunk to more suitable sizes
    #-----------------------------------------------------

    if do_dask:
        logging.info('rechunk')
        new_chunks = {'S':1,'L':len(L),'M':n_ensemble}

        #if 'z_t' in ds[var].dims:
        #    new_chunks = {'S':len(S),'L':len(L),'M':n_ensemble,'nlat':16,'nlon':16}
        ds = ds.chunk(new_chunks)

    logging.info(ds)
    logging.info(f'dataset size: {ds.nbytes/1024**3:0.4f} GB')

    #-----------------------------------------------------
    #-- persist dataset
    #-----------------------------------------------------

    if do_dask:
        if ds.nbytes/1024**3 < mem_limit_local:
            logging.info('computing')
            ds = ds.compute()
        else:
            logging.info('persisting')
            ds = ds.persist()

        logging.info('done computing')

    #-----------------------------------------------------
    #-- compute ensemble mean
    #-----------------------------------------------------

    logging.info('computing ensemble mean')

    dse = ds.mean('M')

    logging.info(dse)

    #-----------------------------------------------------
    #-- compute drift
    #-----------------------------------------------------

    logging.info('computing drift')

    drift = dse.where((first_clim_year<verification_time) &
                      (verification_time<last_clim_year+1) ).mean('S')

    for v in drift.variables:
        if v in ds[v].attrs:
            drift[v].attrs = ds[v].attrs[v]
        if v in ds[v].encoding:
            drift[v].encoding = ds[v].encoding[v]

    logging.info(drift)

    #-----------------------------------------------------
    #-- compute anomaly
    #-----------------------------------------------------

    logging.info('computing anomaly')

    anom = ds - drift

    for v in anom.variables:
        if v in ds[v].attrs:
            anom[v].attrs = ds[v].attrs[v]
        if v in ds[v].encoding:
            anom[v].encoding = ds[v].encoding[v]

    logging.info(anom)

    #-----------------------------------------------------
    #-- write output
    #-----------------------------------------------------

    dsattrs = ds.attrs
    dsattrs['history'] = f'created by {USER} on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    dsattrs['climatology'] = f'{first_clim_year:d}-{last_clim_year}, computed separately for each lead time'

    if var != varout:
        ds.rename({var:varout},inplace=True)
        anom.rename({var:varout},inplace=True)
        drift.rename({var:varout},inplace=True)

    write_output(ds, file_out = file_out['mean'], attrs = dsattrs)
    write_output(anom, file_out = file_out['anom'], attrs = dsattrs)
    write_output(drift, file_out = file_out['drift'], attrs = dsattrs)

    logging.info('done.')
    logging.info('-'*80)

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def index_str(dimname,indexer):
    if isinstance(indexer,int):
        index_str = f'{dimname}-{indexer:03d}'
    elif isinstance(indexer,slice):
        index_str = f'{dimname}-{indexer.start:03d}-{indexer.stop:03d}'
    else:
        raise ValueError(f'indexer type not recognized: {type(indexer)}')
    return index_str

#-------------------------------------------------------------------------------
#-- main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Process CESM-DPLE')

    p.add_argument('-i', dest = 'forecast_year_index',
                   type = int,
                   required = False,
                   help = 'index')

    p.add_argument('-v', dest = 'var',
                   type = str,
                   required = True,
                   help = 'variable')

    args = p.parse_args()

    npac_isel = {'nlat':slice(187,331),'nlon':slice(137,276)}

    var = args.var
    forecast_year_range = [(y,y) for y in range(1,11)]
    if args.forecast_year_index is not None:
        i = args.forecast_year_index
        forecast_year_range = [forecast_year_range[i]]

    for i,yr in enumerate(forecast_year_range):
        drift_corr_dple(var,forecast_year_range=yr,isel = npac_isel)

    logging.info('done')
