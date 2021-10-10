#! /usr/bin/env python
import os
import sys
from glob import glob
from subprocess import call,Popen,PIPE
from datetime import datetime

import netCDF4 as nc
from netcdftime import utime
import numpy as np
import xarray as xr
import multiprocessing as mp
import csv

from wod_info import wodvars,wodvars_attrs,standard_z

if not os.path.exists('logs'): call(['mkdir','logs'])

cdftime = utime(wodvars_attrs['time']['units'],
                calendar=wodvars_attrs['time']['calendar'])

#-- determine os
p = Popen(['uname','-a'], stdin=None, stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()
opsys = stdout.split()[0]
wodASC_exe = 'wodASC.'+opsys
if not os.path.exists(wodASC_exe):
    print('missing %s'%wodASC_exe)
    sys.exit(1)

max_cast = None

#-----------------------------------------------------------------
#-- function
#-----------------------------------------------------------------
def wod2asc(file_in,varname,clobber=False):
    '''
    call World Ocean Database's fortran program
    to read native formated and convert to ascii columns
    with a single variable
    '''

    #-- output file for this variable
    file_out = file_in+'.'+varname
    if os.path.exists(file_out) and not clobber:
        return file_out

    #-- integer field index
    ifld = [i for i in range(len(wodvars))
            if varname in wodvars[i].lower()][0]+1


    #-- namelist variables
    input_nml = {'filename' : '\'{0}\''.format(file_in),
                 'ifld' : '%d'%ifld,
                 'fileout' : '\'{0}\''.format(file_out)}

    #-- write the namelist file
    namelist = os.path.basename(file_in+'.'+varname+'.nml')
    with open(namelist,'w') as nml:
        nml.write('&input\n')
        for k,v in input_nml.items():
            nml.write('%s = %s\n'%(k,v))
        nml.write('/\n')

    #-- call the program
    stat = call('./'+wodASC_exe+' < '+namelist,shell=True)
    if stat != 0:
        return None
    else:
        return file_out


#-----------------------------------------------------------------
#-- function
#-----------------------------------------------------------------

def make_cast_file(z,data,varname,meta,castnumber,file_inout,standard_z):
    '''
    write cast data to netCDF file
    '''

    if os.path.exists(file_inout):
        cast = xr.open_dataset(file_inout,decode_times=False,decode_coords=False)
        cast.load()
        for k,v in meta.items():
            if cast[k].values[0] != v:
                print('ERROR: mismatch in cast file')
                print('file: %s'%file_inout)
                print('expecting: %s for %s'%(str(cast[k].values[0]),k))
                print('got: %s'%(str(v)))
                print('Cast file')
                print(cast)
                print
                print('meta')
                print(meta)
                sys.exit(1)
    else:
        cast = xr.Dataset(coords={'depth':standard_z,
                                  'cast':np.array([castnumber])})

        types = {'lat':np.float64,'lon':np.float64,'time':np.float64,
                 'cast':np.int32}
        for k,v in meta.items():
            cast[k] = xr.DataArray(np.array([v]),dims=('cast')).astype(types[k])

        for v in ['time','depth','lat','lon']:
            cast[v].attrs = wodvars_attrs[v]

    data_std_z = np.ones(len(standard_z)).astype(np.float32)
    data_std_z[:] = nc.default_fillvals['f4']

    for z,x in zip(z,data):
        if z > max_depth: continue
        znx = (z == standard_z)
        if not znx.any():
            print(z)
            print(data)
            print('error')
            sys.exit(1)
        data_std_z[znx] = x

    attrs = wodvars_attrs[varname]
    attrs.update({'_FillValue':nc.default_fillvals['f4']})
    cast[varname] = xr.DataArray(data_std_z[None,:],
                                 dims=('cast','depth'),
                                 attrs=attrs)

    if os.path.exists(file_inout):
        call(['rm','-f',file_inout])
    cast.to_netcdf(file_inout,mode='w',
                   unlimited_dims='cast')

#-----------------------------------------------------------------
#-- function
#-----------------------------------------------------------------
def count_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#-----------------------------------------------------------------
#-- function
#-----------------------------------------------------------------
def asc_get_casts(file_in):
    '''
    get cast numbers from ascii file
    '''

    nlines = count_lines(file_in)

    casts = []
    ncast = 0
    with open(file_in, 'rb') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        for ir,row in enumerate(csvdata):
            #-- cast definition line
            if row[0] == '%':
                casts.append(int(row[1]))
                ncast = len(casts)
            if max_cast is not None:
                if ncast == max_cast:
                    break
    return casts

#-----------------------------------------------------------------
#-- function
#-----------------------------------------------------------------
def asc2nc(file_in,varname,odir,standard_z,file_cast={}):
    '''
    covert ascii column format to netCDF
    '''

    nlines = count_lines(file_in)

    with open(file_in, 'rb') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
        z = []
        data = []
        ncast = 0
        for ir,row in enumerate(csvdata):

            #-- cast definition line
            if row[0] == '%':

                #-- write cast info from previously read definition
                if ir > 0:
                    if meta:
                        make_cast_file(z,data,varname,meta,castnumber,file_out,standard_z)
                    z = []
                    data = []

                #-- read cast definition line
                ncast += 1
                castnumber = int(row[1])
                lon = float(row[2])
                lat = float(row[3])
                year = int(row[4])
                mon = int(row[5])
                day = int(row[6])

                err = ''
                if not (-180. <= lon <= 180.):
                    err = 'lon value wrong'
                if not (-90. <= lat <= 90.):
                    err = 'lat value wrong'
                if not (1 <= mon <= 12):
                    err = 'mon value wrong'
                if not (1 <= day <= 31):
                    if day == 0:
                        day = 15
                    else:
                        err = 'day value wrong'

                meta = {}
                if err:
                    print('line %d, %s:'%(ir+1,file_in))
                    print('WARNING: %s'%err)
                    print('\t'.join(row))
                    print
                else:
                    file_out = os.path.join(odir,'wod2013_%012d.%s.nc'%(castnumber,varname))
                    if castnumber in file_cast.keys():
                        file_cast[castnumber].append(file_out)
                    else:
                        file_cast[castnumber] = [file_out]

                    time = cdftime.date2num(datetime(year,mon,day))
                    meta = {'lon' : lon, 'lat' : lat, 'time' : time}

                if max_cast is not None:
                    if ncast == max_cast:
                        break


            #-- data lines
            else:
                z.append(float(row[0]))
                data.append(float(row[1]))

            percent_complete = ir*100./nlines
            if sys.stdout.isatty():
                if ir%(nlines/10000) == 0:
                    print "\r{}: {:0.2f} %".format(os.path.basename(file_in),
                                                   percent_complete),
            else:
                if ir%(nlines/200) == 0:
                    print "{}: {:0.2f} %".format(os.path.basename(file_in),
                                                 percent_complete)

        #-- finish off last cast
        make_cast_file(z,data,varname,meta,castnumber,file_out,standard_z)

    return file_cast


if __name__ == '__main__':
    import argparse
    import json

    #-- set defaults
    control_defaults = {
        'file_in':'',
        'output_dir':'',
        'max_depth':None,
        'variables':[],
        'clobber' : False}

    p = argparse.ArgumentParser(
        description='convert WOD native format to netCDF')
    p.add_argument('json_control',
                   default=control_defaults)
    p.add_argument('-f',dest='json_as_file',
                   action='store_true',default=False,
                   help='Interpret input as a file name')
    args = p.parse_args()
    if not args.json_as_file:
        control_in = json.loads(args.json_control)
    else:
        with open(args.json_control,'r') as fp:
            control_in = json.load(fp)
    control = control_defaults
    control.update(control_in)


    #-- begin
    file_in = control['file_in']
    variables = control['variables']
    output_dir = control['output_dir']
    max_depth = control['max_depth']
    clobber = control['clobber']

    #-- input file absent?
    if not os.path.join(file_in):
        print('missing input file: %s'%file_in)
        sys.exit(1)

    #-- subset z
    if max_depth is not None:
        standard_z_selection = (standard_z<=max_depth)
        standard_z = standard_z[standard_z_selection]

    #-- convert from WOD native to ASCII
    odir = os.path.join(output_dir,os.path.basename(file_in))
    if not os.path.exists(odir): call(['mkdir','-p',odir])

    #-- make single variable ascii format files
    file_asc = []
    for varname in variables:
        file_out = wod2asc(file_in,varname,clobber)
        file_asc.append(file_out)

    #-- check for sucess
    if not all(file_asc):
        print('ERROR')
        sys.exit(1)

    #-- compile netCDF files for each cast
    file_cast = {}
    for file_in,varname in zip(file_asc,variables):
        print('writing %s data to netCDF'%varname)
        file_cast = asc2nc(file_in=file_in,
                           varname=varname,
                           odir=odir,
                           standard_z=standard_z,
                           file_cast=file_cast)
