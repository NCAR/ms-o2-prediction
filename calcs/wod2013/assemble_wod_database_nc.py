#! /usr/bin/env python
import os
import sys
from glob import glob
from subprocess import call
import xarray as xr

import task_manager as tm
import wod_asc2nc

#-----------------------------------------------------------------
#-- main
#-----------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import json

    #-- set defaults
    control_defaults = {
        'name_of_extraction':None,
        'variables':[],
        'output_dir':'',
        'file_in':'',
        'file_out':'',
        'max_depth':None,
        'clobber' : False}

    p = argparse.ArgumentParser(description='convert WOD native format to netCDF')
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
    FILE_OUT = control['file_out']
    name_of_extraction = control['name_of_extraction']
    variables = control['variables']
    output_dir = control['output_dir']
    max_depth = control['max_depth']
    clobber = control['clobber']

    #-- output file present?
    if os.path.exists(FILE_OUT):
        if clobber:
            call(['rm','-f',FILE_OUT])
        else:
            print('Existing file found:\n\t%s'%FILE_OUT)
            sys.exit(0)

    #-- input file absent?
    if not os.path.join(file_in):
        print('missing input file: %s'%file_in)
        sys.exit(1)

    #-- convert from WOD native to ASCII
    odir = os.path.join(output_dir,os.path.basename(file_in))
    if not os.path.exists(odir): call(['mkdir','-p',odir])


    script = './wod_asc2nc.py'
    kwargs = {'file_in':file_in,
              'output_dir':output_dir,
              'max_depth':max_depth,
              'clobber':clobber}

    for varname in variables:
        kwargs['variables'] = [varname]
        jid = tm.submit([script,'\'{0}\''.format(json.dumps(kwargs))])

    tm.wait()

    file_cast = {castnumber:
                  [os.path.join(odir,'wod2013_%012d.%s.nc'%(castnumber,v))
                   for v in variables]
                   for castnumber in
                   wod_asc2nc.asc_get_casts(file_in+'.'+variables[0])}

    file_cast_merged = []
    for castnumber,files in file_cast.items():
        file_out = os.path.join(odir,'wod2013_%012d.nc'%(castnumber))
        file_cast_merged.append(file_out)

        print('writing %s'%file_out)
        ds = xr.merge([xr.open_dataset(f) for f in files])
        ds.to_netcdf(file_out,unlimited_dims='cast')


    #-- compile one big file
    print('Generating:\n\t%s'%FILE_OUT)
    ds = xr.concat([xr.open_dataset(f) for f in file_cast_merged],
                   dim='cast')
    ds.to_netcdf(FILE_OUT,unlimited_dims='cast')
