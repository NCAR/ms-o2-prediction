#! /usr/bin/env python
import argparse
import xarray as xr
import numpy as np

xr_open_ds = {'decode_coords' : False,
              'decode_times' : False}


def check_allclose(file1,file2,variable_list,rtol=1e-7,atol=0):
    ds1 = xr.open_dataset(file1,**xr_open_ds)
    ds2 = xr.open_dataset(file2,**xr_open_ds)


    #-- absolute(a - b) <= (atol + rtol * absolute(b))
    for v in variable_list:
        np.testing.assert_allclose(ds1[v],ds2[v],rtol=rtol,atol=atol)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Process timeseries files.')
    p.add_argument('file1', type = str)
    p.add_argument('file2',type = str)

    p.add_argument('-v', dest = 'variable_list',
                   default = [],
                   type = lambda kv: kv.split(','),
                   required = False,
                   help = 'variable list')

    p.add_argument('--rtol', dest = 'rtol',
                   default = 1e-7,
                   required = False)

    args = p.parse_args()

    check_allclose(args.file1,args.file2,args.atol)
