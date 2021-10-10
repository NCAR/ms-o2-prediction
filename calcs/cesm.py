from glob import glob
import xarray as xr

xr_open_ds = {'chunks' : {'time':1},
              'decode_coords' : False,
              'decode_times' : False,
              'data_vars' : 'minimal'}

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def list_files(droot,case,stream,datestr,variable=None):
    '''Construct glob pattern and return list files.'''

    files = sorted(glob(patt))
    if not files:
        raise IOError(f'No files: {patt}')

    return files

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def open_dataset(format,
                 file_name_pattern,
                 variable_list=[],
                 **kwargs):

    '''Open dataset from CESM output.

    There are two formats:
    - multi_variable:
        - if variable_list has been specified, drop extraneous variables

    - single_variable:
        - require variable_list, merge all variables into dataset
    '''

    if isinstance(variable_list,str):
        variable_list = [variable_list]

    if format == 'multi_variable':

        files = sorted(glob(file_name_pattern.format(**kwargs)))
        if not files:
            raise IOError(f'No files: {file_name_pattern}')

        ds = xr.open_mfdataset(files,**xr_open_ds)
        if 'bounds' in ds['time'].attrs:
            tb_name = ds['time'].attrs['bounds']
        elif 'time_bound' in ds:
            tb_name = 'time_bound'
        else:
            raise ValueError('No time_bound variable found')

        if variable_list:
            keep_vars = ['time',tb_name]+variable_list
            drop_vars = [v for v in ds.variables if 'time' in ds[v].dims and v not in keep_vars]
            ds = ds.drop(drop_vars)

    elif format == 'single_variable':
        if not variable_list:
            raise ValueError(f'Format {format} requires variable_list.')

        ds = xr.Dataset()
        for variable in variable_list:
            files = sorted(glob(file_name_pattern.format(variable=variable,**kwargs)))
            if not files:
                raise IOError(f'No files: {file_name_pattern}')

            ds = xr.merge((ds,xr.open_mfdataset(files,**xr_open_ds)))

    else:
        raise ValueError(f'Uknown format: {format}')

    return ds
