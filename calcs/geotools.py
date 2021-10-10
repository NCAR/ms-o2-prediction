from __future__ import absolute_import, division, print_function

import numpy as np
import xarray as xr

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def weighted_rmsd(da_x,da_y,weights,avg_over_dims=[]):

    if not avg_over_dims:
        avg_over_dims = weights.dims

    #-- apply NaN mask
    valid = (da_x.notnull() & da_y.notnull())
    weights = weights.where(valid)

    return np.sqrt(weighted_mean((da_x-da_y)**2,weights,avg_over_dims,apply_nan_mask=False))


#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def weighted_cov(da_x,da_y,weights,avg_over_dims=[]):

    if not avg_over_dims:
        avg_over_dims = weights.dims

    #-- apply NaN mask
    valid = (da_x.notnull() & da_y.notnull())
    weights = weights.where(valid)

    mean_x = weighted_mean(da_x,weights,avg_over_dims,apply_nan_mask=False)
    mean_y = weighted_mean(da_y,weights,avg_over_dims,apply_nan_mask=False)

    dev_x = da_x - mean_x
    dev_y = da_y - mean_y

    return weighted_mean(dev_x*dev_y,weights,avg_over_dims,apply_nan_mask=False)

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def weighted_cor(da_x,da_y,weights,avg_over_dims=[]):
    return weighted_cov(da_x,da_y,weights,avg_over_dims) / \
           np.sqrt(weighted_cov(da_x,da_x,weights,avg_over_dims) *
                   weighted_cov(da_y,da_y,weights,avg_over_dims))

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def weighted_sum(da,weights,sum_over_dims=[]):

    if not sum_over_dims:
        sum_over_dims = weights.dims

    sum_over_dims_v = [k for k in sum_over_dims if k in da.dims]
    if not sum_over_dims_v:
        raise ValueError('Unexpected dimensions for variable {0}'.format(da.name))

    attrs = da.attrs.copy()
    encoding = da.encoding

    #-- compute weighted sum
    dao = (da * weights).sum(sum_over_dims_v)

    if 'units' in attrs and 'units' in weights.attrs:
        attrs['units'] = attrs['units']+' '+weights.attrs['units']

    for att in ['grid_loc','coordinates']:
        if att in attrs:
            del attrs[att]

    dao.attrs = attrs
    dao.encoding = {key:val for key,val in encoding.items() if key in ['_FillValue','dtype']}

    return dao

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def weighted_std(da,weights,avg_over_dims=[],apply_nan_mask=True,ddof=0):

    if not avg_over_dims:
        avg_over_dims = weights.dims

    avg_over_dims_v = [k for k in avg_over_dims if k in da.dims]
    if not avg_over_dims_v:
        raise ValueError(('Unexpected dimensions for variable {0}: {1}\n\n'
                          'Average over dimensions: {2}').format(da.name,da,avg_over_dims))

    attrs = da.attrs.copy()
    encoding = da.encoding

    #-- mask weights where field is missing
    if apply_nan_mask:
        weights = weights.where(da.notnull())
        np.testing.assert_allclose((weights/weights.sum(avg_over_dims_v)).sum(avg_over_dims_v),1.)

    da_mean = weighted_mean(da,weights,avg_over_dims,apply_nan_mask=False)
    dao = np.sqrt((weights * (da - da_mean)**2).sum(avg_over_dims_v) / (weights.sum(avg_over_dims_v) - ddof))

    for att in ['grid_loc','coordinates']:
        if att in attrs:
            del attrs[att]

    dao.attrs = attrs
    dao.encoding = {key:val for key,val in encoding.items() if key in ['_FillValue','dtype']}

    return dao

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def weighted_mean(da,weights,avg_over_dims=[],apply_nan_mask=True):

    if not avg_over_dims:
        avg_over_dims = weights.dims

    avg_over_dims_v = [k for k in avg_over_dims if k in da.dims]
    if not avg_over_dims_v:
        raise ValueError(('Unexpected dimensions for variable {0}: {1}\n\n'
                          'Average over dimensions: {2}').format(da.name,da,avg_over_dims))

    attrs = da.attrs.copy()
    encoding = da.encoding

    #-- mask weights where field is missing (takes time)
    if apply_nan_mask:
        weights = weights.where(da.notnull())
        np.testing.assert_allclose((weights/weights.sum(avg_over_dims_v)).sum(avg_over_dims_v),1.)

    dao = (da * weights).sum(avg_over_dims_v) / weights.sum(avg_over_dims_v)

    for att in ['grid_loc','coordinates']:
        if att in attrs:
            del attrs[att]

    dao.attrs = attrs
    dao.encoding = {key:val for key,val in encoding.items() if key in ['_FillValue','dtype']}

    return dao
