#%%
from agmetpy.thornthwaite import *
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cftime
import datetime

def thornthwaite_etp(temp: xr.DataArray):
    temp.sum(dim='month')
    pass

if __name__ == '__main__':
    dataset = xr.Dataset(
        data_vars={
            'p': xr.DataArray([268, 218, 159, 81, 55, 31, 28, 25, 58, 139, 174, 298], dims=('month',)),
            'tavg': xr.DataArray([23.6, 23.6, 23.4, 22.0, 19.7, 18.7, 18.7, 20.9, 22.5, 23.3, 23.5, 23.3], dims=('month',)),
            'etp': xr.DataArray([120, 105, 101, 84, 63, 51, 54, 77, 96, 104, 116, 111], dims=('month',)),
        },
        coords={
            'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    weather = Weather(dataset, dim_name='month')
    for i, w in weather:
        print(i, w)
    
# %%
thornthwaite_evapotranspiration(np.minimum(dataset.tavg, -10))
# %%
