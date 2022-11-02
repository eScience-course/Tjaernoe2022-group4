import re
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

# Files from https://digital.csic.es/handle/10261/219679


def read_satellite_data_9D(start_year=2011, num_years=10):
    '''
        Function to read in satellite data
    Args:
        start_year  [int]   :   first year of data
        num_years   [int]   :   number of years in the data, default 10
    Returns:
        file_dict   [dict]  :   a dictionary containing the files
    '''
    years = np.array([start_year+i for i in range(num_years-1)])
    file_dict = {}
    for year in years:
        file_dict[year] = {}
        for i, file in enumerate(os.listdir(f'09D/{str(year)}')):
            file_dict[year][i] = f'09D/{str(year)}/{file}'
        
    return file_dict

def plot_sss_sat(sss):
    '''
        Plots the sea surface salinity from satellite data
    Args:
        sss    [DataArray]   :   An xarray contining data about the sea surface salinity
    '''
    sat_proj = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':sat_proj})
    sss.plot.pcolormesh(
        ax = ax,
        cbar_kwargs={
            'orientation':'vertical',
            'shrink':.8
            },
        transform=sat_proj,
        levels=10,
        )
    ax.gridlines(draw_labels=True)
    fig.tight_layout()

if __name__ == '__main__':
    a = read_satelite_data_9D(num_years=2)
    print(a[2011][1])