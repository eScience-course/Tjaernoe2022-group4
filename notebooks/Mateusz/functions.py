import re
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import s3fs

# Files from https://digital.csic.es/handle/10261/219679


def _files(year, path=''):
    '''
        A function for finding all satellite files for a certain year and sorting the files
    Args:
        year    [int]   :   The year
        path    [str]   :   Path to file
    Returns:
        files   [list]  :   A list with filenames for the given year, sorted by date
    '''
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    files = []
    for file in s3.ls(f'{path}{year}/'):
        files.append(file)
        
    files.sort()
    return files

def read_satellite_data(start_year=2011, num_years=10, path=''):
    '''
        Function to read in satellite data
    Args:
        start_year  [int]   :   first year of data
        num_years   [int]   :   number of years in the data, default 10
        path        [str]   :   Path to file
    Returns:
        file_dict   [dict]  :   a dictionary containing the files
    '''
    years = np.array([start_year+i for i in range(num_years)])
    file_dict = {}
    
    for year in years:
        file_dict[year] = []
        files = _files(year, path)
        for file in files:
            file_dict[year].append(f's3://{file}')
            
    return file_dict

def plot_sss_sat(sss, levels=10, extent=[-180,180,90,50], vmin=20, vmax=40):
    '''
        Plots the sea surface salinity from satellite data
    Args:
        sss    [DataArray]   :   An xarray contining data about the sea surface salinity
        levels [int]         :   Levels in the plot
        extent [list]        :   A list of the extent of the plot
        vmin   [float]       :   Minimum value to show
        vmax   [float]       :   Maximum value to show
    '''
    sat_proj = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':sat_proj})
    sss.plot.pcolormesh(
        ax = ax,
        cbar_kwargs={
            'orientation':'vertical',
            'shrink':.8
            },
        transform=ccrs.epsg(6931),
        levels=levels,
        #vmin=vmin,
        #vmax=vmax
        robust=True
        )
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')
    fig.tight_layout()
    
def plot_argo_positions(data, extent=[-180,180,90,65]):
    '''
        Function for plotting the positions of ARGO profiles
    Args:
        data       [DataArray]  :  The dataset which contains the positions of the profiles
        extent     [list]       :  A list of the extent of the plot
    '''
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    plt.scatter(data['LONGITUDE'], data['LATITUDE'], transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')
    fig.tight_layout()

def create_xr(file):
    '''
        Function for creating a single xarray dataset
    Args:
        file   [str]        :   Complete path to the file
    Returns:
        tmp    [DataArray]  :   The xarray dataset
    '''
    s3 = s3fs.S3FileSystem(key="K1CQ7M1DMTLUFK182APD", secret="3JuZAQm5I03jtpijCpHOdkAsJDNLNfZxBpM15Pi0", client_kwargs=dict(endpoint_url="https://rgw.met.no"))
    tmp = xr.open_dataset(s3.open(file), drop_variables=['sss_error', 'sss_anomaly', 'sss_flag']).squeeze()
    if tmp['x'].attrs['units']=='km':
       tmp['x'] = tmp['x']*1000
       tmp['x'].attrs['units'] = 'm'
    if tmp['y'].attrs['units']=='km':
       tmp['y'] = tmp['y']*1000
       tmp['y'].attrs['units'] = 'm'
    return tmp

def slicing_data(data, min_time='1950-01-01', max_time='2022-06-01', min_lon=-180, max_lon=180, min_lat=-90, max_lat=90):
    '''
        Function for slicing data by time, lon and lat
    Args:
        data       [DataArray]  :  The dataset to be sliced
        min_time   [str]        :  A string representing the earliest date
        max_time   [str]        :  A string representing the latest date
        min_lon    [float]      :  The smallest longitude
        max_lon    [float]      :  The largest longitude
        min_lat    [float]      :  The smallest latitude
        max_lat    [float]      :  The largest latitude
    '''
    ndata = data.where(
        (data['time'] > np.datetime64(min_time))
        & (data['time'] < np.datetime64(max_time)) 
        & (data['lon'] > min_lon)
        & (data['lon'] < max_lon)
        & (data['lat'] < min_lat)
        & (data['lat'] > max_lat),
        drop=True)
    return ndata

if __name__ == '__main__':
    path='escience2022/Antoine/ESA_SMOS_Arctic_Sea_Surface_Salinity/'
    a = read_satellite_data(num_years=2, path=path)
    print(a[2011][0])