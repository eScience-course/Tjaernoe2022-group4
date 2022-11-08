import re
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import s3fs
from joblib import Parallel, delayed

# Files from https://digital.csic.es/handle/10261/219679

def collect_satellite_data(years=[2011,2012,2013,2014,2015,2016,2017,2018,2019], path='escience2022/Antoine/ESA_SMOS_Arctic_Sea_Surface_Salinity/'):
    '''
        Function for collecting satellite data into one DataArray
    Args:
        years    [list]       :   A list of years included
        path     [str]        :   path to the files containing the data
    returns:
        sat_data [DataArray]  :   A DataArray containing all the data.
    '''
    sat_files = _read_satellite_data(num_years=len(years), path=path)
    tmp = []
    for year in years:
        tmp.append(xr.concat(Parallel(n_jobs=16)(delayed(create_xr)(file) for file in sat_files[year]), dim='time'))
    sat_data = xr.concat(tmp, dim='time')
    del(tmp)
    return sat_data
    
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

def _read_satellite_data(start_year=2011, num_years=10, path=''):
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

def ConvertModelGrid(_ds):
    """
    convert longitude from 0-360 to -180 -- 180 deg
    """

    if 'i' in _ds.coords:
        _ds = _ds.rename({'i': 'lon'})
    if 'j' in _ds.coords:
        _ds = _ds.rename({'j': 'lat'})
    
    if _ds['lon'].attrs['units']=='1':
       _ds['lon'] = _ds['lon']*360/(len(_ds['lon']))
       _ds['lon'].attrs['units'] = 'degrees east'
    if _ds['lat'].attrs['units']=='1':
       _ds['lat'] = _ds['lat']*180/(len(_ds['lat']))
       _ds['lat'].attrs['units'] = 'degrees north'
    
    # check if already 
    
    attrs = _ds['lon'].attrs
    if _ds['lon'].min() >= 0:
        with xr.set_options(keep_attrs=True): 
            _ds.coords['lon'] = (_ds['lon'] + 260) % 360 - 180
        _ds = _ds.sortby('lon')
    
    attrs = _ds['lat'].attrs
    if _ds['lat'].min() >= 0:
        with xr.set_options(keep_attrs=True):
            _ds.coords['lat'] = (_ds['lat'] + 90) % 180 - 90 
        _ds = _ds.sortby('lat')
    
    return _ds


def slice_data(data, min_time='1950-01-01', max_time='2022-06-01', min_lon=-180, max_lon=180, min_lat=-90, max_lat=90):
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
    Returns:
        ndata      [DataArray]  :  A sliced DataArray
    '''
    try:
        ndata = data.where(
            (data['time'] > np.datetime64(min_time))
            & (data['time'] < np.datetime64(max_time)) 
            & (data['lon'] > min_lon)
            & (data['lon'] < max_lon)
            & (data['lat'] > min_lat)
            & (data['lat'] < max_lat),
            drop=True)
    except:
        ndata = data.where(
            (data['longitude'] > min_lon)
            & (data['longitude'] < max_lon)
            & (data['latitude'] > min_lat)
            & (data['latitude'] < max_lat),
            drop=True)
    return ndata

def PlotModel(data, extent=[-180,180,90,50]):
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection':ccrs.NorthPolarStereo()})
    plt.pcolormesh(
        data['longitude'],
        data['latitude'],
        data['so'],
        transform=ccrs.PlateCarree()
        )
    ax.set_extent(extent, ccrs.PlateCarree())
    plt.colorbar()
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    fig.tight_layout()
    
def PlotSeasonality(ds, ac):
    #total_area = np.sum(ac.areacello)
    mean = ds['so'].groupby('time.month').mean('time').mean('i').mean('j').compute()
    std = ds['so'].groupby('time.month').std('time').std('i').std('j')
    #print(std)
    t = ds['so'].mean('i').mean('j').compute().groupby('time.month')
    month = ds['time.month']
    years = np.arange(1950,2015,1)
    x = np.arange(1,13,1)

    fig, ax = plt.subplots()
    max_ = []
    min_ = []
    for month in t.groups.keys():
        max_.append(np.max(t[month]))
        min_.append(np.min(t[month]))
    ax.fill_between(x, min_, max_, color='seashell')
    for month in t.groups.keys():
        m = np.full(len(t[month]), month)
        plt.scatter(m, t[month], color='peachpuff')
    plt.plot(x, mean, 'cadetblue', lw=5)
    plt.plot(x, mean-std)
    plt.plot(x, mean+std)
    plt.grid()
    ticks = [1,2,3,4,5,6,7,8,9,10,11,12]

    dic = { 1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    plt.title(f'Seasonality of Ocean Salinity at {ds.lev:.2f}m depth')
    plt.show()
    
if __name__ == '__main__':
    path='escience2022/Antoine/ESA_SMOS_Arctic_Sea_Surface_Salinity/'
    a = read_satellite_data(num_years=2, path=path)
    print(a[2011][0])
