import re
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import s3fs
from joblib import Parallel, delayed
from ipywidgets import interactive, interact

# Files from https://digital.csic.es/handle/10261/219679
def interactive_plot_whole_region(ds, title):
    '''
        Function for creating an interactive plot of the whole region
    Args:
        ds    [DataArray] : DataArray containing salinity data
        title [str]       : A string to be used as figure title
    '''
    times = np.arange(365)

    def g(time=250):
        
        ttl = f'{title} \n{str(ds.sss[time].time.values)[0:10]}'
        sat_proj = ccrs.NorthPolarStereo()
        fig, ax = plt.subplots(figsize=(5,5), dpi=150, subplot_kw={'projection':sat_proj})
        ds['sss'][time].plot.pcolormesh(
            ax = ax,
            cbar_kwargs={
                'orientation':'vertical',
                'shrink':.8,
                'label': 'Sea Surface Salinity [psu]'
                },
            transform=ccrs.epsg(6931),
            robust=True,
            cmap='RdYlGn'
            )
        plt.title(ttl)
        ax.set_extent([-180,180,90,50], ccrs.PlateCarree())
        ax.gridlines(draw_labels=True)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')
        ax.add_feature(cartopy.feature.RIVERS, zorder=1, edgecolor='blue')
        fig.tight_layout()
        plt.show()

    interactive_plot = interact(g, time=times)
    interactive_plot
    
def interactive_plot(SatRegion1, SatRegion2):
    '''
        A function which creates an interactive plot of the two regions
    Args:
        SatRegion1   [DataArray] : DataArray containing salinity data for the first region
        SatRegion2   [DataArray] : DataArray containing salinity data for the second region
    '''
    times = np.arange(365)

    def g(time=250):
        fig, axs = plt.subplots(1,2,figsize=(10,5), dpi=150, subplot_kw={'projection':ccrs.NorthPolarStereo()})

        SatRegion1['sss'][time].plot.pcolormesh(
            ax=axs[0],
            cbar_kwargs={
                'orientation':'vertical',
                'shrink':.8,
                'label': 'Sea Surface Salinity [psu]'
                },
            transform=ccrs.epsg(6931),
            vmin=15,
            vmax=35,
            cmap='RdYlGn')

        SatRegion2['sss'][time].plot.pcolormesh(
            ax=axs[1],
            cbar_kwargs={
                'orientation':'vertical',
                'shrink':.8,
                'label': 'Sea Surface Salinity [psu]'
                },
            transform=ccrs.epsg(6931),
            vmin=15,
            vmax=35,
            cmap='RdYlGn')

        axs[0].set_extent([-20,20,67,85], ccrs.PlateCarree())
        axs[1].set_extent([82,180,70,85], ccrs.PlateCarree())
        axs[0].gridlines(draw_labels=True)
        axs[1].gridlines(draw_labels=True)
        axs[0].coastlines()
        axs[0].add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')
        axs[1].coastlines()
        axs[1].add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')
        axs[0].add_feature(cartopy.feature.RIVERS, zorder=1, edgecolor='blue')
        axs[1].add_feature(cartopy.feature.RIVERS, zorder=1, edgecolor='blue')
        time1 = f'{str(SatRegion1.sss[time].time.values)[0:10]}'
        time2 = f'{str(SatRegion2.sss[time].time.values)[0:10]}'
        axs[0].set_title(f'Sea Surface Salinity Greenland Sea \n{time1}')
        axs[1].set_title(f'Sea Surface Salinity Laptev Sea \n{time2}')
        plt.tight_layout()
        plt.show()

    interactive_plot = interact(g, time=times)
    interactive_plot

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

def plot_sss_sat(sss, levels=10, extent=[-180,180,90,50], title=None):
    '''
        Plots the sea surface salinity from satellite data
    Args:
        sss    [DataArray]   :   An xarray contining data about the sea surface salinity
        levels [int]         :   Levels in the plot
        extent [list]        :   A list of the extent of the plot
        vmin   [float]       :   Minimum value to show
        vmax   [float]       :   Maximum value to show
    '''
    ttl = f'{title} \n{str(sss.time.values)[0:10]}'
    sat_proj = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(figsize=(5,5), dpi=150, subplot_kw={'projection':sat_proj})
    sss.plot.pcolormesh(
        ax = ax,
        cbar_kwargs={
            'orientation':'vertical',
            'shrink':.8,
            'label': 'Sea Surface Salinity [psu]'
            },
        transform=ccrs.epsg(6931),
        levels=levels,
        robust=True,
        cmap='RdYlGn'
        )
    plt.title(ttl)
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

def WeightedMean(ds,ac):
    try:
        RegionMean = ((ds * ac['areacello']).sum(dim=['i','j'])/ac['areacello'].sum()).compute()
    except: 
        RegionMean = ((ds * ac['areacello']).sum(dim=['x','y'])/ac['areacello'].sum()).compute()
    return RegionMean

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
    
def PlotSeasonality(ds):
    #total_area = np.sum(ac.areacello)
    mean = ds.groupby('time.month').mean('time').mean('i').mean('j').compute()
    print(mean)
    std = ds.groupby('time.month').std('time').std('i').std('j')
    #print(std)
    t = ds.mean('i').mean('j').compute().groupby('time.month')
    month = ds['time.month']
    years = np.arange(1950,2015,1)
    x = np.arange(1,13,1)

    fig, ax = plt.subplots()
    max_ = []
    min_ = []
    #for month in t.groups.keys():
    #    max_.append(np.max(t[month]))
    #    min_.append(np.min(t[month]))
    #ax.fill_between(x, min_, max_, color='seashell')
    #for month in t.groups.keys():
        #m = np.full(len(t[month]), month)
        #plt.scatter(m, t[month], color='peachpuff')
    plt.plot(x, mean, 'cadetblue', lw=5)
    #plt.plot(x, mean-std)
    #plt.plot(x, mean+std)
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
