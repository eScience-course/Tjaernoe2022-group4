import xarray as xr
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def interGali(ds,var):
    '''
        Interploting data on a regular lat,lon grid
        grid is built from latitudes and the longitudes on the lowest latitude
        Input : 
            ds : xarray dataset with latitude, longitude and variable "var" to be interpolated. 
                var, lat,long must be 1d (ie. not depend on time)
            var : string name of variable
        Returns a xarray dataArray with the gridded variable and lat lon as coords
    '''
    
    # sorting according to latitudes
    lalo = pd.MultiIndex.from_arrays([ds.latitude.values, ds.longitude.values]).sortlevel(level=0)[0] 
    # get longitudes of lowest latitude
    lowlat = lalo.values[0][0]
    lon1d = np.array([x[1] for x in lalo.values if x[0]==lowlat])
    # get latitudes
    lat1d = np.unique(ds.latitude.values)
    # Building regular grid of lat and lon
    lon, lat = np.meshgrid(lon1d,lat1d)

    # interpolation
    var_gridded = griddata(
            (ds.longitude.values, ds.latitude.values), #points
            ds[var].values, #data
            (lon, lat), #grid on which to interpolate
            method='nearest')
    
    return  xr.DataArray(
        var_gridded,
        dims=("lat","lon"),
        coords={
            "lat": lat1d,
            "lon": lon1d
        }
    )


def pltNPS(var):
    fig, ax = plt.subplots(figsize=(8,8),
                           subplot_kw={'projection':ccrs.NorthPolarStereo()}
                           # subplot_kw={'projection':ccrs.PlateCarree()
                                      )
    var.plot.pcolormesh( 
        ax = ax,
        cbar_kwargs={ 'orientation':'vertical', 'shrink':.8},
        transform=ccrs.PlateCarree(),
        robust=True
    )
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    fig.tight_layout()
    
    
    
def rect(ax, lola):
    ### rectangle
    lo1,lo2, la1,la2 = lola
    N=10
    ax.plot(np.linspace(lo1,lo2,N), la1*np.ones(N), 'k--', transform=ccrs.PlateCarree())
    ax.plot(np.linspace(lo1,lo2,N), la2*np.ones(N), 'k--', transform=ccrs.PlateCarree())
    ax.plot(lo1*np.ones(N), np.linspace(la1,la2,N), 'k--', transform=ccrs.PlateCarree())
    ax.plot(lo2*np.ones(N), np.linspace(la1,la2,N), 'k--', transform=ccrs.PlateCarree())
