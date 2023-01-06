from scipy.interpolate import griddata
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
    lalo = pd.MultiIndex.from_arrays([ds.latitude.values, 
                                      ds.longitude.values]
                                    ).sortlevel(level=0)[0] 
    # get longitudes of lowest latitude
    lowlat = lalo.values[0][0]
    lon1d = np.array([x[1] 
                      for x in lalo.values if x[0]==lowlat])
    # get latitudes
    lat1d = np.unique(ds.latitude.values)
    # Building regular grid of lat and lon
    lon, lat = np.meshgrid(lon1d,lat1d)

    # interpolation
    var_gridded =griddata(
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

## Function to plot 2 panels maps
def plot_map_2panels(lon1,lat1,data1,title1,lon2,lat2,data2,title2,extent):
    mpl.rcParams.update({'font.size':10})
    
    fig, ax = plt.subplots(1,2,figsize=(6,5),subplot_kw={'projection'
                :ccrs.NorthPolarStereo()})
    MapL=ax[0].pcolormesh(lon1,lat1,data1,
        vmin=0,vmax=2.0e-6,          
        transform=ccrs.PlateCarree(),
            )
    ax[0].add_feature(cartopy.feature.LAND, zorder=1, 
                        edgecolor='black')
    ax[0].set_title(title1, fontsize = 10)
    ax[0].gridlines(draw_labels=True)
    ax[0].coastlines()
    ax[0].set_extent(extent, ccrs.PlateCarree())
    ax[0].add_feature(cartopy.feature.RIVERS, zorder=1, 
                  edgecolor='blue')

#    title1=ax[0].set_title('A) GLS - UKESM', fontsize = 10)
#    extent1 = ax[0].set_extent([-20, 20,65,90], ccrs.PlateCarree())
    MapR=ax[1].pcolormesh(lon2,lat2,data2, 
        vmin=0,vmax=2.0e-6,          
        transform=ccrs.PlateCarree(),
            )
    ax[1].add_feature(cartopy.feature.LAND, zorder=1, 
                edgecolor='black')
    ax[1].gridlines(draw_labels=True)
    ax[1].coastlines()
    ax[1].set_extent(extent, ccrs.PlateCarree())
    ax[1].set_title(title2, fontsize = 10)
    ax[1].add_feature(cartopy.feature.RIVERS, zorder=1, 
                  edgecolor='blue')

#fig.colorbar(pl_CMIP6,shrink=0.4)
############# COLORBAR properties ##############
# create the ax based on an ax dimensions
    axins = inset_axes(ax[1],
            width="6%",  
            height="100%",
            loc='right',
            borderpad=-3
                )

# add colorbar from ax
    cbar = fig.colorbar(MapL, cax=axins, orientation='vertical')
    cbar = fig.colorbar(MapR, cax=axins, orientation='vertical')
# style(axins)
    cbar.set_label('DMSOS (mol/m$^{-3}$)', fontsize=10)

# thickness of spines around the colorbar
    cbar.outline.set_linewidth(1)
    fig.tight_layout()
    def style_cbar(ax):
        ax.tick_params(axis='both', which='major', 
                   labelsize=10)
        ax.tick_params(axis='both', which='both', 
                   length=8, width=1, direction='out')
    style_cbar(axins)
    return (MapL,MapR)


## Function to calculate the linear regression 
## function to calculate the trend
from scipy import stats
def linreg(x,y):
    dict_ = {'x':x.values,'y':y.values}
    df_ = pd.DataFrame.from_dict(dict_)
    df_ = df_.dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_['x'],df_['y'])
    fitted_model = slope*df_['x'] + intercept
    return (df_['x'],df_['y'],fitted_model,slope,intercept)