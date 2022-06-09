import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_columns', None)
fsize = 20
plt.rcParams.update({'font.size': fsize})


#define plotting function
def plotmap(arr, ax, label='', colormapname='RdBu',  normalize=True, vmin=None, vmax=None):
    datalat=np.load('data/EASE_lat.npy')
    datalon=np.load('data/EASE_lon.npy')
    
    #clip area
    datalat = datalat[35:-30, 50:-33]
    datalon = datalon[35:-30, 50:-33]
    arr = arr[35:-30, 50:-33]
    
    #colormap gray
    colormap = plt.get_cmap(colormapname)
    colormap.set_bad('lightgray')
    
    ax.set_extent([np.min(datalon), np.max(datalon), np.min(datalat), np.max(datalat)])
    ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face',facecolor='white')
    ax.add_feature(ocean_50m, edgecolor='k')

    if vmin==None:
        high = np.max([abs(np.nanmax(arr)), abs(np.nanmin(arr))])
        im = ax.pcolormesh(datalon, datalat, arr, cmap=colormap, vmin=-high, vmax=high)
    elif np.isfinite(vmin):
        im = ax.pcolormesh(datalon, datalat, arr, cmap=colormap, vmin=vmin, vmax=vmax)
    
    #colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom',pad=0.05,size='5%', axes_class=plt.Axes)
    cb = fig.colorbar(im,cax=cax, orientation='horizontal')
    cb.set_label(label)
    cb.ax.tick_params(labelsize=fsize, length=4, width=2, )
    #other
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    
    ax.text(x=0.7, y=0.4, s='Borneo', transform=ax.transAxes, fontsize=fsize-4)
    ax.text(x=0.1, y=0.32, s='Sumatra', transform=ax.transAxes, fontsize=fsize-4)
    

# ====================== MAPS ==========================

# ========== Climate variables

# function to load climate data
def load_climate(rcm, period):
    
    ## Setup climate variables
    if period == 'fut':
        root = 'data/climate/{}_{}85_{}.npy'
    elif period == 'ref':
        root = 'data/climate/{}_{}RF_{}.npy' 
    else:
        print('need to specify fut or ref for period')
        return None
    
    #annual entropy
    ann_ent_pre = np.load(root.format('annent', rcm, 'pr'))
    
    #annual mean
    ann_mean_pre = np.load(root.format('annmean', rcm, 'pr'))
    
    #annual standard deviation
    ann_std_pre = np.load(root.format('annstd', rcm, 'pr'))
    ann_std_tem = np.load(root.format('annstd', rcm, 'tas'))
    ann_std_pet = np.load(root.format('annstd', rcm, 'pet'))
    
    #dry season mean
    dry_mean_pre = np.load(root.format('drymean', rcm, 'pr'))
    dry_mean_tem = np.load(root.format('drymean', rcm, 'tas'))
    dry_mean_pet = np.load(root.format('drymean', rcm, 'pet'))
    
    #output variables:
    #annual entropy precipitation, annual mean precipitation
    #annual standard deviation of precipitation, annual standard dev temperature, annual std dev PET
    #dry season mean precipitation, temperature, and PET
    return ann_ent_pre, ann_mean_pre, ann_std_pre, ann_std_tem, ann_std_pet, dry_mean_pre, dry_mean_tem, dry_mean_pet


HA_fut = load_climate('HA','fut') #8x20x198x276
MP_fut = load_climate('MP','fut')
NO_fut = load_climate('NO','fut')
HA_ref = load_climate('HA','ref')
MP_ref = load_climate('MP','ref')
NO_ref = load_climate('NO','ref')

fut = np.concatenate([HA_fut, MP_fut, NO_fut], axis=1) #8x60x198x276
ref = np.concatenate([HA_ref, MP_ref, NO_ref], axis=1)


#calculate median over time & model, then get difference between periods
clim_fut = np.nanmedian(fut, axis=1)
clim_ref = np.nanmedian(ref, axis=1)
clim_diff = clim_fut - clim_ref  #8x198x276


#take out variables
ann_ent_pre_diff = clim_diff[0]
ann_mean_pre_diff = clim_diff[1]
dry_mean_pet_diff = clim_diff[-1]
dry_mean_tem_diff = clim_diff[-2]
dry_mean_pre_diff = clim_diff[-3]

ann_ent_pre_fut = clim_fut[0]
ann_mean_pre_fut = clim_fut[1]
dry_mean_pet_fut = clim_fut[-1]
dry_mean_tem_fut = clim_fut[-2]
dry_mean_pre_fut = clim_fut[-3]






def av(arr): return np.nanmean(arr, axis=0)


#plot
fig = plt.figure(figsize=(20,12))
gs = fig.add_gridspec(ncols = 2, nrows = 2)


#plot differences
#arr, ax, label='', colormapname='RdBu',  normalize=True
#DRY SEASON PRECIP
ax1 = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
plotmap(dry_mean_pre_diff, ax1, label='$\Delta$ Dry season precipitation (mm/day)', vmin=-2, vmax=2)
#DRY SEASON PET
ax2 = fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())
plotmap(dry_mean_pet_diff, ax2, colormapname='RdBu_r', label='$\Delta$ Dry season PET (mm/day)', vmin=-2, vmax=2)#, normalize=False)
#ANNUAL PRECIP
ax3 = fig.add_subplot(gs[1,0],projection=ccrs.PlateCarree())
plotmap(ann_mean_pre_diff * 365 /10, ax3, label='$\Delta$ Annual precipitation (cm/yr)')
#PRECIP ENTROPY
ax4 = fig.add_subplot(gs[1,1],projection=ccrs.PlateCarree())
plotmap(ann_ent_pre_diff, ax4, label='$\Delta$ Precipitation entropy (-)', )

#plotmap(av(dry_mean_pre_diff), '$\Delta$ Dry season precipitation')

#latitude ticks
lat_formatter = LatitudeFormatter()
ax1.set_yticks([-3,0,3], crs=ccrs.PlateCarree())
ax2.set_yticks([-3,0,3], crs=ccrs.PlateCarree())
ax3.set_yticks([-3,0,3], crs=ccrs.PlateCarree())
ax4.set_yticks([-3,0,3], crs=ccrs.PlateCarree())
ax1.yaxis.set_major_formatter(lat_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)
ax4.yaxis.set_major_formatter(lat_formatter)
ax1.tick_params(direction='in', length=6, width=2)
ax2.tick_params(direction='in', length=6, width=2)
ax3.tick_params(direction='in', length=6, width=2)
ax4.tick_params(direction='in', length=6, width=2)




ax1.text(-.04,0.99,'a', transform = ax1.transAxes, weight='bold')
ax2.text(-.04,0.99,'b', transform = ax2.transAxes, weight='bold')
ax3.text(-.04,0.99,'c', transform = ax3.transAxes, weight='bold')
ax4.text(-.04,0.99,'d', transform = ax4.transAxes, weight='bold')

#plt.tight_layout()
plt.show()





### calculate percentage area
print('pct decreasing precip: '+str(np.count_nonzero(dry_mean_pre_diff[dry_mean_pre_diff<0])/1800))
print('pct decreasing PET: '+str(np.count_nonzero(dry_mean_pet_diff[dry_mean_pet_diff>0])/1800))


