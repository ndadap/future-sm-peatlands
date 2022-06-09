
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from scipy.stats import linregress

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 16})

import matplotlib as mpl
from matplotlib import patches as mpatches
import matplotlib.transforms as mtransforms


# fn that plots and returns map from dataframe in ease grid array
def tomap(df, var):
    
    #load lat, lon and data
    lats = np.load('data/ease2_9km_lats_ISEA.npy')
    lons = np.load('data/ease2_9km_lons_ISEA.npy')
    
    peatarea9 = np.load('data/peatarea_9km.npy')
    peatarea9[peatarea9==0]=np.nan
    
    #initialize
    varmap = np.full((len(lats),len(lons)), np.nan)
    
    #iterate over each row in dataframe
    for i in range(len(df)):
        row = df.iloc[i]
        latidx = int(row.latidx)
        lonidx = int(row.lonidx)
        
        #put data into map        
        varmap[latidx,lonidx] = row[var]
    
#    #plot
#    plt.figure(figsize=(6,6))    
#    im = plt.imshow(varmap*peatarea9)
#    plt.colorbar(im,fraction=0.032, pad=0.04, label=var)
#    plt.show()
#    
    return varmap



### observed values

#load data
dfall = pd.read_pickle('data/trainingdata.pkl') #see temporal_datasets.py

#covert lowsmpct to pct
dfall['lowsmpct'] = dfall['lowsmpct']*100

#remove other lut
dfall = dfall.loc[np.isfinite(dfall.lut2015)]
dfall = dfall.loc[dfall.lut2015!=0]

#split into elnino and normal years
dfnino = dfall.loc[(dfall.year==2015) | (dfall.year==2019)]
dfnorm = dfall.loc[(dfall.year!=2015) & (dfall.year!=2019)]
df2015 = dfall.loc[dfall.year==2015]
df2019 = dfall.loc[dfall.year==2019]

#%% model predictions

home = 'data/model_outputs/'
ref_msm = np.load(home+'ref_00_meansm.npy')
ref_lpc = np.load(home+'ref_00_lowpct.npy')*100
fut_msm = np.load(home+'fut_00_meansm.npy')
fut_lpc = np.load(home+'fut_00_lowpct.npy')*100

def av(arr): return np.nanmean(arr.reshape((60,198,276)), axis=0)
#climate shifts (percent)
dmsm_clim = (av(fut_msm) - av(ref_msm))
dlpc_clim = (av(fut_lpc) - av(ref_lpc))
#calculate elnino shift map
map_2015_msm = tomap(dfall.loc[dfall.year==2015], 'drymeansm')
map_2016_msm = tomap(dfall.loc[dfall.year==2016], 'drymeansm')
map_2017_msm = tomap(dfall.loc[dfall.year==2017], 'drymeansm')
map_2018_msm = tomap(dfall.loc[dfall.year==2018], 'drymeansm')
map_2019_msm = tomap(dfall.loc[dfall.year==2019], 'drymeansm')
map_2020_msm = tomap(dfall.loc[dfall.year==2020], 'drymeansm')
map_2015_lpc = tomap(dfall.loc[dfall.year==2015], 'lowsmpct')
map_2016_lpc = tomap(dfall.loc[dfall.year==2016], 'lowsmpct')
map_2017_lpc = tomap(dfall.loc[dfall.year==2017], 'lowsmpct')
map_2018_lpc = tomap(dfall.loc[dfall.year==2018], 'lowsmpct')
map_2019_lpc = tomap(dfall.loc[dfall.year==2019], 'lowsmpct')
map_2020_lpc = tomap(dfall.loc[dfall.year==2020], 'lowsmpct')

norm_msm = np.nanmean(np.stack((map_2016_msm, map_2017_msm,map_2018_msm,map_2019_msm)), axis=0)
nino_msm = np.nanmean(np.stack((map_2015_msm, map_2019_msm)), axis=0)
norm_lpc = np.nanmean(np.stack((map_2016_lpc, map_2017_lpc,map_2018_lpc,map_2019_lpc)), axis=0)
nino_lpc = np.nanmean(np.stack((map_2015_lpc, map_2019_lpc)), axis=0)

dmsm_nino = (nino_msm-norm_msm)
dlpc_nino = (nino_lpc-norm_lpc)

#remove really high or low values which arise from small divisor
def clean(arr):
    arr[(arr>1e3) | (arr<-1e3)] = np.nan #remove infinity values
    return arr
dlpc_clim = clean(dlpc_clim)
dlpc_nino = clean(dlpc_nino)






#%% PLOT V2
fig = plt.figure(figsize=(30,15))
gs = fig.add_gridspec(ncols = 2, nrows = 1)

def clip(arr): return arr[35:-30, 50:-33]
datalat = clip(np.load('data/EASE_lat.npy'))
datalon = clip(np.load('data/EASE_lon.npy'))


#setup plot for MSM
msm_cmap = mpl.cm.get_cmap('RdBu')
msm_cmap.set_bad(color='lightgray')

ax1 = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax1.set_extent([np.min(datalon), np.max(datalon), np.min(datalat), np.max(datalat)])
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face',facecolor='white')
ax1.add_feature(ocean_50m, edgecolor='k')
#data
divnorm=colors.TwoSlopeNorm(vmin=-.15, vcenter=0., vmax=.15)
im = ax1.pcolormesh(datalon, datalat, clip(dmsm_clim) - clip(dmsm_nino), norm=divnorm,  cmap=msm_cmap)
#colorbar
divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom',pad=0.07,size='3%', axes_class=plt.Axes)
cb = fig.colorbar(im,cax=cax, orientation='horizontal')
cb.ax.tick_params(labelsize=32, length=5, width=2)
ax1.set_title(r"$\Delta SM^{\bf climate}_{dry season} - \Delta SM^{\bf El Nino}_{dry season} (cm^3/cm^3)$", size=32)
#other
ax1.set_xticks([],[])
#label
# ax1.text(-.05,-.25,'Clim Chg\ngreater', transform = ax1.transAxes, weight='bold',size=20)
# ax1.text(.95,-.25,'El Nino\ngreater', transform = ax1.transAxes, weight='bold',size=20)
#latitude ticks
lat_formatter = LatitudeFormatter()
ax1.set_yticks([-3,0,3], crs=ccrs.PlateCarree())
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.tick_params(direction='in', length=10, width=4, labelsize=32)

#arrows
arrow_style ="simple, head_length = 60,\
 head_width = 80, tail_width = 60"
rect_style ="simple, tail_width = 25"
line_style ="simple, tail_width = 1"
trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)

#right arrow
arrow1 = mpatches.FancyArrowPatch((0.5, -7),(0, -7), arrowstyle = arrow_style,transform = trans, color='#C84640')
arrow1.set_clip_on(False)
ax1.add_patch(arrow1)
#right arrow text
ax1.text(.25, -.25, "Climate Change\nImpact Greater", ha="center", va="center", size=24, weight='bold',c='white', transform = ax1.transAxes)

#left arrow
arrow2 = mpatches.FancyArrowPatch((0.5, -7),(1, -7), arrowstyle = arrow_style,transform = trans, color='#3783BB')
arrow2.set_clip_on(False)
ax1.add_patch(arrow2)
#right arrow text
ax1.text(.75, -.25, "El Nino\nImpact Greater", ha="center", va="center", size=24, weight='bold',c='white', transform = ax1.transAxes)





#setup plot for LPC
lpc_cmap = mpl.cm.get_cmap('RdBu_r')
lpc_cmap.set_bad(color='lightgray')

ax2 = fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())
ax2.set_extent([np.min(datalon), np.max(datalon), np.min(datalat), np.max(datalat)])
ax2.add_feature(ocean_50m, edgecolor='k')
#data
divnorm=colors.TwoSlopeNorm(vmin=-30, vcenter=0., vmax=30)
im2 = ax2.pcolormesh(datalon, datalat, clip(dlpc_clim) - clip(dlpc_nino), norm=divnorm,  cmap=lpc_cmap)
#colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes('bottom',pad=0.07,size='3%', axes_class=plt.Axes)
cb = fig.colorbar(im2,cax=cax, orientation='horizontal')
cb.ax.tick_params(labelsize=32, length=5, width=2)
ax2.set_title(r"$\Delta pct^{\bf climate}_{low sm} (\%) - \Delta pct^{\bf El Nino}_{low sm} (\%)$", size=32)
#other
ax2.set_xticks([],[])
ax2.set_yticks([],[])
#label
# ax2.text(-.05,-.25,'El Nino\ngreater', transform = ax2.transAxes, weight='bold',size=20)
# ax2.text(.95,-.25,'Clim Chg\ngreater', transform = ax2.transAxes, weight='bold',size=20)
#latitude ticks
lat_formatter = LatitudeFormatter()
ax2.set_yticks([-3,0,3], crs=ccrs.PlateCarree())
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.tick_params(direction='in', length=10, width=4, labelsize=32)

#arrows
arrow_style ="simple, head_length = 60,\
 head_width = 80, tail_width = 60"
rect_style ="simple, tail_width = 25"
line_style ="simple, tail_width = 1"
trans = mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData)

#right arrow
arrow1 = mpatches.FancyArrowPatch((0.5, -7),(1, -7), arrowstyle = arrow_style,transform = trans, color='#C84640')
arrow1.set_clip_on(False)
ax2.add_patch(arrow1)
#right arrow text
ax2.text(.75, -.25, "Climate Change\nImpact Greater", ha="center", va="center", size=24, weight='bold',c='white', transform = ax2.transAxes)

#left arrow
arrow2 = mpatches.FancyArrowPatch((0.5, -7),(0, -7), arrowstyle = arrow_style,transform = trans, color='#3783BB')
arrow2.set_clip_on(False)
ax2.add_patch(arrow2)
#right arrow text
ax2.text(.25, -.25, "El Nino\nImpact Greater", ha="center", va="center", size=24, weight='bold',c='white', transform = ax2.transAxes)




#labels
ax1.text(-.04,0.99,'a', transform = ax1.transAxes, weight='bold', size=24)
ax2.text(-.04,0.99,'b', transform = ax2.transAxes, weight='bold', size=24)

plt.show()


