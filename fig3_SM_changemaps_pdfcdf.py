import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns

from scipy import io
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import os
import shutil
from sklearn.model_selection import KFold
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms


from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 36})

def cl(arr): return arr[np.isfinite(arr)]

#define plotting function
def plotmap(arr, label='', colormapname='RdBu',  normalize=True):
    datalat=np.load('data/EASE_lat.npy')
    datalon=np.load('data/EASE_lon.npy')
    
    #clip area
    datalat = datalat[35:-30, 50:-33]
    datalon = datalon[35:-30, 50:-33]
    arr = arr[35:-30, 50:-33]
    
    #colormap gray
    colormap = plt.get_cmap(colormapname)
    colormap.set_bad('lightgray')
    
    fig,ax = plt.subplots(figsize=(15,12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([np.min(datalon), np.max(datalon), np.min(datalat), np.max(datalat)])
    ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face',facecolor='white')
    ax.add_feature(ocean_50m, edgecolor='k')
    #plot data
    if normalize==True:
        norm = mpl.colors.DivergingNorm(vmin=np.nanmin(arr), vmax = np.nanmax(arr), vcenter=0)
        im = ax.pcolormesh(datalon, datalat, arr, cmap=colormap, norm=norm)
    else:
        im = ax.pcolormesh(datalon, datalat, arr, cmap=colormap)
    #colorbar
    cax,kw = mpl.colorbar.make_axes(ax,location='right',pad=0.01,shrink=0.59)
    cbar=fig.colorbar(im,cax=cax,**kw)
    cbar.set_label(label)
    #other
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    plt.show()
    

# ====================== MAPS ==========================

# ========== SM variables
home = 'data/model_outputs/'
ref_msm = np.load(home+'ref_00_meansm.npy')
ref_lpc = np.load(home+'ref_00_lowpct.npy')*100
fut_msm = np.load(home+'fut_00_meansm.npy')
fut_lpc = np.load(home+'fut_00_lowpct.npy')*100

ref_lpc[ref_lpc<0] = 0
fut_lpc[fut_lpc<0] = 0

# ===== difference
def av(arr): return np.nanmean(np.nanmean(arr, axis=0), axis=0)
msmdiff = av(fut_msm)-av(ref_msm)
lpcdiff = av(fut_lpc)-av(ref_lpc)

#plotmap(msmdiff, '$\Delta$ sm$_{dry season}$ (m$^3$/$m^3)$')
#plotmap(lpcdiff, '$\Delta pct_{low sm}$ (%)')



fig = plt.figure(figsize=(30,20))
widths = [5,3.5]
heights = [1,1]
gs = fig.add_gridspec(ncols = 2, nrows = 2, width_ratios = widths, height_ratios = heights)

colormapname='RdBu'
normalize = True

# ======================================== MSM Map
arr = msmdiff
datalat=np.load('data/EASE_lat.npy')
datalon=np.load('data/EASE_lon.npy')

#clip area
datalat = datalat[35:-30, 50:-33]
datalon = datalon[35:-30, 50:-33]
arr = arr[35:-30, 50:-33]

#colormap gray
colormap = plt.get_cmap(colormapname)
colormap.set_bad('lightgray')

ax1 = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax1.set_extent([np.min(datalon), np.max(datalon), np.min(datalat), np.max(datalat)])
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face',facecolor='white')
ax1.add_feature(ocean_50m, edgecolor='k')
#plot data
norm = mpl.colors.TwoSlopeNorm(vmin=np.nanmin(arr), vmax = np.nanmax(arr), vcenter=0)
im = ax1.pcolormesh(datalon, datalat, arr, cmap=colormap, norm=norm)
#colorbar
#cax,kw = mpl.colorbar.make_axes(ax1,location='bottom',pad=0.01,shrink=0.7, aspect=30)
#cbar=fig.colorbar(im,cax=cax,**kw)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom',pad=0.03,size='3%', axes_class=plt.Axes)
cb = fig.colorbar(im,cax=cax, orientation='horizontal')
cb.set_label('$\Delta$ sm$_{dry}$ $_{season}$ (cm$^3$/cm$^3$)')
cb.ax.tick_params(reset=True, labelsize=32, length=5, width=2)
#ticklocs = cb.get_ticks
#cb.ax.tick_params(axis='x', direction='in', len)
#other
ax1.set_xticks([],[])
ax1.set_yticks([],[])
#ax1.set_title('$\Delta$ sm$_{dry season}$ (m$^3$/$m^3)$')


# #annotations
# #Sebangau
# rect = patches.Rectangle((113.25, -3), 1.2,1.1, linewidth=3, edgecolor='black', facecolor='none', linestyle = '--')
# ax1.add_patch(rect)
# ax1.text(112, -1.5, 'Sebangau', fontsize=28)

# #Western Sarawak
# rect = patches.Rectangle((111.4, 1.9), .8,.8, linewidth=3, edgecolor='black', facecolor='none', linestyle = '--')
# ax1.add_patch(rect)
# ax1.text(112, 1.4, 'W Sarawak', fontsize=28)

# ##Riau
# #rect = patches.Rectangle((102.2, -0.7), 1.5,1.2, linewidth=3, edgecolor='black', facecolor='none', linestyle = '--')
# #ax1.add_patch(rect)
# #ax1.text(101, -1.2, 'Riau', fontsize=28)





# ============================ LPC Map
arr = lpcdiff
datalat=np.load('data/EASE_lat.npy')
datalon=np.load('data/EASE_lon.npy')

#clip area
datalat = datalat[35:-30, 50:-33]
datalon = datalon[35:-30, 50:-33]
arr = arr[35:-30, 50:-33]

#colormap gray
colormap = plt.get_cmap(colormapname)
colormap.set_bad('lightgray')


ax3 = fig.add_subplot(gs[1,0],projection=ccrs.PlateCarree())
ax3.set_extent([np.min(datalon), np.max(datalon), np.min(datalat), np.max(datalat)])
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face',facecolor='white')
ax3.add_feature(ocean_50m, edgecolor='k')
#plot data
norm = mpl.colors.TwoSlopeNorm(vmin=np.nanmin(arr), vmax = np.nanmax(arr), vcenter=0)
im = ax3.pcolormesh(datalon, datalat, arr, cmap=colormap, norm=norm)
#colorbar
divider = make_axes_locatable(ax3)
cax = divider.append_axes('bottom',pad=0.03,size='3%', axes_class=plt.Axes)
cb = fig.colorbar(im,cax=cax, orientation='horizontal')
cb.ax.tick_params(reset=True, labelsize=32, length=5, width=2)
cb.set_label('$\Delta$ pct$_{low}$ $_{sm}$ (% year)')
#other
ax3.set_xticks([],[])
ax3.set_yticks([],[])
#ax3.set_title('$\Delta pct_{low low}sm$ (%)')

# #annotations
# #N Sumatra
# rect = patches.Rectangle((100.7, 0), 2,2, linewidth=3, edgecolor='black', facecolor='none', linestyle = '--', clip_on=False)
# ax3.add_patch(rect)
# ax3.text(100.5, -1.25, 'North\nSumatra', fontsize=28)

# # W Kalimantan
# rect = patches.Rectangle((111.75, 0.5), 1.3,.8, linewidth=3, edgecolor='black', facecolor='none', linestyle = '--')
# ax3.add_patch(rect)
# ax3.text(112, 1.6, 'W Kalimantan', fontsize=28)

# # C Kalimantan
# rect = patches.Rectangle((111.8, -3.5), 3.2,2, linewidth=3, edgecolor='black', facecolor='none', linestyle = '--')
# ax3.add_patch(rect)
# ax3.text(111, -1.3, 'C Kalimantan', fontsize=28)








#========================================== MSM PDF
color_list = ['cornflowerblue', 'sandybrown']
labels = ['1985-2005', '2040-2060']
#setup plot

ax2 = fig.add_subplot(gs[0,1])
#between reference and future
for j, arr in enumerate([ref_msm, fut_msm]):
    #individual RCM curves
    for i in range(3):
        sns.kdeplot(cl(arr[i]), color=color_list[j], linewidth=3, alpha=0.5, ax=ax2, bw_adjust=1.25)
    #plot pdf of all models
    sns.kdeplot(cl(arr), color=color_list[j], linewidth=4, label=labels[j], ax=ax2, bw_adjust=1.25)

    #plot vertical medians
    linestyle = '--'
    comparison_median = np.nanmedian(arr) ###
    ax2.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 4, label='_nolegend_', linestyle=linestyle) 
    
ax2.tick_params(reset=True, labelsize=32, length=12, width=2, right=False, top=False)
ax2.set_xlim([0,0.8])
ax2.set_ylabel('Probability density')
ax2.set_xlabel('sm$_{dry}$ $_{season}$ (cm$^3$/cm$^3$)')


#change in medians
print('change in median sm in fut vs ref'+str( np.nanmedian(fut_msm) - np.nanmedian(ref_msm)))


# ========================================= PLSM CDF
ax4 = fig.add_subplot(gs[1,1])
for j, arr in enumerate([ref_lpc, fut_lpc]):           

    comparison_median = np.nanmedian(arr) ###
    linestyle = '--'
            
    for i in range(3):
        #per model
        ax4.plot(np.sort(cl(arr[i])), np.linspace(0, 1, len(cl(arr[i])), endpoint=False), linewidth=3, alpha=0.5,
                 color=color_list[j])
    #median of models
    ax4.plot(np.sort(cl(arr)), np.linspace(0, 1, len(cl(arr)), endpoint=False), linewidth=4, color=color_list[j], label=labels[j])

    #vertical lines at median
    ax4.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 4, label='_nolegend_', linestyle=linestyle) 
        
#horizontal line at 0.5
ax4.axhline(y=0.5, color='black', linestyle='--', linewidth = 2,label='_nolegend_')

ax4.set_ylabel('Fraction of data')
ax4.set_xlim([-2,100])
ax4.set_ylim([0,1])
ax4.tick_params(reset=True, labelsize=32, length=12, width=2, right=False, top=False)
ax4.legend()
ax4.set_xlabel('pct$_{low}$ $_{sm}$ (% year)')






ax1.text(-.04,0.99,'a', transform = ax1.transAxes, weight='bold')
ax2.text(-.15,0.99,'b', transform = ax2.transAxes, weight='bold')
ax3.text(-.04,0.99,'c', transform = ax3.transAxes, weight='bold')
ax4.text(-.15,0.99,'d', transform = ax4.transAxes, weight='bold')
# plt.tight_layout()
plt.show()




#change in medians
print('change in median PLSM in fut vs ref '+str( np.nanmedian(fut_lpc) - np.nanmedian(ref_lpc)))
print('fut lpc: '+str(np.nanmedian(fut_lpc)))
print('ref lpc: '+str(np.nanmedian(ref_lpc)))













# =======================================================
#plot showing just change, no maps

fig = plt.figure(figsize=(30,20))
widths = [5,5]
heights = [1,1]
gs = fig.add_gridspec(ncols = 2, nrows = 2, width_ratios = widths, height_ratios = heights)
r'$E^{\alpha}_{\beta}$'
#msm change histogram
ax1 = fig.add_subplot(gs[1,0])
ax1.hist(msmdiff[np.isfinite(msmdiff)], bins=20)
ax1.set_xlabel(r"$\Delta SM^{climate}_{dry season} (cm^3/cm^3)$")
ax1.set_ylabel('count')
ax1.axvline(x=0, linewidth=2, color='black')

#lpc change histogram
ax3 = fig.add_subplot(gs[1,1])
ax3.hist(lpcdiff[np.isfinite(lpcdiff)], bins=20)
ax3.set_xlabel(r"$\Delta pct_{low sm}^{climate} (\% year)$")
ax3.set_ylabel('count')
ax3.axvline(x=0, linewidth=2, color='black')

#========================================== MSM PDF
color_list = ['cornflowerblue', 'sandybrown']
labels = ['1985-2005', '2040-2060']
#setup plot

ax2 = fig.add_subplot(gs[0,0])
#between reference and future
for j, arr in enumerate([ref_msm, fut_msm]):
    #individual RCM curves
    for i in range(3):
        sns.kdeplot(cl(arr[i]), color=color_list[j], linewidth=3, alpha=0.5, ax=ax2, bw_adjust=1.25)
    #plot pdf of all models
    sns.kdeplot(cl(arr), color=color_list[j], linewidth=4, label=labels[j], ax=ax2, bw_adjust=1.25)

    #plot vertical medians
    linestyle = '--'
    comparison_median = np.nanmedian(arr) ###
    ax2.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 4, label='_nolegend_', linestyle=linestyle) 
    
ax2.tick_params(reset=True, labelsize=32, length=12, width=2, right=False, top=False)
ax2.set_xlim([0,0.8])
ax2.set_ylabel('Probability density')
ax2.set_xlabel('sm$_{dry}$ $_{season}$ (cm$^3$/cm$^3$)')


#change in medians
print('change in median sm in fut vs ref'+str( np.nanmedian(fut_msm) - np.nanmedian(ref_msm)))


# ========================================= PLSM CDF
ax4 = fig.add_subplot(gs[0,1])
for j, arr in enumerate([ref_lpc, fut_lpc]):           

    comparison_median = np.nanmedian(arr) ###
    linestyle = '--'
            
    for i in range(3):
        #per model
        ax4.plot(np.sort(cl(arr[i])), np.linspace(0, 1, len(cl(arr[i])), endpoint=False), linewidth=3, alpha=0.5,
                 color=color_list[j])
    #median of models
    ax4.plot(np.sort(cl(arr)), np.linspace(0, 1, len(cl(arr)), endpoint=False), linewidth=4, color=color_list[j], label=labels[j])

    #vertical lines at median
    ax4.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 4, label='_nolegend_', linestyle=linestyle) 
        
#horizontal line at 0.5
ax4.axhline(y=0.5, color='black', linestyle='--', linewidth = 2,label='_nolegend_')

ax4.set_ylabel('Fraction of data')
ax4.set_xlim([-2,100])
ax4.set_ylim([0,1])
ax4.tick_params(reset=True, labelsize=32, length=12, width=2, right=False, top=False)
ax4.legend()
ax4.set_xlabel('pct$_{low}$ $_{sm}$ (% year)')


#a-d letters
ax3.text(-.15,0.99,'d', transform = ax3.transAxes, weight='bold')
ax4.text(-.15,0.99,'b', transform = ax4.transAxes, weight='bold')
ax1.text(-.15,0.99,'c', transform = ax1.transAxes, weight='bold')
ax2.text(-.15,0.99,'a', transform = ax2.transAxes, weight='bold')


#drier direction arrows
depth=-0.3
label='Drier'
ax1.arrow(x=0.9, y=depth, dx=-0.8, dy=0, width=0.09, transform=ax1.transAxes, facecolor='white', clip_on=False, length_includes_head=True, head_width=0.12, head_length=0.1, linewidth=5)
ax1.text(.5, depth, label, ha="center", va="center", size=32, weight='bold',c='black', transform = ax1.transAxes)

ax3.arrow(x=0.1, y=depth, dx=0.8, dy=0, width=0.09, transform=ax3.transAxes, facecolor='white', clip_on=False, length_includes_head=True, head_width=0.12, head_length=0.1, linewidth=5)
ax3.text(.5, depth, label, ha="center", va="center", size=32, weight='bold',c='black', transform = ax3.transAxes)


# plt.tight_layout()
plt.show()
