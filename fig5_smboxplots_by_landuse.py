
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 16})



# load data
features=['treecoverpct','lutpct1','lutpct2','lutpct3','lutpct4','lutpct5','dpe','cdens','annmeanprecip','drymeanprecip','drymeantemp','drymeanpet','pr_entropy','firepix1215','firecount1215','region','lats','lons']
features_notemp= [x for x in features if x != 'drymeantemp']
feature_sets = [features, features_notemp]#features_all, 
feature_set_names = ['ft2', 'ft_notemp2']#'ft_all',


#filename
home = 'data/model_outputs/'
ref_msm = np.load(home+'ref_00_meansm.npy')
ref_lpc = np.load(home+'ref_00_lowpct.npy')*100
fut_msm = np.load(home+'fut_00_meansm.npy')
fut_lpc = np.load(home+'fut_00_lowpct.npy')*100


# ===================

#remove negative values
fut_lpc[fut_lpc<0] = 0
ref_lpc[ref_lpc<0] = 0


def tempavg(arr): return np.nanmean(np.nanmean(arr, axis=0), axis=0)

#calculate change 
dmsm = tempavg(fut_msm) - tempavg(ref_msm)
dlpc = tempavg(fut_lpc) - tempavg(ref_lpc)


#figure out ex-MRP area
exMRP = np.full((198,276), 0)
exMRP[115:, 194:] = 1
#plt.figure(figsize=(15,15))
#plt.imshow(lutmap)
#plt.imshow(exMRP, alpha=0.25)
#plt.show()



# BY LAND USE TYPE

def tempavg(arr): return np.nanmean(arr,axis=(0,1))

lutmap = np.load('data/lut9km_2015.npy') #drainage_landuse_upscaling.py
luts = ['Pristine\nForest','Degraded\nForest','Open\nUndeveloped','Smallholder\nPlantation','Industrial\nPlantation']
#luts = ['PF','DF','OU','SP','IP']

#=========== mean sm
df = pd.DataFrame({'lut':lutmap.flatten(), '1985-2005':tempavg(ref_msm).flatten(), '2040-2060':tempavg(fut_msm).flatten(), 'exMRP':exMRP.flatten()})#, 'dmsm':dmsm.flatten()})
df['dmsm'] = df['2040-2060'] - df['1985-2005']

df.dropna(inplace=True)
df = df.loc[df.lut!=0]#drop other land uses

fig = plt.figure(constrained_layout=True, figsize=(15,8))
heights = [3,1]
gs = fig.add_gridspec(ncols = 2, nrows = 2, height_ratios = heights)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0], sharex=ax1)
ax4 = fig.add_subplot(gs[1,1], sharex=ax2)

#plot absolute
dd = pd.melt(df, id_vars=['lut'], value_vars = ['1985-2005','2040-2060'], var_name='Time period')
sns.boxplot(x='lut', y = 'value', data = dd, hue='Time period', ax=ax1, palette = sns.color_palette('pastel'), showfliers = False)#, showmeans=True)
ax1.set_ylim((0,0.7))
ax1.set_ylabel('sm$_{dry}$ $_{season}$\n (cm$^3$/cm$^3$)')
ax1.legend(loc='upper left', framealpha=1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_xlabel(None)

###### (difference in medians)
#ax3.bar(np.arange(5), df.groupby('lut').dmsm.median(), color='gray', width=0.3)#, alpha = 0.5) #bar
###### (difference in means)
ax3.bar(np.arange(5), df.groupby('lut')['2040-2060'].median()-df.groupby('lut')['1985-2005'].median(), color='gray', width=0.3)
ax3.set_ylabel('$\Delta$ sm$_{dry}$ $_{season}$\n (cm$^3$/cm$^3$)')
ax3.set_xlabel('Majority land use type')
ax3.set_xticklabels(luts, rotation='45')




#============= percent low soil moisture
df = pd.DataFrame({'lut':lutmap.flatten(), '1985-2005':tempavg(ref_lpc).flatten()*100, '2040-2060':tempavg(fut_lpc).flatten()*100, 'exMRP':exMRP.flatten()})#, 'dlpc':dlpc.flatten()*100
df['dlpc'] = df['2040-2060'] - df['1985-2005']

df.dropna(inplace=True)
df = df.loc[df.lut!=0]#drop other land uses



#plot absolute
dd = pd.melt(df, id_vars=['lut'], value_vars = ['1985-2005','2040-2060'], var_name='Time period')
sns.boxplot(x='lut', y = 'value', data = dd, hue='Time period', ax=ax2, palette = sns.color_palette('pastel'), showfliers = False)#, showmeans=True)
ax2.set_ylabel('pct$_{low}$ $_{sm}$\n (% year)')
ax2.legend(loc='upper right', framealpha=1)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_xlabel(None)

###### (difference in medians)
#ax4.bar(np.arange(5), df.groupby('lut').dlpc.mean(), color='gray', width=0.3)#, alpha=0.5) #bar
###### (difference in mean)
ax4.bar(np.arange(5), df.groupby('lut')['2040-2060'].median()-df.groupby('lut')['1985-2005'].median(), color='gray', width=0.3)
ax4.set_ylabel('$\Delta$ pct$_{low}$ $_{sm}$\n (% year)')
ax4.set_xlabel('Majority land use type')
ax4.set_xticklabels(luts, rotation='45')
ax4.axhline(y=-0.01, linestyle='--', color='gray', linewidth=1)


ax1.text(-.38,0.99,'a', transform = ax1.transAxes, weight='bold')
ax2.text(-.28,0.99,'b', transform = ax2.transAxes, weight='bold')
ax3.text(-.38,0.99,'c', transform = ax3.transAxes, weight='bold')
ax4.text(-.28,0.99,'d', transform = ax4.transAxes, weight='bold')


plt.subplots_adjust(hspace=0.1, wspace=0.4)
plt.show()

