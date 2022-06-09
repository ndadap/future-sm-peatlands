

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 16})



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

        

#%% histograms of climate changes

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



#initialize the matrices
RCMs = ['MP', 'NO', 'HA']
fut_ann_ent_pre_all = np.empty((0,198,276))
fut_ann_mean_pre_all = np.empty((0,198,276))
fut_ann_std_pre_all = np.empty((0,198,276))
fut_ann_std_tem_all = np.empty((0,198,276))
fut_ann_std_pet_all = np.empty((0,198,276))
fut_dry_mean_pre_all = np.empty((0,198,276))
fut_dry_mean_tem_all = np.empty((0,198,276))
fut_dry_mean_pet_all = np.empty((0,198,276))

futs_all = [fut_ann_ent_pre_all,fut_ann_mean_pre_all, fut_ann_std_pre_all, fut_ann_std_tem_all, fut_ann_std_pet_all, fut_dry_mean_pre_all, fut_dry_mean_tem_all,  fut_dry_mean_pet_all]

names = ['precip entropy', 'annual precip', 'std precip', 'std temp','std pet','dry season precip', 'dry season temp', 'dry season pet']

ref_ann_ent_pre_all = np.empty((0,198,276))
ref_ann_mean_pre_all = np.empty((0,198,276))
ref_ann_std_pre_all = np.empty((0,198,276))
ref_ann_std_tem_all = np.empty((0,198,276))
ref_ann_std_pet_all = np.empty((0,198,276))
ref_dry_mean_pre_all = np.empty((0,198,276))
ref_dry_mean_tem_all = np.empty((0,198,276))
ref_dry_mean_pet_all = np.empty((0,198,276))

refs_all = [ref_ann_ent_pre_all,ref_ann_mean_pre_all, ref_ann_std_pre_all, ref_ann_std_tem_all, ref_ann_std_pet_all, ref_dry_mean_pre_all, ref_dry_mean_tem_all,  ref_dry_mean_pet_all]

#iterating through each RCM
for RCM in RCMs:
    
    #load data
    futs = load_climate(RCM, 'fut')
    refs = load_climate(RCM, 'ref')

    #append for each variable
    for i in range(len(futs)):
        futs_all[i] = np.append(futs_all[i], futs[i], axis=0)
        refs_all[i] = np.append(refs_all[i], refs[i], axis=0)

#function to remove nans from each vector
def flatten(arrin):
    return arrin[np.isfinite(arrin)]




#%% CHANGE IN MEDIANS BY LAND USE TYPE

#PREPARE A DATAFRAME WITH PET, PRECIP, PERIOD, AND LAND USE
lut= np.load('data/lut9km_2015.npy')
reg= np.load('data/region.npy')

#future
lutarr = np.repeat(lut[np.newaxis,:,:], np.shape(futs_all[-1])[0], axis=0)
regarr = np.repeat(reg[np.newaxis,:,:], np.shape(futs_all[-1])[0], axis=0)
dffut = pd.DataFrame({'pet':futs_all[-1].flatten(), 'precip':futs_all[-3].flatten(), 'Period':'Future (2040-2060)', 'lut':lutarr.flatten()})
#reference
lutarr = np.repeat(lut[np.newaxis,:,:], np.shape(refs_all[-1])[0], axis=0)
regarr = np.repeat(reg[np.newaxis,:,:], np.shape(refs_all[-1])[0], axis=0)
dfref = pd.DataFrame({'pet':refs_all[-1].flatten(), 'precip':refs_all[-3].flatten(), 'Period':'Reference (1985-2005)', 'lut':lutarr.flatten()})
#remove nans and combine
dffut.dropna(inplace=True)
dfref.dropna(inplace=True)
df = pd.concat((dffut, dfref)).reset_index()
df = df.drop(columns='index')
#only lut 1-5
df=df.loc[df.lut!=0]
#convert name of land use
def label_lut(row):
  lutnames = ['PF', 'DF', 'OU', 'SP', 'IP'] #['Pristine Forest', 'Degraded Forest','Open Undeveloped','Smallholder Plantation','Industrial Plantations']
  if row['lut'] == 1:
      return lutnames[0]
  if row['lut'] == 2:
      return lutnames[1]
  if row['lut'] == 3:
      return lutnames[2]
  if row['lut'] == 4:
      return lutnames[3]
  if row['lut'] == 5:
      return lutnames[4]
df['Land Use'] = df.apply(lambda row: label_lut(row), axis=1)




#PLOTS
#calculate mean by LAND USE
grpmed = df.groupby(['Period','Land Use']).median()
grpmed.reset_index(inplace=True)
grpmed.sort_values('lut', inplace=True)

futmed = grpmed.loc[grpmed.Period == 'Future (2040-2060)'].reset_index()
refmed = grpmed.loc[grpmed.Period == 'Reference (1985-2005)'].reset_index()
dpet = (futmed['pet']-refmed['pet'])/refmed['pet']
dprecip = (futmed['precip']-refmed['precip'])/refmed['precip']








#%% BARPLOT: ABSOLUTE CHANGE VS LAND USE FOR SM AND CLIMATE VARIABLES


#dmsm
def av(arr): return np.nanmean(arr.reshape((60,198,276)), axis=0)
dmsm = (av(fut_msm) - av(ref_msm)) / av(ref_msm)
dlpc = (av(fut_lpc) - av(ref_lpc)) / av(ref_lpc)

dsm = pd.DataFrame({'dmsm':dmsm.flatten(), 'dlpc':dlpc.flatten(), 'lut':lut.flatten()})
dsm = dsm.dropna().reset_index().drop(columns='index')

#add landuse
dsm['Land Use'] = dsm.apply(lambda row: label_lut(row), axis=1)
#count
print(dsm.groupby(['lut'])['lut'].count())

#calculate median
dsm_med = dsm.groupby('Land Use').median().reset_index()

#reorder to PF, DF, OU, SP, IP, then add dpet and dprecip
dsm_med = dsm_med.reindex([3,0,2,4,1]).reset_index().drop(columns='index')
dsm_med['dpet'] = dpet
dsm_med['dprecip'] = dprecip    

# #reshape and plot
# #Mean SM
# toplot = dsm_med.melt(id_vars=[('Land Use')], value_vars=['dmsm','dpet','dprecip'])
# toplot['value'] = abs(toplot['value'])*100
# plt.figure(figsize=(6,4))
# ax = sns.barplot(x="Land Use", y="value", hue="variable", data=toplot)
# ax.set_ylabel('abs(% change)')
# h, l = ax.get_legend_handles_labels()
# ax.legend(h, ['$\Delta$ sm$_{dry}$ $_{season}$', '$\Delta $ PET','$\Delta$ precip'], bbox_to_anchor=(1.05, 1))
# plt.show()

# #pcct low
# toplot = dsm_med.melt(id_vars=[('Land Use')], value_vars=['dlpc','dpet','dprecip'])
# toplot['value'] = abs(toplot['value'])*100
# plt.figure(figsize=(6,4))
# ax = sns.barplot(x="Land Use", y="value", hue="variable", data=toplot)
# ax.set_ylabel('abs(% change)')
# h, l = ax.get_legend_handles_labels()
# ax.legend(h, ['$\Delta$ pct$_{low}$ $_{sm}$', '$\Delta $ PET','$\Delta$ precip'], bbox_to_anchor=(1.05, 1))
# plt.show()


#combined figure
fig, ax = plt.subplots(1,1, figsize=(6,6))
pal = sns.diverging_palette(220, 20, as_cmap=True)
toplot = dsm_med.melt(id_vars=[('Land Use')], value_vars=['dmsm','dlpc','dpet','dprecip'])
toplot['value'] = abs(toplot['value'])*100
plt.figure(figsize=(6,4))
sns.barplot(x="value", y="Land Use", hue="variable", data=toplot, palette=['red','orange','black','gray'], ax=ax)
ax.set_xlabel('% change')
ax.set_yticklabels(['Pristine\nForest','Degraded\nForest', 'Open\nUndeveloped', 'Smallholder\nPlantation', 'Industrial\nPlantation'])
h, l = ax.get_legend_handles_labels()
ax.legend(h, ['- $\Delta$ sm$_{dry}$ $_{season}$','$\Delta$ pct$_{low}$ $_{sm}$', '$\Delta $ PET','- $\Delta$ precip'], bbox_to_anchor=(1.05, 1))
plt.show()




import scipy.stats
scipy.stats.pearsonr(dsm_med.dmsm, dsm_med.dpet)
scipy.stats.spearmanr(dsm_med.dmsm, dsm_med.dprecip)
scipy.stats.spearmanr(dsm_med.dmsm, dsm_med.dpet)
