



import numpy as np
import pandas as pd
import sklearn
from sklearn import ensemble
import matplotlib.pyplot as plt
from copy import copy
import pickle
from sklearn.metrics import mean_squared_error
import seaborn as sns
import tensorflow as tf
import sys
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors
from scipy import io
from scipy import stats
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl


pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 20})




#%% functions


#shorten non-nan function
def cl(arr): return arr[np.isfinite(arr)]

#flatten
def f(arr): return arr.flatten()

#adjusted index finder to find closest index from below
def idx(array, value):
    idx = (np.abs(array-value)).argmin()
    
    if array[idx]>value:
        return idx-1
    else:
        return idx
    
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


#rescale function (hardcoded)
def rescale(var, val, inv=False):
    
    #give flexibility to handle different datasets
    if '_ERA5' in var:
        var = var[:-5]
    if '_GLDAS' in var:
        var = var[:-6]
    if '_CHIRPS' in var:
        var = var[:-7]
        
    #hardcoded max and min values
    maxvals = {'lut2015':5,'lut2007':5, 'lut1990':5,
               'dpe':483, 'cdens':0.033, 'treelossyr':20, 'treecoverpct':1,
               'firepix1215': 86, 'firecount1215':3232,
               'lats':7.2, 'lons':120.5,'latidx':198, 'lonidx':276, 'region': 4,
               'annmeanprecip':15,
               'drymeanprecip':12.5, 'drymeantemp':303, 'drymeanpet':6.5,
               'annstdprecip':10, 'annstdtemp':2, 'annstdpet':2,
               'pr_entropy':2.5,# 'temp_entropy':2.484907, 'pet_entropy':2.483646,
               
               'drymeansm':0.8, 'lowsmpct':1,
               
               'lutpct1':1,'lutpct2':1,'lutpct3':1,'lutpct4':1,'lutpct5':1,
               }
    
    minvals = {'lut2015':0,'lut2007':0, 'lut1990':0,
               'dpe':1.5, 'cdens':0, 'treelossyr':3, 'treecoverpct':0,
               'firepix1215': 0, 'firecount1215':0,
               'lats':-7.2, 'lons':94,'latidx':0, 'lonidx':0, 'region': 1,
               'annmeanprecip':2.9, 
               'drymeanprecip':0.35, 'drymeantemp':297, 'drymeanpet':0,
               'annstdprecip':0, 'annstdtemp':0, 'annstdpet':0,
               'pr_entropy':2,# 'temp_entropy': 2.484900e+00, 'pet_entropy':6.931472e-01,
               
               'drymeansm':0, 'lowsmpct':0,
               
               'lutpct1':0,'lutpct2':0,'lutpct3':0,'lutpct4':0,'lutpct5':0,
               }
    if inv==False:
        try:
            return (val - minvals[var]) / (maxvals[var] - minvals[var])
        except KeyError:
            return np.nan
    elif inv==True:
        try:
            return val*(maxvals[var] - minvals[var]) + minvals[var]
        except KeyError:
            return np.nan


# plotting function to compare histograms - emphasize median
def hist_comp(array_list, label_list, color_list, xlabel, cdf=False, title=None, gray_first=True):
    
    models = ['MP', 'NO', 'HA']
    
    plt.figure(figsize=(10,6))

    if cdf==False: #PDF
        #iterating over prediction set
        for j, arr in enumerate(array_list):
            #iterating over each model
            for i, mod in enumerate(models):
                sns.kdeplot(cl(arr[i]), color=color_list[j], linewidth=1, alpha=0.5)
            #plot median of models
            sns.kdeplot(cl(arr), color=color_list[j], linewidth=3, label=label_list[j])
            
            
            #set linestyle for first array
            if (j==0) and (gray_first==True):
                linestyle='solid'
            else:
                linestyle = '--'
            plt.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 2, label='_nolegend_', linestyle=linestyle) 
            
        plt.ylabel('Probability density')
        plt.xlim(left=0)
        
    if cdf==True: #CDF
        for j, arr in enumerate(array_list):
            
            #set linestyle for first array
            if (j==0) and (gray_first==True):
                linestyle='solid'
            else:
                linestyle = '--'
            
            for i, mod in enumerate(models):
                #per model
                plt.plot(np.sort(cl(arr[i])), np.linspace(0, 1, len(cl(arr[i])), endpoint=False), linewidth=1, alpha=0.5,
                         color=color_list[j])
            #median of models
            plt.plot(np.sort(cl(arr)), np.linspace(0, 1, len(cl(arr)), endpoint=False), linewidth=3, color=color_list[j], label=label_list[j])
        
            #vertical lines at median
            plt.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 2, label='_nolegend_', linestyle=linestyle) 
        
        #horizontal line at 0.5
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth = 1,label='_nolegend_')
        
        plt.ylabel('Fraction of data')
        plt.xlim(left=-0.02)
        plt.ylim([0,1])
        
    plt.legend()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


# plotting function to compare histograms - emphasize median
def hist_comp_subplot(array_list, label_list, color_list, xlabel, ax, cdf=False, title=None, gray_first=True):
    
    models = ['MP', 'NO', 'HA']

    if cdf==False: #PDF
        #iterating over prediction set
        for j, arr in enumerate(array_list):
            #iterating over each model
            for i, mod in enumerate(models):
                sns.kdeplot(cl(arr[i]), color=color_list[j], linewidth=1, alpha=0.5, ax=ax)
            #plot median of models
            sns.kdeplot(cl(arr), color=color_list[j], linewidth=3, label=label_list[j], ax=ax)
            
            
            #set linestyle for first array
            if (j==0) and (gray_first==True):
                linestyle='solid'
                baseline_median = np.nanmedian(arr) ###
            else:
                linestyle = '--'
                comparison_median = np.nanmedian(arr) ###
            ax.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 2, label='_nolegend_', linestyle=linestyle) 
            
            
            
        ax.set_ylabel('Probability density')
        ax.set_xlim(left=0)
        
    if cdf==True: #CDF
        for j, arr in enumerate(array_list):
            
            #set linestyle for first array
            if (j==0) and (gray_first==True):
                linestyle='solid'
                baseline_median = np.nanmedian(arr) ###
            else:
                comparison_median = np.nanmedian(arr) ###
                linestyle = '--'
            
            for i, mod in enumerate(models):
                #per model
                ax.plot(np.sort(cl(arr[i])), np.linspace(0, 1, len(cl(arr[i])), endpoint=False), linewidth=1, alpha=0.5,
                         color=color_list[j])
            #median of models
            ax.plot(np.sort(cl(arr)), np.linspace(0, 1, len(cl(arr)), endpoint=False), linewidth=3, color=color_list[j], label=label_list[j])
        
            #vertical lines at median
            ax.axvline(x=np.nanmedian(arr),color=color_list[j], linewidth = 2, label='_nolegend_', linestyle=linestyle) 
        
        #horizontal line at 0.5
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth = 1,label='_nolegend_')
        
        ax.set_ylabel('Fraction of data')
        ax.set_xlim(left=-0.02)
        ax.set_ylim([0,1])
        
    ax.legend()
    ax.set_xlabel(xlabel)
    
    #calculate shift 
    medianshift = comparison_median-baseline_median ###
    
    ax.set_title(title+', shift='+str(round(medianshift,2))) ###


#split data by region
def getregions(arrin):
    
    ### array in must be in dimensions: [model, years, lat, lon] ###
    
    #define region divides
    NS_divide = 100 #indices splitting north-south (equator), east-west (~107.7)
    EW_divide = 141
    
    #initialize arrays out per region
    arroutNW = np.full((3, np.shape(arrin)[1], 100, 141), np.nan)
    arroutNE = np.full((3, np.shape(arrin)[1], 100, 135), np.nan)
    arroutSW = np.full((3, np.shape(arrin)[1], 98, 141), np.nan)
    arroutSE = np.full((3, np.shape(arrin)[1], 98, 135), np.nan)
    
    #iterating over each model
    for model in range(3):
        
        #get out data as array
        modarr = arrin[model]

        #get 4 arrays representing each region
        arroutNW[model,:,:,:] = modarr[:, :NS_divide, :EW_divide]
        arroutNE[model,:,:,:] = modarr[:, :NS_divide, EW_divide:]
        arroutSW[model,:,:,:] = modarr[:, NS_divide:, :EW_divide]
        arroutSE[model,:,:,:] = modarr[:, NS_divide:, EW_divide:]
        
    return [arroutNW, arroutNE, arroutSW, arroutSE]

#get data for one region
def getregion(arrin, reg):
       
    #define region divides
    NS_divide = 100 #indices splitting north-south (equator), east-west (~107.7)
    EW_divide = 141
    
    if reg == 'NW':
        return arrin[..., :NS_divide, :EW_divide]
    elif reg == 'NE':
        return arrin[..., :NS_divide, EW_divide:]
    elif reg == 'SW':
        return arrin[..., NS_divide:, :EW_divide]
    elif reg == 'SE':
        return arrin[..., NS_divide:, EW_divide:]
    else:
        return None
    
#plot regional histograms
def plot_reg_histograms(array_list, label_list, color_list, xlabel, cdf, title, filename=None):
    
    #setup plot
    fig,axs=plt.subplots(5,1, figsize=(7,25))
    
    #plot for full area
    hist_comp_subplot(array_list=array_list, label_list=label_list, color_list=color_list, xlabel=xlabel, ax=axs[0], cdf=cdf, title='All pixels')
    
    #regions
    region_names = ['NW','NE','SW','SE']
    
    #initialize arrays
    array_list_NW = []
    array_list_NE = []
    array_list_SW = []
    array_list_SE = []
    
    #iterating over each array in
    for arrin in array_list:
        
        #split into regions
        listofnewarrays = getregions(arrin)
        
        #save into new array list
        array_list_NW.append(listofnewarrays[0])
        array_list_NE.append(listofnewarrays[1])
        array_list_SW.append(listofnewarrays[2])
        array_list_SE.append(listofnewarrays[3])

    #iterating over each region
    for i, reg_array_list in enumerate([array_list_NW, array_list_NE, array_list_SW, array_list_SE]):

        #plot
        hist_comp_subplot(array_list=reg_array_list, label_list=label_list, color_list=color_list, xlabel=xlabel, ax=axs[int(i+1)], cdf=cdf, title=region_names[i])
        
    plt.tight_layout()
    plt.suptitle(title)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


#define function to shift by a certain degradation
def shift(valsin, pctdeg, tcdegonly):
    
    if tcdegonly == False:
        
        tcin, cdin = valsin
        
        # regression parameters
        slope = -0.03415253332304479#-0.013523728743216315
        
    #    intercept = 0.01409418618706534
        
        # change in tree cover (positive degradation ~ negative shift in treecover)
        dtc = (-pctdeg/100)
        
        #change in canal density = ynew - ymid (note midpoint and intercept terms cancel out)
        dcd =  slope*(dtc)
        
        #calculate changes
        tcout = tcin + dtc
        cdout = cdin + dcd
        
        #enforce boundary conditions
        if tcout < 0:
            tcout = 0
        if tcout > 1:
            tcout = 1
        if cdout < 0:
            cdout = 0
        
        return tcout, cdout
    
    if tcdegonly == True:
        
        tcin = valsin
        
        # change in tree cover (positive degradation ~ negative shift in treecover)
        dtc = (-pctdeg/100)
        
        #calculate changes
        tcout = tcin + dtc
        
        #enforce boundary conditions
        if tcout < 0:
            tcout = 0
        if tcout > 1:
            tcout = 1

        return tcout


#function to get mean/mode for other features
def retrieve_shifted_ft(valsin, df, feature, returntype, tcdegonly):
    
    if tcdegonly == False:
        
        tcval, cdval = valsin
        
        #arrays to define grid lines
        gridtc = np.linspace(0, 1, 11) #space is divided by 10
        gridcd = np.linspace(0, 0.0328*1.2, 11) ### 0.0328 == np.nanmax(cdens)
        tcsize = (gridtc[1] - gridtc[0])/2
        cdsize = (gridcd[1] - gridcd[0])/2
                
        #get high and low values
        tclo = tcval-tcsize
        tchi = tcval+tcsize
        cdlo = cdval-cdsize
        cdhi = cdval+cdsize
    
        #get subset; while subset length < 5, keep expanding the box
        subset_length=0
        while subset_length<5:
               
            #get subset for grid cell
            condition = (df.treecoverpct>tclo) & (df.treecoverpct<=tchi) & (df.cdens>cdlo) & (df.cdens<=cdhi)
            subset = df.loc[condition]
        
            #check subset length
            subset_length = len(subset)
            
            #stop early if sufficiently long
            if subset_length >= 5:
                break
            
            else:
                tclo -= tcsize
                tchi += tcsize
                cdlo -= cdsize
                cdhi += cdsize
        
        #take desired value
        if returntype == 'mean':
            valout = subset[feature].mean()
        elif returntype == 'median':
            valout = subset[feature].median()
        elif returntype == 'mode':
            valout = subset[feature].mode()[0]
        else:
            print('need to select returntype of mean, median, or mode')
            return None
    
    
    if tcdegonly == True:
        
        tcval = valsin
        
        gridtc = np.linspace(0, 1, 11)
        tcsize = (gridtc[1] - gridtc[0])/2
        tclo = tcval-tcsize
        tchi = tcval+tcsize
    
        #get subset; while subset length < 5, keep expanding the box
        subset_length=0
        while subset_length<5:
               
            #get subset for grid cell
            condition = (df.treecoverpct>tclo) & (df.treecoverpct<=tchi)
            subset = df.loc[condition]
        
            #check subset length
            subset_length = len(subset)
            
            #stop early if sufficiently long
            if subset_length >= 5:
                break
            
            else:
                tclo -= tcsize
                tchi += tcsize
                
        
        #take desired value
        if returntype == 'mean':
            valout = subset[feature].mean()
        elif returntype == 'median':
            valout = subset[feature].median()
        elif returntype == 'mode':
            valout = subset[feature].mode()[0]
        else:
            print('need to select returntype of mean, median, or mode')
            return None
        
    return valout

#function to load training data (True denotes including exMRP data)
def load_training_data(exMRP=True):
    dfall = pd.read_pickle('data/trainingdata.pkl')
    
    #filter for no exMRP
    if exMRP==False:
        latbound = 120
        lonbound = 206
        dfall.loc[(dfall.lats > latbound) & (dfall.lons > lonbound)] = np.nan
       
    # RESCALE DATA
    #take out years
    yearSeries = dfall.year #take out year
    dfall = dfall.drop('year', axis='columns')
    
    #take out lat and lon indices
    latidxSeries = dfall.latidx
    lonidxSeries = dfall.lonidx
    
    #rescale
    for col in dfall.columns:
        dfall[col] = rescale(col, dfall[col])
    
    #put year and lat/lon indices back
    dfall['year'] = yearSeries
    dfall['latidx'] = latidxSeries
    dfall['lonidx'] = lonidxSeries
    
    return dfall

#setup features for each dataset
def prepare_training_data(dep, df, features):
    sublist = features + [dep]
    dfsub = df[sublist].dropna().reset_index().drop(columns=['index'])
    train_features = dfsub.copy()
    train_labels = train_features.pop(dep) #returns item and SIMULTANEOUSLY drops from frame!
    return train_features, train_labels
    
#calculate and plot feature importance    
def feat_importance(model, train_features, train_labels, titlein, fname):

    #calculate predictions and store
    df = pd.DataFrame({'pred':np.reshape(model.predict(train_features), newshape=len(train_labels)), 'obs':train_labels})
    
    #calculate baseline performance (R2 or RMSE)
    baseline_mse = mean_squared_error(df.obs, df.pred)
    
    #set number of iterations
    n = 10
    
    #initialize dataframe to store r2/mse with permuted feature
    df_10itr_r2 = pd.DataFrame(index = range(n), columns = train_features.columns)
    df_10itr_mse = pd.DataFrame(index = range(n), columns = train_features.columns)
    
    #for each feature
    print('calculating feature importance for ' + titlein)
    for feat in train_features:
        
        sys.stdout.write("\rCurrent feature: %s" % feat)
        sys.stdout.flush()
        
        #initialize lists to save R2s
        r2 = []
        mse = []
        
        #vary seed a number of times
        for seed in np.arange(n):
            
            #initialize copy of feature dfs
            shuffled = copy(train_features)
            
            #shuffle one column randomly
            shuffled[feat] = np.random.RandomState(seed=seed).permutation(shuffled[feat].values)
    
            #calculate predictions and store
            df_oneitr = pd.DataFrame({'pred':np.reshape(model.predict(shuffled), newshape=len(train_labels)), 'obs':train_labels})
            
            #calculate performance
            r2_oneitr = r2_score(df_oneitr.obs, df_oneitr.pred)
            mse_oneitr = mean_squared_error(df_oneitr.obs, df_oneitr.pred)
            
            #save into list
            r2.append(r2_oneitr)
            mse.append(mse_oneitr)
            
        #calculate R2 and save
        df_10itr_r2[feat] = r2
        df_10itr_mse[feat] = mse
    
    #calculate averages from permutations
    perm_r2 = df_10itr_r2.mean()
    perm_mse = df_10itr_mse.mean()
    
    # ======== calculate feature importance
    FI = perm_mse/baseline_mse
    
    ####
    #higher error when including permuted feature -> more important (model can't do as well)
    #converse is true: lower error -> less important
    
    #similarly, higher R2 when using permuted feature -> less important feature
    # lower R2 when using permuted feature -> higher importance
    
    #normalize values
    FI = FI/FI.sum()
    
    #plot
    fig, ax = plt.subplots(figsize=(7,6))
    FI.sort_values().plot.barh(ax=ax)
    ax.set_xlabel('Relative Importance')
    ax.set_title(titlein)
    plt.savefig(fname)
    plt.show()
    
    

#function to load model
def load_model(dep, ftset_name, exMRPchoice):
    modelfolder = 'data/model_checkpoints/'
    #note prior folder was named: 'G:/My Drive/peat_project/soildrying/scripts/temporal_cv/working_models/'
    if exMRPchoice == True:
        modelname = dep+'_model_'+ftset_name
    elif exMRPchoice == False:
        modelname = dep+'_model_'+ftset_name +'_noMRP'
        
    print(modelname)
    model = tf.keras.models.load_model(modelfolder+modelname)
    return model


# PARTIAL DEPENDENCE PLOT
def str_dec_arr(arr):
    out = []
    for i in arr:
        out.append("{:.2f}".format(i))
    return out

def pdp_all(df, featureset, title, model, fname):
    
    #remove nans
    df=df[featureset].dropna()
    
    #setup plot
    # Compute Rows required
    Tot = len(featureset)
    Cols = 4
    Rows = Tot // Cols 
    Rows += Tot % Cols

    Position = range(1,Tot + 1)# Create a Position index
    fig = plt.figure(figsize=(20,20))

  # add every single subplot to the figure with a for loop

    for n, feature in enumerate(featureset):
    
        #copy
        df_copy = df.copy()
        #create test values
        test_vals = np.linspace(df_copy[feature].min(), df_copy[feature].max(), 20)#np.unique(df_copy[feature].values)
        y = []
        
        for val in test_vals:
            sys.stdout.write("\rCurrent feature: %s" % val)
            sys.stdout.flush()
            
            df_copy[feature] = val #set feature to one value
            X = df_copy[featureset]
            y.append(np.average(model.predict(X)))
            
        ax = fig.add_subplot(Rows,Cols,Position[n])
        
        #line plot
        xvals=rescale(feature, test_vals, inv=True)
        g = sns.lineplot(x=xvals, y=y, ax=ax)
        
        
        #ticks for samples
        unique_vals = np.unique(df_copy[feature].values)
        unique_vals_rescaled = rescale(feature, unique_vals, inv=True)
        for tick in unique_vals_rescaled:
            ax.axvline(x=tick, ymax=0.01, color='black')
            
        #xticks
        g.set(xticks=xvals[::4])
        g.set(xticklabels=str_dec_arr(xvals[::4]))
        g.set(xlabel = feature)
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


#function to setup disturbance features
def load_disturbance():
        
    #load tree cover / canal density disturbance    
    treecoverpct = np.load('data/disturbance/treecoverpct2015.npy') #save_metrics.py
    cdens = np.load('data/disturbance/canaldens9km.npy') #drainage_landuse_upscaling.py
    
    #load disturbance features
#    lut2015 = np.load('G:/My Drive/peat_project/soildrying/quick_data/maps/lut9km_2015.npy') #drainage_landuse_upscaling.py
#    lut2007 = np.load('G:/My Drive/peat_project/soildrying/quick_data/maps/lut9km_2007.npy') #drainage_landuse_upscaling.py
#    lut1990 = np.load('G:/My Drive/peat_project/soildrying/quick_data/maps/lut9km_1990.npy') #drainage_landuse_upscaling.py
    lutpct1 = np.load('data/disturbance/lutpct1.npy')
    lutpct2 = np.load('data/disturbance/lutpct2.npy')
    lutpct3 = np.load('data/disturbance/lutpct3.npy')
    lutpct4 = np.load('data/disturbance/lutpct4.npy')
    lutpct5 = np.load('data/disturbance/lutpct5.npy')
    
    firepix1215 = np.load('data/disturbance/firepixels12-15.npy') #save_metrics.py
    firecount1215 = np.load('data/disturbance/firecount12-15.npy') #save_metrics.py
    treelossyr = np.load('data/disturbance/hansenlossyear.npy') #save_metrics.py


    return treecoverpct, cdens, lutpct1, lutpct2, lutpct3, lutpct4, lutpct5, firepix1215, firecount1215, treelossyr
    

# function to load climate data
def load_climate(rcm, period):
    
    ## Setup climate variables
    if period == 'fut':
        root = 'data/climate/{}_{}85_{}.npy'
    elif period == 'ref':
        root = 'data/climate/{}_{}RF_{}.npy' 
    else:
        print('need to specify fut, ref, or past for period')
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
    
    return ann_ent_pre, ann_mean_pre, ann_std_pre, ann_std_tem, ann_std_pet, dry_mean_pre, dry_mean_tem, dry_mean_pet

# function to load location data
def load_location():
    region = np.load('data/region.npy') #saved outside of script
#    lats = np.load('G:/My Drive/peat_project/soildrying/quick_data/maps/lats.npy')
#    lons = np.load('G:/My Drive/peat_project/soildrying/quick_data/maps/lons.npy')
    latvec = np.load('data/ease2_9km_lats_ISEA.npy')
    lonvec = np.load('data/ease2_9km_lons_ISEA.npy')
    lons,lats = np.meshgrid(lonvec,latvec)
    dpe = np.load('data/dpe9km.npy') #drainage_landuse_upscaling.py
    return region, lats, lons, dpe


#function to run experiment  for a given time period and degree degradation
def run_exp(period,
            features,
            exMRP,
            meansm_model,# = 'G:/My Drive/peat_project/soildrying/scripts/temporal_cv/working_models/meansm',
            lowpct_model):# = 'G:/My Drive/peat_project/soildrying/scripts/temporal_cv/working_models/lowpct'):
    
    #load fitted models
#    nn_meansm = tf.keras.models.load_model(meansm_model)
#    nn_lowpct = tf.keras.models.load_model(lowpct_model)

    # LOAD LOCATION DATA (2d matrix)
    region, lats, lons, dpe = load_location()
    print('loaded location data, loading disturbance data...')
    
    # LOAD DISTURBANCE DATA (2d matrix)
    treecoverpct, cdens, lutpct1, lutpct2, lutpct3, lutpct4, lutpct5, firepix1215, firecount1215, treelossyr = load_disturbance()
    
    #specify number of years to predict
    if (period == 'past') or (period == 'ref'):
        num_years = 20
    elif period == 'fut':
        num_years = 20
    else:
        print('need to specify fut, ref, or past for period')
        
    #initialize arrays to hold predictions (indices: model, year, lat, lon)
    pred_meansm = np.full((3, num_years, 198,276), np.nan)
    pred_lowpct = np.full((3, num_years, 198,276), np.nan)
    
    
    #iterate over each model
    RCMs = ['MP', 'NO', 'HA'] #corresponding to low, medium, and high sensitivity
    for modelidx, RCM in enumerate(RCMs):
        
        print('processing model: '+RCM)
        
        # LOAD CLIMATE DATA (3d matrix specific to a certain RCM)
        ann_ent_pre, ann_mean_pre, ann_std_pre, ann_std_tem, ann_std_pet, dry_mean_pre, dry_mean_tem, dry_mean_pet = load_climate(rcm = RCM, period = period)

        #iterate over each year
        for year in range(num_years):
            
            sys.stdout.write("\rcurrent year out of "+str(num_years)+" years: %s" % year)
            sys.stdout.flush()
            
            #set up dataframe, with same order as in training
            feature_df = pd.DataFrame({
                               #disturbance
                               'treecoverpct':f(treecoverpct),
                               'lutpct1':f(lutpct1),
                               'lutpct2':f(lutpct2),
                               'lutpct3':f(lutpct3),
                               'lutpct4':f(lutpct4),
                               'lutpct5':f(lutpct5),
                               'dpe':f(dpe),
                               'cdens':f(cdens),
                               
                               #climate
                               'annmeanprecip':f(ann_mean_pre[year,:,:]),
                               'drymeanprecip':f(dry_mean_pre[year,:,:]),
                               'drymeantemp':f(dry_mean_tem[year,:,:]),
                               'drymeanpet':f(dry_mean_pet[year,:,:]),
                               'annstdprecip':f(ann_std_pre[year,:,:]),
                               'annstdtemp':f(ann_std_tem[year,:,:]),
                               'annstdpet':f(ann_std_pet[year,:,:]),
                               'pr_entropy':f(ann_ent_pre[year,:,:]),
    #                           'temp_entropy':f(ann_ent_tem[year,:,:]),
    #                           'pet_entropy':f(ann_ent_pet[year,:,:]),
                               
                               #disturbance
                               'firepix1215':f(firepix1215),
                               'firecount1215':f(firecount1215),
                               
                               #location
                               'region':f(region),
                               'lats':f(lats),
                               'lons':f(lons)
                               })
        
            #select subset of features
            feature_df = feature_df[features]
            
            #add lat and lonidx
            latidx = np.load('data/lat_idx.npy')
            lonidx = np.load('data/lon_idx.npy')
            
            feature_df['latidx']=f(latidx)
            feature_df['lonidx']=f(lonidx)
            
            #take out exMRP
            if exMRP==False:
                latbound = 120
                lonbound = 206
                feature_df.loc[(feature_df.lats > latbound) & (feature_df.lons > lonbound)] = np.nan
    
            
            #drop nans
            feature_df.dropna(inplace=True)
            
            #check if there is data for a given year, and if not continue to next year (fill with nans)
            if len(feature_df)==0:
                pred_meansm[modelidx, year,:,:] = np.full((198,276), np.nan)
                pred_lowpct[modelidx, year,:,:] = np.full((198,276), np.nan)
                continue
            
            #save out copy of lats and lons
            latSeries = copy(feature_df.latidx)
            lonSeries = copy(feature_df.lonidx)
            feature_df = feature_df.drop(columns=['latidx','lonidx'])
            
            #save feature dataframe
            if year == 0:
                fulldffut = copy(feature_df)
            else:
                fulldffut = fulldffut.append(feature_df)
            
            #rescale
            for col in feature_df.columns:
                feature_df[col] = rescale(col, feature_df[col])
            
            #create copy to save prediction
            base_meansm = copy(feature_df[[]])
            base_lowpct = copy(feature_df)
            
            #MAKE PREDICTION (for a given year)
            base_meansm['drymeansm'] = meansm_model.predict(feature_df)
            base_lowpct['lowsmpct'] = lowpct_model.predict(feature_df)
            
            #rescale back out of transformed space
            base_meansm['drymeansm'] = rescale('drymeansm', base_meansm['drymeansm'], inv=True)
            base_lowpct['lowsmpct'] = rescale('lowsmpct', base_lowpct['lowsmpct'], inv=True)
            
            #add lats and lons (not rescaled) back to dataframe
            base_meansm['latidx'] = latSeries
            base_meansm['lonidx'] = lonSeries
            base_lowpct['latidx'] = latSeries
            base_lowpct['lonidx'] = lonSeries
            
            #save to dataframe
            pred_meansm[modelidx, year,:,:] = tomap(base_meansm, 'drymeansm')
            pred_lowpct[modelidx, year,:,:] = tomap(base_lowpct, 'lowsmpct')
            
    return pred_meansm, pred_lowpct


# plot mean of response variables vs percent degradation
def degradation_response(tcdegonly, features, exMRP, meansm_model, lowpct_model, plottitle, fnameout):
    
    #iterating over degree of degradation
    changelist_meansm = [[], [], [], []]
    changelist_lowpct = [[], [], [], []]
    change_meansm_allreg = []
    change_lowpct_allreg = []
    regions = ['NW','NE','SW','SE']
    
    #vector of degradation
    pctdegvec = np.arange(15,50,15)
    
    #baseline
    ref_00_meansm, ref_00_lowpct = run_exp(period='ref', pctdeg=0, tcdegonly=tcdegonly, features=features, exMRP=exMRP, meansm_model=meansm_model, lowpct_model=lowpct_model)
    
    #iterate over each value of degradation
    for deg in pctdegvec:
        newmeansm, newlowpct = run_exp(period='ref', pctdeg=deg, tcdegonly=tcdegonly, features=features, exMRP=exMRP, meansm_model=meansm_model, lowpct_model=lowpct_model)
        
        #calculate for full area
        diff_meansm_all = np.nanmean(newmeansm) - np.nanmean(ref_00_meansm)
        diff_lowpct_all = np.nanmean(newlowpct) - np.nanmean(ref_00_lowpct)
        change_meansm_allreg.append(diff_meansm_all)
        change_lowpct_allreg.append(diff_lowpct_all)
        
        #iterating per region
        for i, region in enumerate(regions):
            
            #get mean value for reference
            diff_meansm = np.nanmean(getregion(newmeansm, region)) - np.nanmean(getregion(ref_00_meansm, region))
            diff_lowpct = np.nanmean(getregion(newlowpct, region)) - np.nanmean(getregion(ref_00_lowpct, region))
                
            #save to list of lists
            changelist_meansm[i].append(diff_meansm)
            changelist_lowpct[i].append(diff_lowpct)
    
    #plot overall
    fig, ax= plt.subplots(5,2, figsize=(15,25))
    #meansm
    ax[0,0].plot(pctdegvec, change_meansm_allreg)
#    ax[0,0].set_ylabel('$\Delta$Mean SM (deg - ref)')
#    ax[0,0].set_xlabel('Pct degradation')
    #lowpct
    ax[0,1].plot(pctdegvec, change_lowpct_allreg)
#    ax[0,1].set_ylabel('$\Delta$Pct low SM (deg - ref)')
#    ax[0,1].set_xlabel('Pct degradation')
    
    #plot per region
    for i, region in enumerate(regions):    
        row=int(i+1)
        
        #meansm
        ax[row,0].plot(pctdegvec, changelist_meansm[i])

        #lowpct
        ax[row,1].plot(pctdegvec, changelist_lowpct[i])
        
        if i==3: #if last row, add labels
            ax[row,0].set_ylabel('$\Delta$Mean SM (deg - ref)')
            ax[row,0].set_xlabel('Pct degradation')
            ax[row,1].set_ylabel('$\Delta$Pct low SM (deg - ref)')
            ax[row,1].set_xlabel('Pct degradation')
        
    fig.suptitle(plottitle)
    plt.tight_layout()
    plt.savefig(fnameout)
    plt.show()


#%%set features


features=['treecoverpct','lutpct1','lutpct2','lutpct3','lutpct4','lutpct5','dpe','cdens','annmeanprecip','drymeanprecip','drymeantemp','drymeanpet','pr_entropy','firepix1215','firecount1215','region','lats','lons']
features_notemp= [x for x in features if x != 'drymeantemp']




#%% SAVE DATA

feature_sets = [features, features_notemp]#features_all, 
feature_set_names = ['ft2', 'ft_notemp2']#'ft_all',
exMRPchoices = [True, False]

j=0
exMRPchoice = exMRPchoices[j]
k = 1
featureset = feature_sets[k]



meansm_mod = load_model('drymeansm', feature_set_names[k], True)
lowpct_mod = load_model('lowsmpct', feature_set_names[k], True)
        
#baseline
ref_00_meansm, ref_00_lowpct = run_exp(period='ref', features = featureset,
                                         exMRP = exMRPchoice, meansm_model = meansm_mod, lowpct_model = lowpct_mod)

#climate change only
fut_00_meansm, fut_00_lowpct = run_exp(period='fut', features = featureset,
                                         exMRP = exMRPchoice, meansm_model = meansm_mod, lowpct_model = lowpct_mod)




#%% save experiments

#save
meansmvars = [ref_00_meansm, fut_00_meansm]
lowpctvars = [ref_00_lowpct, fut_00_lowpct]

file_ends = ['ref_00_{}', 'fut_00_{}']

folder = 'data/example_outputs'

for i in range(len(meansmvars)):
    np.save(folder + file_ends[i].format('meansm'), meansmvars[i])
    np.save(folder + file_ends[i].format('lowpct'), lowpctvars[i])










