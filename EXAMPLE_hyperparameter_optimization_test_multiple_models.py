

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt
from kerastuner.engine.hyperparameters import HyperParameters

from scipy import io
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import os
import shutil
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
plt.rcParams.update({'font.size': 16})




# %% MODEL FUNCTIONS

# Independent-dependent split function
def Xy(dfin, independent_list, dependent, seed=0):

    #get subset without nans
    all_columns = np.append(np.array(independent_list),dependent).tolist()
    subset = dfin[all_columns]
    subset.dropna(inplace=True)

    #setup array
    cols = []
    for col in independent_list:
        cols.append(np.array(subset[col]))
    features = np.array(cols).T 
    dependent_array = np.array(subset[dependent])
    
    #shuffle randomly
    stack = np.c_[features, dependent_array]
    np.random.seed(seed)
    np.random.shuffle(stack)
    
    #get X and y
    X = stack[:,:-1]
    y = stack[:,-1]
    
    return X, y

#cross validation split
def cvsplit(X, y, nfolds):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = nfolds)
    
    X_tr = []
    X_te = []
    y_tr = []
    y_te = []
    
    for train_index, test_index in kf.split(X):
        X_tr.append(X[train_index])
        X_te.append(X[test_index])
        y_tr.append(y[train_index])
        y_te.append(y[test_index])
        
    return X_tr, y_tr, X_te, y_te #return lists of X,y, train-test splits. use same index for list




def calccorr(x1, x2):
    mask = np.isfinite(x1) & np.isfinite(x2)
    tempcorr = np.corrcoef(x1[mask], x2[mask])[1,0]
    return tempcorr


# NN model
def build_model(hp, features):
    
    #hyperparameters
    num_layers = hp['num_layers']
    num_neurons = hp['num_neurons']
    learning_rate = hp['learning_rate']
    lr_schedule = hp['lr_schedule']
    dropout_fraction = hp['dropout_hidden']
    
    #learning rate (only if lr_schedule=True)
    lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay( #hyperbolic decrease of learning rate
        learning_rate,
        decay_steps=10000,
        decay_rate=10,
        staircase=False)
    
    #initialize model
    model = keras.Sequential()
    
    #first hidden layer
    model.add(layers.Dense(units = num_neurons,
                           input_shape=(len(features),), #input_shape looking at non-passed-in argument
                           activation='relu')) 
    
    # hidden layers
    for layer_number in range(num_layers):
        model.add(layers.Dense(num_neurons, activation='relu'))
        model.add(layers.Dropout(dropout_fraction))
        
    # output layer
    model.add(layers.Dense(1))
    
    #compile
    if lr_schedule:
        model.compile(optimizer = keras.optimizers.Adam(lr_decay), loss='mse')
    else:
        model.compile(optimizer = keras.optimizers.Adam(learning_rate), loss='mse')
    
    return model

#setup model with specified hyperparameters
def model_for_tuning(hp):
    
    #hyperparameters
    num_layers = hp.get('num_layers')
    num_neurons = hp.get('num_neurons')
    learning_rate = hp.get('learning_rate')
    lr_schedule = hp.get('lr_schedule')
    dropout_fraction = hp.get('dropout_hidden')
    
    #learning rate (only if lr_schedule=True)
    lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay( #hyperbolic decrease of learning rate
        learning_rate,
        decay_steps=10000,
        decay_rate=10,
        staircase=False)
    
    #initialize model
    model = keras.Sequential()
    
    #first hidden layer
    model.add(layers.Dense(units = num_neurons,
                           input_shape=(len(feature_set),), #input_shape looking at non-passed-in argument
                           activation='relu')) 
    
    # hidden layers
    for layer_number in range(num_layers):
        model.add(layers.Dense(num_neurons, activation='relu'))
        model.add(layers.Dropout(dropout_fraction))
        
    # output layer
    model.add(layers.Dense(1))
    
    #compile
    if lr_schedule:
        model.compile(optimizer = keras.optimizers.Adam(lr_decay), loss='mse')
    else:
        model.compile(optimizer = keras.optimizers.Adam(learning_rate), loss='mse')
    
    return model

#hyperparameter finding function
def get_nn_hp(train_features, train_labels, test_features, test_labels, ntrials=50):
    
    #initialize hyperparameter options
    hp = HyperParameters()
    hp.Choice('learning_rate', [1e-3]) ## always 1e-3 in previous tests
    hp.Choice('lr_schedule', [False])
    hp.Int('num_layers', 2,20)
    hp.Int('num_neurons', min_value = 25, max_value = 75, step = 10)
    hp.Float('dropout_hidden', min_value = 0, max_value=0.5, step = 0.1)
    
    #necessary to shorten filename otherwise throws error
    import os
    os.chdir('C:')
    
    #initialize tuner
    tuner = kt.tuners.RandomSearch(
            model_for_tuning,
            objective =  kt.Objective('val_loss', direction='min'),
            hyperparameters=hp,
            max_trials=ntrials, #100 searches in parameter space
#            directory='test_dir',
            overwrite=True,
            )
    
    #perform the search
    tuner.search(x=train_features,
                 y=train_labels,
                 epochs=60, # of epochs for each hp search
                 validation_data=(test_features, test_labels),
                 verbose=2)
    
    #get the best hyper parameters and model
    best_hp1 = tuner.get_best_hyperparameters()[0]
    
    return best_hp1
    


# %% LOAD DATA AND FILTER IF NEEDED
            
dfall = pd.read_pickle('G:/My Drive/peat_project/soildrying/quick_data2/trainingdata2.pkl') #see temporal_datasets.py


# RESCALE DATA
#take out years
yearSeries = dfall.year #take out year
dfall = dfall.drop('year', axis='columns')

#take out lat and lon indices
latidxSeries = dfall.latidx
lonidxSeries = dfall.lonidx

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

#dfall = (dfall-dfall.min())/(dfall.max()-dfall.min())
for col in dfall.columns:
    dfall[col] = rescale(col, dfall[col])

#put year and lat/lon indices back
dfall['year'] = yearSeries
dfall['latidx'] = latidxSeries
dfall['lonidx'] = lonidxSeries

## CHECK DATA (NOTE SOME WILL BE BLANK DUE TO NOT LOADING DATA)
#for col in dfall.columns:
#    print(col)
#    plt.hist(dfall[col].dropna(), density=True)
#    plt.xlabel(col)
#    plt.show()




# %% SETUP FEATURES

## test different sets
#features_all=['treelossyr','treecoverpct','lut2015','lut2007','lut1990','dpe','cdens','annmeanprecip','drymeanprecip','drymeantemp','drymeanpet','annstdprecip','annstdtemp','annstdpet','pr_entropy','firepix1215','firecount1215','region','lats','lons']
#
#features_crosscorr=['treelossyr','treecoverpct','dpe','cdens','drymeanprecip','drymeantemp','drymeanpet','annstdprecip','firecount1215','lats','lons']
#
#features_intuitive=['treecoverpct','lut2015','dpe','cdens','annmeanprecip','drymeanprecip','drymeantemp','drymeanpet','pr_entropy','firepix1215','firecount1215','region','lats','lons']
#
#features_sparse=['treecoverpct','dpe','cdens','annmeanprecip','drymeanprecip','drymeantemp','drymeanpet','firecount1215','lats','lons']
#
#features_notemp=['treelossyr','treecoverpct','lut2015','lut2007','lut1990','dpe','cdens','annmeanprecip','drymeanprecip','drymeanpet','annstdprecip','annstdpet','pr_entropy','firepix1215','firecount1215','region','lats','lons']


features=['treecoverpct','lutpct1','lutpct2','lutpct3','lutpct4','lutpct5','dpe','cdens','annmeanprecip','drymeanprecip','drymeantemp','drymeanpet','pr_entropy','firepix1215','firecount1215','region','lats','lons']
features_notemp= [x for x in features if x != 'drymeantemp']


# %% GET OPTIMAL HYPERPARAMETERS
# THEN ASSESS PERFORMANCE USING ANNUAL CROSS VALIDATION

#out:
# 1) best hyperparameters (dictionary) per feature set and depvar
# 2) plots of training history per year, feature_set, and depvar
# 3) plots of test data performance (distribution & scatterplot) per year of CV, feature_set, and depvar
# 4) R2 values per feature_set, dep, and year


#set feature sets
feature_sets = [features, features_notemp]#features_all, 
feature_set_names = ['ft2', 'ft_notemp2']#'ft_all',

#set dependent variable and 
dep_list = ['drymeansm','lowsmpct']
exMRPchoices = [True, False]

#initialize holders for R2s
R2_NNmeansm_temporalcv_df = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])
R2_NNlowpct_temporalcv_df = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])


R2_NNmeansm_randomcv_df = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])
R2_NNlowpct_randomcv_df = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])

#iterate over each combination of features
for i, feature_set in enumerate(feature_sets):

    #iterate over each dependent variable
    for depnum, dep in enumerate(dep_list):
        
        
        ### CALCULATE BEST HYPERPARAMETERS
        
        # GET all data for a given variable
        sublist = feature_set + [dep]
        dfsub = dfall[sublist].dropna().reset_index().drop(columns=['index'])
        # SETUP TRAIN/TEST
        train_dataset = dfsub.sample(frac=0.8, random_state=0)
        test_dataset = dfsub.drop(train_dataset.index)
        # SETUP FEATURES/LABELS
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        train_labels = train_features.pop(dep) #returns item and SIMULTANEOUSLY drops from frame!
        test_labels = test_features.pop(dep)
        
        #get hyperparameters
        tuned_hyperparams = get_nn_hp(train_features, train_labels, test_features, test_labels)
        
        #save best hyperparameters
        modfolder = 'G:/My Drive/peat_project/soildrying/quick_data2/models/'
        filename = 'besthp_'+dep+'_'+feature_set_names[i]+'v2'
        with open(modfolder + filename, 'wb') as handle:
            pickle.dump(tuned_hyperparams.values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        ### CALCULATE R2 USING TEMPORAL CROSS-VALIDATION ###
        
        #iterate over each year
        for year in range(6):
    
            # GET DATA for a specific year
            #other years (Train)
            dfnot = dfall.loc[dfall.year != (year+2015)] #first for all other years
            #subset then remove nans
            dfnot = dfnot[sublist]
            dfnot.dropna(inplace=True)
            X_train, y_train = Xy(dfnot, feature_set, dep, seed=0)
            #that year (Test)
            dfyear = dfall.loc[dfall.year == (year+2015)]
            dfyear = dfyear[sublist]
            dfyear.dropna(inplace=True)
            X_test, y_test = Xy(dfyear, feature_set, dep, seed=0)
            
            # TRAIN MODELS
            #NN
            nn = build_model(tuned_hyperparams.values, feature_set)
            history = nn.fit(X_train, y_train, epochs=250)#, callbacks = [EarlyStopping(patience=10)]) ## early stopping
            
#            #plot training history
#            plt.plot(history.history['loss'], label='loss')
#            plt.ylim([0, max(history.history['loss'])*1.2])
#            plt.xlabel('Epoch')
#            plt.ylabel('Cost fn')
#            plt.legend()
#            plt.savefig('G:/My Drive/peat_project/soildrying/plots2/training_history/{}{}_{}'.format(dep,str(year+2015), feature_set_names[i]))
#            plt.show()
        
            # CALCULATE TRAIN R2
            NNtrainr2_temporal = round(r2_score(y_train, nn.predict(X_train)), 2)
            
            # CALCULATE TEST R2
            NNtestr2_temporal = round(r2_score(y_test, nn.predict(X_test)), 2)
            
            #STORE TRAIN AND TEST R2
            row = {'trainR2':NNtrainr2_temporal, 'testR2':NNtestr2_temporal, 'year':str(year+2015),'featureset':feature_set_names[i]}
            if dep=='drymeansm':
                R2_NNmeansm_temporalcv_df = R2_NNmeansm_temporalcv_df.append(row, ignore_index=True)
            elif dep=='lowsmpct':    
                R2_NNlowpct_temporalcv_df = R2_NNlowpct_temporalcv_df.append(row, ignore_index=True)
            
            
            #PLOT MEASURES OF ACCURACY
            #setup obs and preds
            NNobs = y_test
            NNpreds = nn.predict(X_test)
            
            #get range
            NN_ran = np.max([np.nanmax(NNobs), np.nanmax(NNpreds)])
    
            #lists of info
            obs = NNobs
            preds = NNpreds.reshape(np.shape(NNpreds)[0])
            rans = NN_ran
            
            # scatterplot observed vs predicted test data 
            fig,ax = plt.subplots(figsize=(5,5))
            ax.scatter(preds, obs, s=0.5)
            ax.plot([0,1], [0,1], color='gray', linestyle='--')
            ax.set_xlim([0,rans])
            ax.set_ylim([0,rans])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Observed')
            ax.text(0.05, 0.9, 'Test R$^2$={}'.format(NNtestr2_temporal), transform = ax.transAxes)
            ax.text(0.05, 0.8, 'Train R$^2$={}'.format(NNtrainr2_temporal), transform = ax.transAxes)
            for h, key in enumerate(tuned_hyperparams.values):
                ax.text(0.8, 0.6-0.1*h, tuned_hyperparams.values[key], transform = ax.transAxes)
            fig.suptitle('Var={}, Year={}'.format(dep, year+2015))
            plt.savefig('G:/My Drive/peat_project/soildrying/plots2/scatterplots/{}{}_{}v2'.format(dep,str(year+2015), feature_set_names[i]))
            plt.show()
            
            #plot test obs vs predicted pdf
            fig, ax = plt.subplots(figsize=(5,5))
            sns.kdeplot(obs, label='obs')
            sns.kdeplot(preds, label='preds')
            ax.set_xlabel('dependent var')
            ax.set_ylabel('density')
            ax.set_title('Var={}, Year={}'.format(dep, year+2015))
            plt.legend()
            for h, key in enumerate(tuned_hyperparams.values):
                ax.text(0.8, 0.6-0.1*h, tuned_hyperparams.values[key], transform = ax.transAxes)
            plt.savefig('G:/My Drive/peat_project/soildrying/plots2/pdfs/{}{}_{}v2'.format(dep,str(year+2015), feature_set_names[i]))
            plt.show()
            
            
            
        ### TRAIN USING RANDOM CROSS VALIDATION ###
                        
        #get subset of clean data for a given variable
        sublist = feature_set + [dep]
        dfsub = dfall[sublist].dropna().reset_index().drop(columns=['index'])
        train_features = dfsub.copy()
        train_labels = train_features.pop(dep)  
        
        #break up into 4 fold cross validation
        kf = KFold(n_splits=5, random_state=7, shuffle=True)
        
        #initialize holder
        besttestr2=-9999
        
        #iterate over each fold
        for train_index, test_index in kf.split(train_features):

            X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
            y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

            #setup model
            nn = build_model(tuned_hyperparams.values, feature_set)
            checkpoint_filepath = 'G:/My Drive/peat_project/soildrying/quick_data2/models/checkpoints/checkpoint.h5'
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                       save_weights_only=True,
                                                                       monitor='loss',
                                                                       mode='min',
                                                                       save_best_only=True)
            
            #train
            history = nn.fit(X_train, y_train, epochs=300, callbacks=[model_checkpoint_callback])
            # Note: Model weights are saved at the end of every epoch, if it's the best seen so far
            # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
            
            #load best weights into model
            nn.load_weights(checkpoint_filepath)
 
            
            ##evaluate r2 scores
            NNtrainr2_random = round(r2_score(y_train, nn.predict(X_train)), 2)
            NNtestr2_random = round(r2_score(y_test, nn.predict(X_test)), 2)
            
            #STORE TRAIN AND TEST R2
            row = {'trainR2':NNtrainr2_random, 'testR2':NNtestr2_random, 'year':str(year+2015),'featureset':feature_set_names[i]}
            if dep=='drymeansm':
                R2_NNmeansm_randomcv_df = R2_NNmeansm_randomcv_df.append(row, ignore_index=True)
            elif dep=='lowsmpct':    
                R2_NNlowpct_randomcv_df = R2_NNlowpct_randomcv_df.append(row, ignore_index=True)
                
            #SAVE MODEL AND TRAINING HISTORY IF BETTER
            if NNtestr2_random > besttestr2:
            
                #set model name
                modelfolder = 'G:/My Drive/peat_project/soildrying/quick_data2/models/checkpoints/'
                modelname = dep+'_model_'+feature_set_names[i]+'v2'
                histname = dep+'_history_'+feature_set_names[i]+'v2'

                    
                #SAVE MODEL
                print(modelname)
                try:
                    os.mkdir(modelfolder+modelname)
                except FileExistsError:
                    shutil.rmtree(modelfolder+modelname)
                    os.mkdir(modelfolder+modelname)
                nn.save(modelfolder + modelname)
        
                #save training history
                histfolder ='G:/My Drive/peat_project/soildrying/plots2/training_history/'
                plt.plot(history.history['loss'], label='loss')
                plt.ylim([0, max(history.history['loss'])*1.2])
                plt.xlabel('Epoch')
                plt.ylabel('Cost fn')
                plt.legend()
                plt.title(modelname)
                plt.savefig(histfolder+histname)
                plt.show()

            
            
            
#save R2 values
R2_NNmeansm_temporalcv_df.to_pickle('G:/My Drive/peat_project/soildrying/plots2/meansmR2_temporalcv_comparisonv2.pkl')
R2_NNlowpct_temporalcv_df.to_pickle('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_temporalcv_comparisonv2.pkl')
R2_NNmeansm_randomcv_df.to_pickle('G:/My Drive/peat_project/soildrying/plots2/meansmR2_randomcv_comparisonv2.pkl')
R2_NNlowpct_randomcv_df.to_pickle('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_randomcv_comparisonv2.pkl')

R2_NNmeansm_temporalcv_df.groupby('featureset').mean()
R2_NNlowpct_temporalcv_df.groupby('featureset').mean()
R2_NNmeansm_randomcv_df.groupby('featureset').mean()
R2_NNlowpct_randomcv_df.groupby('featureset').mean()

## =========== VIEW R2s
##read info
#def openpkl(path):
#    objects = []
#    with (open(path, "rb")) as openfile:
#        while True:
#            try:
#                objects.append(pickle.load(openfile))
#            except EOFError:
#                break
#    
#    return objects
#
#def savepkl(obj, path):
#    with open(path, 'wb') as handle:
#        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#meansmR2 = openpkl('G:/My Drive/peat_project/soildrying/plots2/meansmR2_comparison.pkl')[0]
#lowpctR2 = openpkl('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_comparison.pkl')[0]
#
#
#print(meansmR2.groupby('featureset').mean())
#print(lowpctR2.groupby('featureset').mean())


















#%%
#run again with no exMRP
latbound = 120
lonbound = 206
dfall.loc[(dfall.latidx > latbound) & (dfall.lonidx > lonbound)] = np.nan



#initialize holders for R2s
R2_NNmeansm_temporalcv_df_noMRP = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])
R2_NNlowpct_temporalcv_df_noMRP = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])


R2_NNmeansm_randomcv_df_noMRP = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])
R2_NNlowpct_randomcv_df_noMRP = pd.DataFrame(columns=['trainR2','testR2','year','featureset'])

#iterate over each combination of features
for i, feature_set in enumerate(feature_sets):

    #iterate over each dependent variable
    for depnum, dep in enumerate(dep_list):
        
        
        ### CALCULATE BEST HYPERPARAMETERS
        
        # GET all data for a given variable
        sublist = feature_set + [dep]
        dfsub = dfall[sublist].dropna().reset_index().drop(columns=['index'])
        # SETUP TRAIN/TEST
        train_dataset = dfsub.sample(frac=0.8, random_state=0)
        test_dataset = dfsub.drop(train_dataset.index)
        # SETUP FEATURES/LABELS
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()
        train_labels = train_features.pop(dep) #returns item and SIMULTANEOUSLY drops from frame!
        test_labels = test_features.pop(dep)
        
        #get hyperparameters
        tuned_hyperparams = get_nn_hp(train_features, train_labels, test_features, test_labels)
        
        #save best hyperparameters
        modfolder = 'G:/My Drive/peat_project/soildrying/quick_data2/models/'
        filename = 'besthp_'+dep+'_'+feature_set_names[i]+'_noMRPv2'
        with open(modfolder + filename, 'wb') as handle:
            pickle.dump(tuned_hyperparams.values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        ### CALCULATE R2 USING TEMPORAL CROSS-VALIDATION ###
        
        #iterate over each year
        for year in range(6):
    
            # GET DATA for a specific year
            #other years (Train)
            dfnot = dfall.loc[dfall.year != (year+2015)] #first for all other years
            #subset then remove nans
            dfnot = dfnot[sublist]
            dfnot.dropna(inplace=True)
            X_train, y_train = Xy(dfnot, feature_set, dep, seed=0)
            #that year (Test)
            dfyear = dfall.loc[dfall.year == (year+2015)]
            dfyear = dfyear[sublist]
            dfyear.dropna(inplace=True)
            X_test, y_test = Xy(dfyear, feature_set, dep, seed=0)
            
            # TRAIN MODELS
            #NN
            nn = build_model(tuned_hyperparams.values, feature_set)
            history = nn.fit(X_train, y_train, epochs=250)#, callbacks = [EarlyStopping(patience=10)]) ## early stopping
            
#            #plot training history
#            plt.plot(history.history['loss'], label='loss')
#            plt.ylim([0, max(history.history['loss'])*1.2])
#            plt.xlabel('Epoch')
#            plt.ylabel('Cost fn')
#            plt.legend()
#            plt.savefig('G:/My Drive/peat_project/soildrying/plots2/training_history/{}{}_{}'.format(dep,str(year+2015), feature_set_names[i]))
#            plt.show()
        
            # CALCULATE TRAIN R2
            NNtrainr2_temporal = round(r2_score(y_train, nn.predict(X_train)), 2)
            
            # CALCULATE TEST R2
            NNtestr2_temporal = round(r2_score(y_test, nn.predict(X_test)), 2)
            
            #STORE TRAIN AND TEST R2
            row = {'trainR2':NNtrainr2_temporal, 'testR2':NNtestr2_temporal, 'year':str(year+2015),'featureset':feature_set_names[i]}
            if dep=='drymeansm':
                R2_NNmeansm_temporalcv_df_noMRP = R2_NNmeansm_temporalcv_df_noMRP.append(row, ignore_index=True)
            elif dep=='lowsmpct':    
                R2_NNlowpct_temporalcv_df_noMRP = R2_NNlowpct_temporalcv_df_noMRP.append(row, ignore_index=True)
            
            
            #PLOT MEASURES OF ACCURACY
            #setup obs and preds
            NNobs = y_test
            NNpreds = nn.predict(X_test)
            
            #get range
            NN_ran = np.max([np.nanmax(NNobs), np.nanmax(NNpreds)])
    
            #lists of info
            obs = NNobs
            preds = NNpreds.reshape(np.shape(NNpreds)[0])
            rans = NN_ran
            
            # scatterplot observed vs predicted test data 
            fig,ax = plt.subplots(figsize=(5,5))
            ax.scatter(preds, obs, s=0.5)
            ax.plot([0,1], [0,1], color='gray', linestyle='--')
            ax.set_xlim([0,rans])
            ax.set_ylim([0,rans])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Observed')
            ax.text(0.05, 0.9, 'Test R$^2$={}'.format(NNtestr2_temporal), transform = ax.transAxes)
            ax.text(0.05, 0.8, 'Train R$^2$={}'.format(NNtrainr2_temporal), transform = ax.transAxes)
            for h, key in enumerate(tuned_hyperparams.values):
                ax.text(0.8, 0.6-0.1*h, tuned_hyperparams.values[key], transform = ax.transAxes)
            fig.suptitle('Var={}, Year={}'.format(dep, year+2015))
            plt.savefig('G:/My Drive/peat_project/soildrying/plots2/scatterplots/{}{}_{}_noMRPv2'.format(dep,str(year+2015), feature_set_names[i]))
            plt.show()
            
            #plot test obs vs predicted pdf
            fig, ax = plt.subplots(figsize=(5,5))
            sns.kdeplot(obs, label='obs')
            sns.kdeplot(preds, label='preds')
            ax.set_xlabel('dependent var')
            ax.set_ylabel('density')
            ax.set_title('Var={}, Year={}'.format(dep, year+2015))
            plt.legend()
            for h, key in enumerate(tuned_hyperparams.values):
                ax.text(0.8, 0.6-0.1*h, tuned_hyperparams.values[key], transform = ax.transAxes)
            plt.savefig('G:/My Drive/peat_project/soildrying/plots2/pdfs/{}{}_{}_noMRPv2'.format(dep,str(year+2015), feature_set_names[i]))
            plt.show()
            
            
            
        ### TRAIN USING RANDOM CROSS VALIDATION ###
                        
        #get subset of clean data for a given variable
        sublist = feature_set + [dep]
        dfsub = dfall[sublist].dropna().reset_index().drop(columns=['index'])
        train_features = dfsub.copy()
        train_labels = train_features.pop(dep)  
        
        #break up into 4 fold cross validation
        kf = KFold(n_splits=5, random_state=7, shuffle=True)
        
        #initialize holder
        besttestr2=-9999
        
        #iterate over each fold
        for train_index, test_index in kf.split(train_features):

            X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
            y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

            #setup model
            nn = build_model(tuned_hyperparams.values, feature_set)
            checkpoint_filepath = 'G:/My Drive/peat_project/soildrying/quick_data2/models/checkpoints/checkpoint.h5'
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                       save_weights_only=True,
                                                                       monitor='loss',
                                                                       mode='min',
                                                                       save_best_only=True)
            
            #train
            history = nn.fit(X_train, y_train, epochs=300, callbacks=[model_checkpoint_callback])
            # Note: Model weights are saved at the end of every epoch, if it's the best seen so far
            # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
            
            #load best weights into model
            nn.load_weights(checkpoint_filepath)
 
            
            ##evaluate r2 scores
            NNtrainr2_random = round(r2_score(y_train, nn.predict(X_train)), 2)
            NNtestr2_random = round(r2_score(y_test, nn.predict(X_test)), 2)
            
            #STORE TRAIN AND TEST R2
            row = {'trainR2':NNtrainr2_random, 'testR2':NNtestr2_random, 'year':str(year+2015),'featureset':feature_set_names[i]}
            if dep=='drymeansm':
                R2_NNmeansm_randomcv_df_noMRP = R2_NNmeansm_randomcv_df_noMRP.append(row, ignore_index=True)
            elif dep=='lowsmpct':    
                R2_NNlowpct_randomcv_df_noMRP = R2_NNlowpct_randomcv_df_noMRP.append(row, ignore_index=True)
                
            #SAVE MODEL AND TRAINING HISTORY IF BETTER
            if NNtestr2_random > besttestr2:
            
                #set model name
                modelfolder = 'G:/My Drive/peat_project/soildrying/quick_data2/models/checkpoints/'
                modelname = dep+'_model_'+feature_set_names[i]+'_noMRPv2'
                histname = dep+'_history_'+feature_set_names[i]+'_noMRPv2'

                    
                #SAVE MODEL
                print(modelname)
                try:
                    os.mkdir(modelfolder+modelname)
                except FileExistsError:
                    shutil.rmtree(modelfolder+modelname)
                    os.mkdir(modelfolder+modelname)
                nn.save(modelfolder + modelname)
        
                #save training history
                histfolder ='G:/My Drive/peat_project/soildrying/plots2/training_history/'
                plt.plot(history.history['loss'], label='loss')
                plt.ylim([0, max(history.history['loss'])*1.2])
                plt.xlabel('Epoch')
                plt.ylabel('Cost fn')
                plt.legend()
                plt.title(modelname)
                plt.savefig(histfolder+histname)
                plt.show()

            
            
            
#save R2 values
R2_NNmeansm_temporalcv_df_noMRP.to_pickle('G:/My Drive/peat_project/soildrying/plots2/meansmR2_temporalcv_comparison_noMRPv2.pkl')
R2_NNlowpct_temporalcv_df_noMRP.to_pickle('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_temporalcv_comparison_noMRPv2.pkl')
R2_NNmeansm_randomcv_df_noMRP.to_pickle('G:/My Drive/peat_project/soildrying/plots2/meansmR2_randomcv_comparison_noMRPv2.pkl')
R2_NNlowpct_randomcv_df_noMRP.to_pickle('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_randomcv_comparison_noMRPv2.pkl')

R2_NNmeansm_temporalcv_df_noMRP.groupby('featureset').mean()
R2_NNlowpct_temporalcv_df_noMRP.groupby('featureset').mean()
R2_NNmeansm_randomcv_df_noMRP.groupby('featureset').mean()
R2_NNlowpct_randomcv_df_noMRP.groupby('featureset').mean()








#%%
#view CV results (including exMRP)
r2msmtem = pd.read_pickle('G:/My Drive/peat_project/soildrying/plots2/meansmR2_temporalcv_comparisonv2.pkl')
r2lpctem = pd.read_pickle('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_temporalcv_comparisonv2.pkl')
#random
r2msmran = pd.read_pickle('G:/My Drive/peat_project/soildrying/plots2/meansmR2_randomcv_comparisonv2.pkl')
r2lpcran = pd.read_pickle('G:/My Drive/peat_project/soildrying/plots2/lowpctR2_randomcv_comparisonv2.pkl')


#train r2s
r2msmtem.trainR2.mean()
r2lpctem.trainR2.mean()
r2msmran.trainR2.mean()
r2lpcran.trainR2.mean()

#test r2s
r2msmtem.testR2.mean()
r2lpctem.testR2.mean()
r2msmran.testR2.mean()
r2lpcran.testR2.mean()

#std dev
print(r2msmtem.std())
print(r2lpctem.std())
print(r2msmran.std())
print(r2lpcran.std())





#%% print hyperparameters
with open('G:/My Drive/peat_project/soildrying/quick_data2/models/besthp_drymeansm_ft2v2','rb') as handle:
    msm_hp = pickle.load(handle)
print(msm_hp)

with open('G:/My Drive/peat_project/soildrying/quick_data2/models/besthp_lowsmpct_ft2v2','rb') as handle:
    lpc_hp = pickle.load(handle)
print(lpc_hp)


