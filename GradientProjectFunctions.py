# Functions for gradient project

# analysis functions
import numpy as np
import xarray as xr
from math import exp, pi, sin, sqrt, log, radians, isnan
from scipy.stats import linregress
import cftime
import pandas as pd
import nc_time_axis
import random

def ClassifyModels(modelList):
    
    '''
    Takes in a list of openDAP urls and checks the length of time that they cover. If they cover the full period, the URLS are saved to a "Full" sub dictionary. If they do not cover the full 
    time they are saved to a sub-dictionary titled with the name of the dataset.
    
    Inputs:
        modelList: list of openDAP urls, separated by commas
        
        
    Outputs:
        modelsDict: dictionary with the classified URLs per the above description
        
        
    FUTURE UPDATE:
        Potentially include check here to ensure that the data is on a monthly basis and potentially also has the same resolution as others.
    '''
    
    # initialise a dictionary to hold the urls
    modelsDict = defaultdict(list)
    
    nModels = len(modelList)
    start_year = 1850
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2014
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well
    
    for url in modelList:

        ds = xr.open_dataset(url)
        
        # run two versions of checking depending on the format that the date time information is in
        if isinstance(ds.time.values[0], np.datetime64):
                start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
                end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

        elif isinstance(ds.time.values[0], cftime.DatetimeNoLeap):
            start_date = cftime.DatetimeNoLeap(start_year, monStart, dayStart)
            end_date = cftime.DatetimeNoLeap(end_year, monEnd, dayEnd)

        # save the full models to Full
        if (ds.time[0] <= start_date) & (ds.time[-1] >= end_date):
            modelsDict['Full'].append(url)
        
        # save the incomplete models to a different dictionary titled with their name
        else:
            modelName = '_'.join((ds.attrs['parent_source_id'], ds.attrs['variant_label']))
            modelsDict[modelName].append(url)    
    
    modelsDict = dict(modelsDict)
    return modelsDict
    
# defining a function to concatenate shorter time series

def ConcatModels(modelDict):
    
    '''
    For the models that weren't complete from the classify models stage (i.e., didn't cover the full period), concatenate all of the models saved
    
    Inputs:
        modelDict: a dictionary pertaining to a specific model (i.e., from the output of ClassifyModels, this should be modelsDict['Name of model'])
        
    Outputs:
        a full xarray dataset that has been concatenated to the full length
    
    '''
    # open one of the datasets
    ds = xr.open_dataset(modelDict[0])
    
    if len(modelDict) > 1:
    
        for i in range(1, len(modelDict)):
            ds2 = xr.open_dataset(modelDict[i])
            ds = xr.concat([ds, ds2], dim = 'time')
            
    else:
        
        ds = ds
    
    # sorting the data by time
    ds = ds.sortby('time')
    
    # run a check to make sure that the dataset encompasses the full period
    start_year = 1850
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2014
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well
    
    # run two versions of checking depending on the format that the date time information is in
    if isinstance(ds.time.values[0], np.datetime64):
            start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
            end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

    elif isinstance(ds.time.values[0], cftime.DatetimeNoLeap):
        start_date = cftime.DatetimeNoLeap(start_year, monStart, dayStart)
        end_date = cftime.DatetimeNoLeap(end_year, monEnd, dayEnd)

    # return the full model
    if (ds.time[0] <= start_date) & (ds.time[-1] >= end_date):
        return ds

    # run an error message if the datasets are incomplete
    else:
        raise ValueError('Concatenated models do not span full period')
    
    return ds
    
    
def ExtendPeriod(key, modelInput, scenarioModels):
    '''
    NOTE: we are working with MIROC model for the moment, where models are seeded from specific parents; not always the case
    Function that takes in the modelInput output and scenarioModels and combines to make one ds that has the full period from the start of the historical model to the end of the scenario model.
    
    Inputs:
        modelInput: an instance of the ModelInput class
        scenarioModels: dictionary of the scenario models labelled with their source_ids
    '''
    # the way that this runs depends on whether there's a scenario that matches the historical run in terms of parent
    
    if key in list(scenarioModels.keys()):

        # execute the code for the situation in which we can directly concatenate the arrays
        dsScenario = ModelInput(scenarioModels[key][0]).ds
        modelFullPeriod = xr.concat([modelInput.ds, dsScenario], dim = 'time')
        print(f'Non-random - Hist: {key} and Scenario: {scenarioModels[key][0]}')

    else:

        # execute the code for the situation in which you have to randomise the assigment
        modelHistID = modelInput.ds.attrs['source_id']

        # now randomly select one of the models from the same source_id
        # create a list of source_IDs (as in model names) so that we can choose an index from that list and randomise
        scenarioModelSource = []

        for i in list(scenarioModels.keys()):
            index = i.rfind('_')
            modelSource = i[:index]
            scenarioModelSource.append(modelSource)

        # create a mask for those model sources that match
        histMask = [modelID == modelHistID for modelID in scenarioModelSource]

        # create a list of integers to be the indices
        indices = list(range(len(list(scenarioModels.keys()))))

        # filter for only the indices that have True in the mask
        indicesMatch = [index for index, flag in zip(indices, histMask) if flag]

        # select a random index for the source
        scenarioRandom = list(scenarioModels)[random.choice(indicesMatch)]
        dsScenario = ModelInput(scenarioModels[scenarioRandom][0]).ds
        modelFullPeriod = xr.concat([modelInput.ds, dsScenario], dim = 'time')
        print(f'Random - Hist: {modelHistID} and Scenario: {scenarioModels[scenarioRandom][0]}')
    
    return modelFullPeriod
    
    
def CreateScenarioDictionary(modelListScenario):
    '''
    Creates a dictionary of scenarios that are the right length of time for this study. The keys are the source_ids of the models.
    
    Inputs:
        modelListScenario: the filtered list of URLs from the scenario models that we are interested in
        
    Outputs:
        a dictionary where the keys are the scenarioIDs for the models and the values are the URLs of models that are the right length for this study; Note that scenarioIDs are a combination of the model and
        parent variant (i.e., the historical model that seeded the model)
    '''
    
    # initialise a defaultdict to store the URLs
    scenarioModels = defaultdict(list)

    # dates for checking that the scenario fits into the right time
    start_year = 2015
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2022
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well

    for model in modelListScenario:
        # check that the scenario actually spans the time that we need before saving it
        ds = xr.open_dataset(model)
        sourceID = ds.attrs['source_id']
        parentVariant = ds.attrs['parent_variant_label']
        scenarioID = sourceID + '_' + parentVariant

        # run two versions of checking depending on the format that the date time information is in
        if isinstance(ds.time.values[0], np.datetime64):
                start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
                end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

        elif isinstance(ds.time.values[0], cftime.DatetimeNoLeap):
            start_date = cftime.DatetimeNoLeap(start_year, monStart, dayStart)
            end_date = cftime.DatetimeNoLeap(end_year, monEnd, dayEnd)

        # save the URL
        if (ds.time[0] <= start_date) & (ds.time[-1] >= end_date):
            # append the value to the list using the source_id as the key
            scenarioModels[scenarioID].append(model)

    # save the default dict as a dict
    scenarioModels = dict(scenarioModels)
    
    return scenarioModels
    
    
def RemoveClimatology(modelFullPeriod):
    '''
    Function that removes the climatology from the ts variable in the dataset
    
    Input: modelFullPeriod in this case is the full length concatenated ds that comes from combing hist and scen data
    
    Output: returns a ds with the climatology removed
    '''

    gb = modelFullPeriod.groupby('time.month')
    dsAnom = gb - gb.mean(dim = 'time')
    dsAnom.attrs = modelFullPeriod.attrs.copy()
    return dsAnom

def CropTrendsDf(trendsDf, offset):
    '''
    Function that crops the trendsDf at an offset from the diagonal so that we don't plot the dead ends of the axes
    
    Inputs: 
        trendsDf: a dataframe of trends from the Trend class
        offset: offset in years (i.e., the minimum trend length)
    '''
    keepRows = [i for i in range(trendsDf.shape[0]) if i < trendsDf.shape[0]-offset]
    keepCols =[j for j in range(trendsDf.shape[1]) if j < trendsDf.shape[1]-offset]

    trendsDfCrop = trendsDf.iloc[keepRows, keepCols]
    
    return trendsDfCrop

def lat_lon_res_Eq(ds):
    '''
    Small function that calculates the mean resolution and std of that resolution between lats -5 to 5.
    
    Inputs:
        Model dataset that has latitude and longitude dimensions labelled as such
        
    Ouputs:
        Prints the resolution mean and std
    '''
    
    
    maskEq = (ds.lat.values > -5) & (ds.lat.values < 5)
    latEq = ds.lat.values[maskEq]

    meanlatEq = np.mean(np.diff(latEq))
    stdlatEq = np.std(np.diff(latEq))

    meanLon = np.mean(np.diff(ds.lon.values))
    stdLon = np.std(np.diff(ds.lon.values))

    print('Latitude: Mean: %.2f and SD: %.3f' %(meanlatEq, stdlatEq))
    print('Longitude: Mean: %.2f and SD: %.3f' %(meanLon, stdLon))
    

def CalculateConcatJump(gradientsDir):
    '''
    Function that calculates the jump in the gradient between the end of the historical period and the start of the scenario period normalised 
    by the standard deviation of the historical
    
    Inputs:
        gradientsDir: directory containing gradient time series
        
    Outputs:
        A list of these values (in units of std)
    '''
    # run a for loop that concatenates all of the input files along the same new dimension ('gradient') then takes the mean along that dimension
    os.chdir(gradientsDir)

    # get a list of all of the files in the directory (getting rid of the python checkpoints one)
    gradientFiles = os.listdir(gradientsDir)
    gradientFiles = [f for f in gradientFiles if '.nc' in f]

    # now iterate through the list and concatenate them
    # adding in another line to check for jump at the concatenation point relative to standard deviation of the historical period

    # first initialise an xarray file for concatenation
    file1 = xr.open_dataset(gradientFiles[0])

    # have to do the first one manually because of the structure of the for loop
    concatJump = []

    histStart = '1850-01-16T12:00:00.000000000'
    histEnd = '2014-12-16T12:00:00.000000000'
    scenStart = '2015-01-16T12:00:00.000000000'
    stdHist = file1.sel(time = slice(histStart, histEnd)).std(dim = 'time').ts.item()
    jump = (file1.sel(time = histEnd) - file1.sel(time = scenStart)).ts.item()
    concatJump.append(jump/stdHist)

    for index, file in enumerate(gradientFiles):
        if index > 0:        
            # calculating the jump relative to std
            modelGradient = xr.open_dataset(file)
            stdHist = modelGradient.sel(time = slice(histStart, histEnd)).std(dim = 'time').ts.item()
            jump = (modelGradient.sel(time = histEnd) - modelGradient.sel(time = scenStart)).ts.item()
            concatJump.append(jump/stdHist)
            
    return concatJump
