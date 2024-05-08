# Functions for gradient project

# analysis libraries
import numpy as np
import xarray as xr
from math import exp, pi, sin, sqrt, log, radians, isnan
from scipy.stats import linregress
import random
from scipy.stats import percentileofscore

# data handling libraries
import pandas as pd
import nc_time_axis
from collections import defaultdict
import cftime

# systems functions
import os

# my own functions
from GradTrendClasses import ModelInput, Gradient

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
    count = 0
    start_year = 1850
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2014
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well
    
    for url in modelList:
        # just to keep track while it's running
        count +=1

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
        
        print(f'Classification complete: {count} / {nModels}')
    
    modelsDict = dict(modelsDict)
    
    return modelsDict
    
    

def ClassifyHistModelsLite(urlList):
    '''
    Function that classifies historical models according to whether they span the full period or not
    
    Inputs:
        urlList: list of historical model urls (per the Haibo openDAP format)
        
    Outputs:
        a dictionary where the keys are either 'Full' or the modelID (i.e., modelName and variant label)
    '''
    
    # initialising a dictionary
    histModels = defaultdict(list)
    
    char = '_'
    char2 = '.'
    fullDates = '185001-201412'
    
    for url in urlList:
        # finding Amon
        indAmon = url.index(char)

        # find the modelName
        indModelNameStart = url.index(char, indAmon+1) + 1
        indModelNameEnd = url.index(char, indModelNameStart+1)

        modelName = url[indModelNameStart:indModelNameEnd]

        # find the run variant
        indVariantStart = url.index(char, indModelNameEnd+1) + 1
        indVariantEnd = url.index(char, indVariantStart+1)

        modelVariant = url[indVariantStart:indVariantEnd]

        # finding the date range
        indDateStart = url.index(char, indVariantEnd+2) + 1
        indDateEnd = url.index(char2, indDateStart)

        dateRange = url[indDateStart:indDateEnd]
        
        # now checking whether full or not
        if dateRange == fullDates:
            key = 'Full'
        else:
            key = modelName + '_' + modelVariant
            
        histModels[key].append(url)
        
    histModels = dict(histModels)
    return histModels


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
    nModels = len(modelListScenario)
    count = 0
    start_year = 2015
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2022
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well

    for model in modelListScenario:
        count +=1
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
        
        # keeping track of progress
        print(f'Scenario dictionary complete: {count} / {nModels}')

    # save the default dict as a dict
    scenarioModels = dict(scenarioModels)
    
    return scenarioModels

def CreateScenarioDictionaryLite(modelListScenario):
    '''
    Creates a dictionary of scenarios that are in theory the right length for this study (note that it doesn't explicitly check the range of the dates, but uses the input openDAP url conventions)
    
    Inputs:
        modelListScenario: the filtered list of URLs from the scenario models that we are interested in
        
    Outputs:
        a dictionary where the keys are the scenarioIDs for the models and the values are the URLs of models that are the right length for this study; Note that scenarioIDs are a combination of the model and
        source variant (i.e., the variant id of the model scenario itself not the parent)
    '''
    
    scenarioModels = defaultdict(list)

    char = '_'
    char2 = '.'
    endDate = 202401

    for url in modelListScenario:
        # finding Amon
        indAmon = url.index(char)

        # find the modelName
        indModelNameStart = url.index(char, indAmon+1) + 1
        indModelNameEnd = url.index(char, indModelNameStart+1)

        modelName = url[indModelNameStart:indModelNameEnd]

        # find the run variant
        indVariantStart = url.index(char, indModelNameEnd+1) + 1
        indVariantEnd = url.index(char, indVariantStart+1)

        modelVariant = url[indVariantStart:indVariantEnd]

        key = modelName + '_' + modelVariant

        # finding the date range
        indDateStart = url.index(char, indVariantEnd+2) + 1

        dateStart = int(url[indDateStart:indDateStart+6])

        # now checking whether full or not
        if dateStart <= endDate:
            scenarioModels[key].append(url)

    scenarioModels = dict(scenarioModels)
    
    return scenarioModels

def CreateScenarioDictionaryNonParent(modelListScenario):
    '''
    *** NOTE: This differs from the CreateScenarioDictionary function in that that function saves the resulting models under the parent ID (i.e., assuming that parent isn't necessarily the same as source ID where this one doesn't)
 
    Creates a dictionary of scenarios that are the right length of time for this study. The keys are the source_ids of the models.
    
    Inputs:
        modelListScenario: the filtered list of URLs from the scenario models that we are interested in
        
    Outputs:
        a dictionary where the keys are the scenarioIDs for the models and the values are the URLs of models that are the right length for this study; Note that scenarioIDs are a combination of the model and
        source variant (i.e., the variant id of the model scenario itself not the parent)
    '''
    
    # initialise a defaultdict to store the URLs
    scenarioModels = defaultdict(list)

    # dates for checking that the scenario fits into the right time
    nModels = len(modelListScenario)
    count = 0
    start_year = 2015
    monStart = 1
    dayStart = 31 # because these are sometimes on the 16th of the month
    end_year = 2022
    monEnd = 12
    dayEnd = 1  # because these are sometimes on the 16th of the month as well

    for model in modelListScenario:
        count +=1
        # check that the scenario actually spans the time that we need before saving it
        ds = xr.open_dataset(model)
        sourceID = ds.attrs['source_id']
        memberVariant = ds.attrs['variant_label']
        scenarioID = sourceID + '_' + memberVariant

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
        
        # keeping track of progress
        print(f'Scenario dictionary complete: {count} / {nModels}')

    # save the default dict as a dict
    scenarioModels = dict(scenarioModels)
    
    return scenarioModels

def ExtendPeriod(key, modelInput, scenarioModels):
    '''
    Function that takes in the modelInput output and scenarioModels and combines to make one ds that has the full period from the start of the historical model to the end of the scenario model.
    
    Inputs:
        modelInput: an instance of the ModelInput class
        scenarioModels: dictionary of the scenario models labelled with their source_ids
    
    Outputs:
        modelFullPeriod: a model that spans the full period of the historical and scenario
        match: a tuple containing an identifier for the scenario model that the historical model was concatenated with and the note of random versus non-random
            NOTE: this is the variant label and not the parent variant label
    '''
    # the way that this runs depends on whether there's a scenario that matches the historical run in terms of parent
    
    
    if key in list(scenarioModels.keys()):
            
        # execute the code for the situation in which we can directly concatenate the arrays
        dsScenario = ModelInput(scenarioModels[key][0]).ds
        modelFullPeriod = xr.concat([modelInput.ds, dsScenario], dim = 'time')
        match = (dsScenario.attrs['parent_source_id'] + '_' + dsScenario.attrs['variant_label'], 'Non-random')
        print(match)

    else:

        # execute the code for the situation in which you have to randomise the assigment
        modelHistID = modelInput.ds.attrs['source_id']
        runHist = modelInput.ds.attrs['variant_label']

        # now randomly select one of the models from the same source_id
        # create a list of source_IDs (as in model names) so that we can choose an index from that list and randomise
        
        # first have to flatten any of them that might have subdictionaries
        scenarioModelsFlat = {}

        for key, value in scenarioModels.items():
            if len(value) > 1:
                counter = 0
                for subValue in value:
                    subValueList = []
                    subValueList.append(subValue)
                    scenarioModelsFlat[key + '_' + str(counter)] = subValueList
                    counter += 1
            else:
                scenarioModelsFlat[key] = value
    
        scenarioModelSource = []

        for i in list(scenarioModelsFlat.keys()):
            index = i.index('_')
            modelSource = i[:index]
            scenarioModelSource.append(modelSource)

        # create a mask for those model sources that match
        histMask = [modelID == modelHistID for modelID in scenarioModelSource]

        # create a list of integers to be the indices
        indices = list(range(len(list(scenarioModelsFlat.keys()))))

        # filter for only the indices that have True in the mask
        indicesMatch = [index for index, flag in zip(indices, histMask) if flag]

        # select a random index for the source
        scenarioRandom = list(scenarioModelsFlat)[random.choice(indicesMatch)]
        dsScenario = ModelInput(scenarioModelsFlat[scenarioRandom][0]).ds
        modelFullPeriod = xr.concat([modelInput.ds, dsScenario], dim = 'time')
        match = (dsScenario.attrs['parent_source_id'] + '_' + dsScenario.attrs['variant_label'], 'Random')
        print(match)
    
    return modelFullPeriod, match
    
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
    by the standard deviation of the historical and the standard deviation of the Jan - Dec differences for the historical
    
    Inputs:
        gradientsDir: directory containing gradient time series
        
    Outputs:
        concatJumpFullHist: A list of values, in units of std of (concat difference - mean of historical period)/std of historical period
        concatJumpJanDec: A list of values, in units of std of (concat difference - mean of historical difference between Jan and Dec) / std of historial Jan / Dec differences
        
    '''
    # want to add in a function to also have it be compared to Dec / Jan jumps for the rest of the period
    os.chdir(gradientsDir)

    # get a list of all of the files in the directory (getting rid of the python checkpoints one)
    gradientFiles = os.listdir(gradientsDir)
    gradientFiles = [f for f in gradientFiles if '.nc' in f]
    concatJumpFullHist = []
    concatJumpJanDec = []

    histStart = '1850-01-16T12:00:00.000000000'
    histEnd = '2014-12-16T12:00:00.000000000'
    scenStart = '2015-01-16T12:00:00.000000000'

    for index, file in enumerate(gradientFiles):      
        # calculating the jump relative to std
        modelGradient = xr.open_dataset(file)
        jump = (modelGradient.sel(time = histEnd) - modelGradient.sel(time = scenStart)).ts.item()

        # calculating standard deviation of historical period
        stdHist = modelGradient.sel(time = slice(histStart, histEnd)).std(dim = 'time').ts.item()
        meanHist = modelGradient.sel(time = slice(histStart, histEnd)).mean(dim = 'time').ts.item()

        concatJumpFullHist.append((jump - meanHist)/stdHist)

        # calculating difference between December and January for historical period
        # first selecting the right period so that we have Jan of 1851 and December of 1850 at the start and Jan 2015 and Dec 2014 at the end
        janStart = '1851-01-16T12:00:00.000000000'
        janEnd = '2015-02-16T12:00:00.000000000'
        decStart = '1850-01-16T12:00:00.000000000'
        decEnd = '2015-01-16T12:00:00.000000000'

        janValues = modelGradient.sel(time = slice(janStart, janEnd)).where(modelGradient['month'] == 1, drop = True).ts.values
        decValues = modelGradient.sel(time = slice(decStart, decEnd)).where(modelGradient['month'] == 12, drop = True).ts.values
        janDecDiff = janValues - decValues
        janDecDiffMean = np.mean(janDecDiff)
        janDecDiffSTD = np.std(janDecDiff)

        concatJumpJanDec.append((jump - janDecDiffMean)/janDecDiffSTD)
            
    return concatJumpFullHist, concatJumpJanDec

def DictToDf(dictionary):
    '''
    Function that takes in a dictionary with keys that are date tuples and outputs a dataframe; designed with the intention of creating triangle plots
    
    Inputs:
        dictionary: dictionary that has date tuples as keys (in the format startDate, endDate) and has one value per tuple for the values (in 
        most cases likely the trend value)
        
    Outputs:
        dataframe with start date as the columns and end date as the rows with one value of trend for every instance that a trend was stored; nans for the other values
    '''
    
    dfOut = pd.DataFrame(list(dictionary.items()), columns = ['Year', 'Trend'])
    dfOut[['start_year', 'end_year']] = pd.DataFrame(dfOut['Year'].tolist(), index = dfOut.index)
    dfOut.drop('Year', axis = 1, inplace = True)
    dfOut = dfOut.pivot('end_year', 'start_year', 'Trend')
    dfOut = dfOut.sort_index(ascending = False)
    
    return dfOut


def TrendsDictFromFiles(trendsDir, modelName):
    '''
    Function that takes in a path to a directory with csv files of trends for start and end years and creates a dictionary of these trends for all of the files in the directory
    Data is cropped to start and end dates (i.e., earliest start and end date and latest start date)
    
    Inputs:
        trendsDir: Directory containing trends csv files; these files should have start years as columns and end years as rows
        modelName: string with the name of the model that you are retrieving files for (e.g., MIROC6)
    
    Outputs:
        trendsDict: Dictionary whose keys are start date, end date tuples and data points are the trends from all of the files for these start and end points
    '''
   
    # get a list of all of the files in the directory (getting rid of the python checkpoints one)
    trendFiles = os.listdir(trendsDir)
    trendFiles = [f for f in trendFiles if '.csv' in f and modelName in f]

    # initialise the dictionary
    trendsDict = defaultdict(list)

    # setting time limits for the periods that we're interested in (specifically the start and end dates at the start of the series)
    firstStartYear = 1870
    firstEndYear = 1890
    lastStartYear = 2002

    for file in trendFiles:
        trendFile = pd.read_csv(file, index_col = 0)
        trendFile.columns = [int(col) for col in trendFile.columns]

        # cropping to the period that we're interested in: 1870 and the first start and 1890 as the first end
        keepCols = [col for col in trendFile.columns if (col >= firstStartYear) & (col <= lastStartYear)]
        keepRows = [row for row in trendFile.index if row >= firstEndYear]
        trendFile = trendFile.loc[keepRows, keepCols]

        # iterate through the start and end years
        for startYear in trendFile.columns.tolist():
            for endYear in trendFile.index.tolist():
                if (endYear > startYear):
                    trend = trendFile.loc[endYear, startYear]
                    trendsDict[startYear, endYear].append(trend)

    trendsDict = dict(trendsDict)
    
    return trendsDict


def CalculateTrendPercentile(trendsDict, perLower, perUpper):
    '''
    Function that takes in a dictionary of trends with start date end date tuples as the keys and trends as the values; also takes in upper and lower percentile and 
    returns those values in two dictionaries
    
    Inputs
        trendsDict: dictionary of trends with start date, end date tuples as keys and trends as values
        perLower, perUpper: upper and lower percentile bounds
        
    Outputs
        dictLower: trend that corresponds with lower percentile for each pair of dates
        dictUpper: trend that corresponds with upper percentile for each pair of dates
    '''
    
    dictLower = defaultdict(list)
    dictUpper = defaultdict(list)

    # loop through the keys and save the values
    for key in trendsDict:
        dictLower[key] = np.percentile(trendsDict[key], perLower)
        dictUpper[key] = np.percentile(trendsDict[key], perUpper)

    dictLower = dict(dictLower)
    dictUpper = dict(dictUpper)
    
    return dictLower, dictUpper

def FlagInRange(dictLower, dictUpper, observations):
    '''
    Function that flags the dates for which the observations lie within the defined range of the trends.
    
    Inputs:
        dictLower, dictUpper: dictionaries of upper and lower percentile trends from the model spread
        observations: dictionary of trends from the observational dataset (or any dataset for which you want to check the points in range)
        
    Outputs:
        dictObsInRange: dictionary of start and end date tuples as keys, where 1's indicate that the observations are within range and 0's that there are no
        nan's are everywhere else
    '''
    dictObsInRange = defaultdict(list)

    for key in observations:
        if key in dictLower.keys():
            if not (np.isnan(dictLower[key]) | np.isnan(dictUpper[key])):
                if (observations[key] > dictLower[key]) & (observations[key] < dictUpper[key]):
                    dictObsInRange[key] = 1
                else:
                    dictObsInRange[key] = 0
            else:
                dictObsInRange[key] = np.nan
        else:
            dictObsInRange[key] = np.nan

    dictObsInRange = dict(dictObsInRange)
    
    return dictObsInRange

def CalculateModelRange(dictLower, dictUpper):
    '''
    Function that takes in two dictionaries of percentile values and calculates the difference between them (i.e., the range)
    
    Inputs
        dictUpper, dictLower: outputs from the CalculateTrendPercentile function: two dictionaries with values of upper and lower percentile where the keys are start
        and end date tuples
    
    Outputs
        dfRange: dataframe of the range for every start and end point
    '''
    # creating a new function that calculates the 95% range of the models
    dictRange = defaultdict(list)

    # iterate through dictionary
    for key in dictLower:
        dictRange[key] = dictUpper[key] - dictLower[key]

    # making a dataframe for this dictionary
    dfRange = DictToDf(dictRange)
    
    return dfRange

def CalculateObsPercentile(trendsDict, trendsObsDict):
    '''
    Function that takes in a dictionary of model trends and a dictionary of observed trends and calculates the percentile of the obs trend relative to the dist of the model trends
    
    Inputs
        trendsDict: a dictionary of model trends where the keys are start, end date tuples and the values are all of the trends from the models that are being 
        included for those dates
        trendsObsDict: a dictionary of observed trends where the keys are start, end date tuples and values are trends for those dates
    '''

    dictObsPercentile = defaultdict(list)

    # iterate through the dictionary

    for key in trendsDict:
        if np.isnan(trendsObsDict[key]):
            dictObsPercentile[key] = np.nan
        else:
            dictObsPercentile[key] = percentileofscore(trendsDict[key], trendsObsDict[key])

    # creating a dataframe for the ObsPercentile
    dfObsPercentile = DictToDf(dictObsPercentile)
    
    return dfObsPercentile

def MakeChangeDir(path):
    '''
    Function that checks to see if a directory given by a path exists; if it doesn't exist, then it creates that directory and changes into it; if it does exist it changes to that directory
    
    Inputs:
        path: path to directory that's in question
    '''
    
    if os.path.isdir(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)
    
def ScaleTrends(trends, ratio):
    '''
    Small function that scales a list of trends by a ratio provided; it scales the trend symmetrically about the mean and then adds the mean back in
    
    Inputs:
        trends: list of trend values
        ratio: scaling factor
    
    Outputs:
        list of scaled trends
    '''
    mean = np.mean(trends)
    
    trendsScaled = [(x-mean) * ratio + mean for x in trends]
    
    return trendsScaled