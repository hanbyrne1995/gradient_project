# GradientProjectClasses

import numpy as np
import xarray as xr
from math import exp, pi, sin, sqrt, log, radians, isnan
from scipy.stats import linregress
import cftime
import pandas as pd
import nc_time_axis
import random
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

class ModelInput:
    def __init__(self, modelID):
        '''
        Takes the input of a modelID (in this case URL), reads in the data and checks the coordinates (makes sure it's lat and lon)
        :param modelID: A unique ID for the model in the form of an openDAP url or in the form of the dataset itself (to handle the cases where models have been concatenated), or in some cases a list of URLs (for the cases where scenarios were not the full period)
        '''
        self.modelID = modelID
        self.ds = self.ExecAllSteps()
        
    
    # starting by assuming I will be putting the URLs in directly; ultimately move to some kind of dictionary
    def LoadData(self, modelID):
        '''
        Loads the data from the openDAP server
        '''
        try:
            
            # checking if we are dealing with a url or dealing with the dataset itself
            if isinstance(modelID, str):
                self.ds = xr.open_dataset(self.modelID)
            
            elif isinstance(modelID, list):
                self.ds = xr.open_mfdataset(self.modelID)
            
            else:
                self.ds = modelID
                
            # print('Successful data loading')
            
        except Exception as e:
            print(f'Error loading data for model {self.modelID}: {e}')
        
    # doing the data checks on the latitudes and longitudes
    def CleanCoords(self):
        '''
        Converts any coordinates that are not lat and lon into the appropriate dimension names
        '''
        try:
            if 'lat' not in self.ds.dims:
                if 'y' in self.ds.dims:
                    self.ds = self.ds.rename({'y': 'lat'})
                elif 'latitude' in self.ds.dims:
                    self.ds = self.ds.rename({'latitude': 'lat'})

            if 'lon' not in self.ds.dims:
                if 'x' in self.ds.dims:
                    self.ds = self.ds.rename({'x': 'lon'})
                elif 'longitude' in self.ds.dims:
                    self.ds = self.ds.rename({'longitude': 'lon'})
            
            # print('Successful coordinate cleaning')
            
        except Exception as e:
            
            print(f'Error checking and correcting coordinates: {e}')
    
    def FixDate(self):
        '''
        Converts all date formats to DateTime64 so no need to check start and end date types
        '''
        try:
            convertedTime = pd.to_datetime(self.ds.time.values.astype(str))
            self.ds['time'] = ('time', convertedTime)
        
        except Exception as e:
            print(f'Error in converting the time: {e}')
            
    def ExecAllSteps(self):
        self.LoadData(self.modelID)
        self.CleanCoords()
        self.FixDate()
        return self.ds
        
        
class Gradient:
    def __init__(self, modelInput):
        '''
        Calculates the area-weighted average in the two regions of interest; definitions per Seager et al 2022
        - gradient is calculated as western Pacific box average minus eastern Pacific box average
        - latitude and longitude extents defined per this paper
        
        :param modelInput: Instance of ModelInput class
        '''
        self.ds = modelInput
        self.modelName = self.SaveAttrs()
        self.boxE = None # initialising the East and West boxes to be filled
        self.boxW = None
        self.weightsE = None
        self.weightsW = None
        self.meansstE = None
        self.meansstW = None
        self.gradient = self.ExecuteAllSteps()
        
    def SaveAttrs(self):
        '''
        Saves the attributes of the dataset as the modelname that we can use to identify the model going forward
        '''
        
        try:
            selected_attrs = ['parent_source_id', 'variant_label'] 
            attributes = {attr: self.ds.attrs[attr] for attr in selected_attrs if attr in self.ds.attrs}
            self.modelName = '_'.join(attributes.values())
            
            # print('Successful attribute finding')
            
        except Exception as e:
            
            print(f'Error in loading attributes to dictionary: {e}')
        
        return self.modelName
        
    def SliceRegions(self):
        '''
        Slice the data into the regions defined in Seager et al 2022:
            East: lat: -3 to 3; lon: 190 to 270
            West: lat: -3 to 3; lon: 140 to 170
        '''
        try:
            # region_East
            lonminE, lonmaxE = 190, 270
            latminE, latmaxE = -3,3

            # region_West
            lonminW, lonmaxW = 140, 170
            latminW, latmaxW = -3, 3

            # slicing the data into these regions
            self.boxE = self.ds.ts.sel(lon = slice(lonminE, lonmaxE), lat = slice(latminE, latmaxE))
            self.boxW = self.ds.ts.sel(lon = slice(lonminW, lonmaxW), lat = slice(latminW, latmaxW))

            # print('Successful region slicing')

        except Exception as e:
            
            print(f'Error slicing regions: {e}')
            
    def CalculateGradient(self):
        '''
        Calculates the difference in weighted average sst variable (West - East)
        '''
        try:
            # note that this isn't regridded but there isn't areacello available so will estimate using latitude
            self.weightsE = np.cos(np.radians(self.boxE.lat))
            self.weightsW = np.cos(np.radians(self.boxW.lat))

            # calculating the mean monthly weighted SST for each box
            # ADDED IN COMPUTE COMMAND HERE
            self.meansstE = self.boxE.weighted(self.weightsE).mean(('lat', 'lon')).compute()
            self.meansstW = self.boxW.weighted(self.weightsW).mean(('lat', 'lon')).compute()

            # calculate the temperature difference (W - E)
            self.gradient = self.meansstW - self.meansstE
            
            # print('Successful gradient calculation')
        
        except Exception as e:
            
            print(f'Error calculating gradient: {e}')
        
    def ExecuteAllSteps(self):
        self.SliceRegions()
        self.CalculateGradient()
        return self.gradient
        
class CalculateMMEGradient:
    def __init__(self, gradientsDir, modelName):
        '''
        Class that takes in the directory of the pre-calcalulated .nc gradient files, concatenates them and calculates the mean gradient across all of them (specific to MME)
        
        :param: gradientsDir: directory with gradient files
        :param: modelName: name of the model in question
        
        Outputs: gradientMean: dataarray with the mean gradient calculated, modelName pulled from the file names
        '''
        self.gradientsDir = gradientsDir
        self.modelName = modelName
        self.gradient = self.ExecuteAllSteps()

        
    def CalculateMean(self, gradientsDir, modelName):
        # run a for loop that concatenates all of the input files along the same new dimension ('gradient') then takes the mean along that dimension
        os.chdir(gradientsDir)

        # get a list of all of the files in the directory (getting rid of the python checkpoints one)
        gradientFiles = os.listdir(gradientsDir)
        gradientFiles = [f for f in gradientFiles if '.nc' in f and modelName in f]

        # now iterate through the list and concatenate them

        # first initialise an xarray file for concatenation
        gradientConcat = xr.open_dataset(gradientFiles[0])

        for index, file in enumerate(gradientFiles):
            if index > 0:
                modelGradient = xr.open_dataset(file)
                gradientConcat = xr.concat([gradientConcat, modelGradient], dim = 'gradient')

        # now calculating the mean along the gradient dimension
        self.gradientMean = gradientConcat.mean(dim = 'gradient')

        # cutting this off at for the period that we are interested in (depends on date time format)
        
        start_year = 1850
        monStart = 1
        dayStart = 1
        end_year = 2022
        monEnd = 12
        dayEnd = 31
        
        start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
        end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

        self.gradientMean = self.gradientMean.ts.sel(time = slice(start_date, end_date))

    def ExecuteAllSteps(self):
        self.CalculateMean(self.gradientsDir, self.modelName)
        return self.gradientMean

class CalcMMEGradSubset:
    def __init__(self, gradientsDir, modelName, subsetList):
        '''
        Like the CalculateMMEGradient class, but includes another input that is a list that the input files must be in (e.g., a list of specific model names)
        
        Class that takes in the directory of the pre-calcalulated .nc gradient files, concatenates them and calculates the mean gradient across all of them (specific to MME)
        
        :param: gradientsDir: directory with gradient files
        :param: modelName: name of the model in question
        :param: subsetList: list of model names (for example) that the input file would need to be in to be included
        
        Outputs: gradientMean: dataarray with the mean gradient calculated, modelName pulled from the file names
        '''
        self.gradientsDir = gradientsDir
        self.modelName = modelName
        self.subsetList = subsetList
        self.gradient = self.ExecuteAllSteps()

        
    def CalculateMean(self, gradientsDir, modelName, subsetList):
        # run a for loop that concatenates all of the input files along the same new dimension ('gradient') then takes the mean along that dimension
        os.chdir(gradientsDir)

        # get a list of all of the files in the directory (getting rid of the python checkpoints one)
        gradientFiles = os.listdir(gradientsDir)
        gradientFiles = [f for f in gradientFiles if '.nc' in f and modelName in f and f[:len(f) - 3] in subsetList]

        # now iterate through the list and concatenate them

        # first initialise an xarray file for concatenation
        gradientConcat = xr.open_dataset(gradientFiles[0])

        for index, file in enumerate(gradientFiles):
            if index > 0:
                modelGradient = xr.open_dataset(file)
                gradientConcat = xr.concat([gradientConcat, modelGradient], dim = 'gradient')

        # now calculating the mean along the gradient dimension
        self.gradientMean = gradientConcat.mean(dim = 'gradient')

        # cutting this off at for the period that we are interested in (depends on date time format)
        
        start_year = 1850
        monStart = 1
        dayStart = 1
        end_year = 2022
        monEnd = 12
        dayEnd = 31
        
        start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
        end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

        self.gradientMean = self.gradientMean.ts.sel(time = slice(start_date, end_date))

    def ExecuteAllSteps(self):
        self.CalculateMean(self.gradientsDir, self.modelName, self.subsetList)
        return self.gradientMean
        
class CalculateObsGradient:
    def __init__(self, modelInput, datasetName):
        '''
        Gradient calculator class for observational data
        ** Note that current variable is sst; change this in CalculateGradient function below **
        Calculates the area-weighted average in the two regions of interest; definitions per Seager et al 2022
        - gradient is calculated as western Pacific box average minus eastern Pacific box average
        - latitude and longitude extents defined per this paper
        
        :param modelInput: Instance of ModelInput class
        :param datasetName: String of the name of the dataset in question (e.g., 'Hadley')
        '''
        self.ds = modelInput
        self.modelName = datasetName
        self.boxE = None # initialising the East and West boxes to be filled
        self.boxW = None
        self.weightsE = None
        self.weightsW = None
        self.meansstE = None
        self.meansstW = None
        self.gradient = self.ExecuteAllSteps()
        
    def SaveAttrs(self):
        '''
        Saves the attributes of the dataset as the modelname that we can use to identify the model going forward
        '''
        
        try:
            selected_attrs = ['parent_source_id', 'variant_label'] 
            attributes = {attr: self.ds.attrs[attr] for attr in selected_attrs if attr in self.ds.attrs}
            self.modelName = '_'.join(attributes.values())
            
            # print('Successful attribute finding')
            
        except Exception as e:
            
            print(f'Error in loading attributes to dictionary: {e}')
        
        return self.modelName
        
    def SliceRegions(self):
        '''
        Slice the data into the regions defined in Seager et al 2022:
            East: lat: -3 to 3; lon: 190 to 270
            West: lat: -3 to 3; lon: 140 to 170
        '''
        try:
            # region_East
            lonminE, lonmaxE = 190, 270
            latminE, latmaxE = -3,3

            # region_West
            lonminW, lonmaxW = 140, 170
            latminW, latmaxW = -3, 3

            # slicing the data into these regions
            self.boxE = self.ds.sst.sel(lon = slice(lonminE, lonmaxE), lat = slice(latmaxE, latminE))
            self.boxW = self.ds.sst.sel(lon = slice(lonminW, lonmaxW), lat = slice(latmaxW, latminW))

            # print('Successful region slicing')

        except Exception as e:
            
            print(f'Error slicing regions: {e}')
            
    def CalculateGradient(self):
        '''
        Calculates the difference in weighted average sst variable (West - East)
        '''
        try:
            # note that this isn't regridded but there isn't areacello available so will estimate using latitude
            self.weightsE = np.cos(np.radians(self.boxE.lat))
            self.weightsW = np.cos(np.radians(self.boxW.lat))

            # calculating the mean monthly weighted SST for each box
            self.meansstE = self.boxE.weighted(self.weightsE).mean(('lat', 'lon'), skipna = True)
            self.meansstW = self.boxW.weighted(self.weightsW).mean(('lat', 'lon'), skipna = True)

            # calculate the temperature difference (W - E)
            self.gradient = self.meansstW - self.meansstE
            
            # print('Successful gradient calculation')
        
        except Exception as e:
            
            print(f'Error calculating gradient: {e}')
        
    def ExecuteAllSteps(self):
        self.SliceRegions()
        self.CalculateGradient()
        return self.gradient
        
        
class Trend:
    def __init__(self, gradient, minTrend, startYear, endYear):
        '''
        Calculates the trends for different start and end points for based on the input gradient time series
        
        Inputs:
        ::param gradient: instance of Gradient class (a gradient time series)
            
        Outputs:
            A dictionary of trends for the time periods indicated below
        '''
        self.gradient = gradient.gradient
        self.modelName = gradient.modelName
        self.minTrend = minTrend
        self.startYear = startYear
        self.endYear = endYear
        self.trendsDf = None
        self.trends = self.ExecuteAllSteps()
    
    
    def CalculateTrends(self):
        '''
        Calculates the moving interval trends (different beginning and end points)
        '''
        try:
            # initialising timing constants
            monStart = 1 # January
            monEnd = 12   # December
            yearStart = self.startYear # Note that this can be changed 
            yearEnd = self.endYear
            dayStart = 1
            dayEnd = 31
            interval = 1 # years
            trendLength = 120 # months
            minTrend = self.minTrend
            self.trends = {}

            # calculating a time index to use in the polyfit
            indexTime = np.arange(len(self.gradient.time))

            # for loops to calculate the trends for each start and end year combination
            for start_year in range(yearStart, yearEnd+1 - minTrend, interval):
                for end_year in range(yearStart + minTrend, yearEnd+1, interval):

                    # fill the triangle where end date is before start date with NaNs
                    if start_year >= end_year:
                        self.trends[start_year, end_year] = np.nan
                    
                    # fill the triangle in for lengths that are shorter than the minTrend set
                    elif end_year - start_year < minTrend:
                        self.trends[start_year, end_year] = np.nan
                
                    else:
                        # create start and end dates for this period to subset the data
                        # check what the format of the time data is (checking the first element)
                        
                        start_date = np.datetime64(f'{start_year}-{monStart:02d}-{dayStart:02d}')
                        end_date = np.datetime64(f'{end_year}-{monEnd:02d}-{dayEnd:02d}')            

                        # create the subsetted dataset and subsetted time index
                        gradientSubset = self.gradient.sel(time = (self.gradient.time >= start_date) & (self.gradient.time <= end_date))
                        indexTimeSubset = indexTime[:len(gradientSubset.time)]

                        # calculate the slope and intercept; multiply slope by number of months for units of K per length of trend period
                        if len(indexTimeSubset) > 12*interval:
                            slope, intercept = np.polyfit(indexTimeSubset, gradientSubset, 1)
                            self.trends[start_year, end_year] = slope*trendLength
                        
                        else:
                            self.trends[start_year, end_year] = np.nan
                            
            # print('Successful trend calculation')
                            
        except Exception as e:
            
            print(f'Error calculating trends: {e}')
            
    def CreateDataFrame(self):
        '''
        Creates a dataframe of the trends that can be stored
        '''
        try:
            self.trendsDf = pd.DataFrame(list(self.trends.items()), columns = ['Year', 'Trend'])
            self.trendsDf[['start_year', 'end_year']] = pd.DataFrame(self.trendsDf['Year'].tolist(), index = self.trendsDf.index)
            self.trendsDf.drop('Year', axis = 1, inplace = True)
            self.trendsDf = self.trendsDf.pivot('end_year', 'start_year', 'Trend')
            self.trendsDf = self.trendsDf.sort_index(ascending = False)
    
        
        except Exception as e:
            print(f'Error in creating dataframe: {e}')
        
        
    def ExecuteAllSteps(self):
        self.CalculateTrends()
        self.CreateDataFrame()
        return self.trends
    
class CalcIntervalDiff:
    def __init__(self, timeSeries, minInterval):
        '''
        Calculates the difference in values between start and end years in an array
        
        Inputs:
        ::param timeSeries: an xarray dataarray that has one attribute and has the time dimension in years
        ::param minInterval: the minimum interval over which you would like the difference to be calculated
        
            
        Outputs:
            A dictionary of trends for the time periods indicated below
            A dataframe of trends for the time periods
        '''
        self.timeseries = timeSeries
        self.minInterval = minInterval
        self.intervalDiffsDf = None
        self.intervalDiffs = self.ExecuteAllSteps()
    
    
    def CalculateIntervalDiff(self):
        '''
        Calculates the moving interval differences (different beginning and end points)
        '''
        try:
            # initialising timing constants
            yearStart = 1870 # Note that this can be changed | note that 1880 choice is for NASA GISS
            yearEnd = 2022
            minInterval = self.minInterval # years
            self.intervalDiffs = {}

            # for loops to calculate the trends for each start and end year combination
            for start_year in range(yearStart, yearEnd+1 - minInterval, 1):
                for end_year in range(yearStart + minInterval, yearEnd+1, 1):

                    # fill the triangle where end date is before start date with NaNs
                    if start_year >= end_year:
                        self.intervalDiffs[start_year, end_year] = np.nan
                    
                    # fill the triangle in for lengths that are shorter than the minTrend set
                    elif end_year - start_year < minInterval:
                        self.intervalDiffs[start_year, end_year] = np.nan
                
                    else:
                        # calculate the difference between the end year and start year values
                        intervalDiff = self.timeseries.sel(year = end_year) - self.timeseries.sel(year = start_year)
                        self.intervalDiffs[start_year, end_year] = intervalDiff.item()
                            
        except Exception as e:
            
            print(f'Error calculating interval differences: {e}')
            
    def CreateDataFrame(self):
        '''
        Creates a dataframe of the trends that can be stored
        '''
        try:
            self.intervalDiffsDf = pd.DataFrame(list(self.intervalDiffs.items()), columns = ['Year', 'Difference'])
            self.intervalDiffsDf[['start_year', 'end_year']] = pd.DataFrame(self.intervalDiffsDf['Year'].tolist(), index = self.intervalDiffsDf.index)
            self.intervalDiffsDf.drop('Year', axis = 1, inplace = True)
            self.intervalDiffsDf = self.intervalDiffsDf.pivot('end_year', 'start_year', 'Difference')
            self.intervalDiffsDf = self.intervalDiffsDf.sort_index(ascending = False)
    
        
        except Exception as e:
            print(f'Error in creating dataframe: {e}')
        
        
    def ExecuteAllSteps(self):
        self.CalculateIntervalDiff()
        self.CreateDataFrame()
        return self.intervalDiffs

class TrendPlotting:    
    def __init__(self, trendsDf, modelName, vmin, vmax, cmap, norm):
        '''
        Plots the trends calculated in the Gradient class as heatmaps

        :param trendsDict: Dictionary of trends for one of the models from the GradientTrend class
        :param modelName: Name to be printed on the top of the plot (e.g., MIROC6_r1 etc.)
        :param vmin and vmax: calculated in the script
        :param cmap: colourmap created with a diverging colour palette
        :param norm: related to the colourmap

        '''
        self.modelName = modelName
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.norm = norm
        self.trendsDf = trendsDf
        self.heatmap = None

    def PlotTrends(self, ax):
        '''
        First converts the input trends dictionary into a dataframe.
        Plot this dataframe as a heatmap.
        '''
        try:
            # creating a figure
            self.heatmap = sns.heatmap(self.trendsDf, 
                                       ax = ax, 
                                       vmin = self.vmin, 
                                       vmax = self.vmax, 
                                       cmap = self.cmap, 
                                       xticklabels = 10,
                                       yticklabels = 10,
                                       center = 0, 
                                       cbar = False, 
                                       norm = self.norm)

            plt.xticks(rotation=0)
            plt.yticks(rotation=45)
            title = self.modelName
            plt.title(title, fontsize = 16)
            plt.ylabel('End year', fontsize = 12)
            plt.xlabel('Start year', fontsize = 12)

            
        except Exception as e:
            print(f'Error in plotting trends: {e}')