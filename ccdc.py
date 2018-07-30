import pandas as pd
import numpy as np
import sys
import datacube
import xarray as xr
import argparse
import csv
import multiprocessing
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datacube.storage.masking import mask_invalid_data
from datacube.api import GridWorkflow
from makeModel import MakeCCDCModel
from removeOutliers import RLMRemoveOutliers
from datetime import datetime
from osgeo import ogr
from random import uniform
from scipy.interpolate import interp1d
from sklearn.externals import joblib

plt_list = []        # List of plots, one for each band

def addChangeMarker(num_bands, start_change, end_change, obs_data):

    """ Adds vertical lines to each plot every time change is detected """
       
    for i in range(num_bands):
        y_min = np.amin(obs_data[:,i+1])
        y_max = np.amax(obs_data[:,i+1])

        plt_list[i].plot([start_change, start_change], [y_min, y_max], 'r', linewidth=2)
        plt_list[i].plot([end_change, end_change], [y_min, y_max], 'y', linewidth=2)

        interp = interp1d(model_list[i].getDateTimes(), model_list[i].getPredicted(), kind='cubic')
        xnew = np.linspace(model_list[i].getDateTimes().min(), model_list[i].getDateTimes().max(), 500)
        
        plt_list[i].plot(xnew, interp(xnew), 'm-', linewidth=2)
        
def setupModels(all_band_data, num_bands, init_obs):
    
    """Creates a model for each band and stores it in model_list"""
    
    # Get column of datetime values
    datetimes = all_band_data[:,0]
    
    # Create a model for each band and store it in model_list
    for i in range(1, all_band_data.shape[1]):
        
        band_data = all_band_data[:,i]
   
        ccdc_model = MakeCCDCModel(datetimes, band_data)
            
        ccdc_model.fitModel(init_obs)
        
        model_list[i-1] = ccdc_model

def getNumYears(date_list):

    """Get number of years (from Python/Rata Die date)"""
    
    last_date = datetime.fromordinal(np.amax(date_list).astype(int)).strftime('%Y')
    first_date = datetime.fromordinal(np.amin(date_list).astype(int)).strftime('%Y')
        
    num_years = int(last_date) - int(first_date)

    return num_years

def transformToArray(dataset_to_transform):

    """Transforms xarray Dataset object into a Numpy array"""
    
    ds_to_array = np.array([pd.Timestamp(x).toordinal() for x in dataset_to_transform.time.data]).reshape(-1, 1)
    
    for var in dataset_to_transform.data_vars:
        ds_to_array = np.hstack((ds_to_array, dataset_to_transform[var].values.reshape(-1, 1)))

    return ds_to_array

def doTmask(input_ts, tmask_ts):
    
    """"Removes outliers from the input dataset using Tmask if possible."""
    
    # Remove NaNs from TOA data
    tmask_nan_mask = np.any(np.isnan(tmask_ts), axis=1)
    tmask_ts = tmask_ts[~tmask_nan_mask]
    
     # Sort data by date
    tmask_ts = tmask_ts[np.argsort(tmask_ts[:,0])]
     
    # Both datasets need to contain the same observations
    if(np.array_equal(input_ts[:,0], tmask_ts[:,0])):
         
        # Get the number of years covered by the data
        num_years = getNumYears(input_ts[:,0])
         
        if(num_years > 0):
             
            # Screen for outliers
            robust_outliers = RLMRemoveOutliers(tmask_ts, input_ts)
    
            input_ts = robust_outliers.cleanData(num_years)
            
        else:
           print("Need at least 1 year of data to screen using Tmask.")
           
    else:
       print("Input dataset and Tmask dataset do not match.")
            
    return input_ts            

def initModel(pixel_data, num_bands, init_obs):

    """Finds a sequence of 6/12/18/24 consecutive clear observations without any change, to initialize the model"""

    # Subset first n clear observations for model initialisation
    curr_obs_list = pixel_data[:init_obs,]
    
    # Start off with the model uninitialized
    model_init = False
    num_iters = 0
    
    # The next observation to be added to the model to detect change
    init_end = None

    # Model initialization sequence - keeps going until a clear set of observations is found
    while(model_init == False):

        num_data_points = len(curr_obs_list)
        
        if(num_data_points < init_obs):
            #print("Could not find a period of no change for model initialization.")
            return None
    
        # Re-initialize the models
        setupModels(curr_obs_list, num_bands, init_obs)

        # Get total time used for model initialization
        total_time = np.max(curr_obs_list[:,0]) - np.min(curr_obs_list[:,0])
        
        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0
  
        # Check for change during the initialization period. We need 12 observations with no change
        for band_model in model_list: # For each model
            
            slope_val = np.absolute(band_model.getCoefficients()[0]) / (3 * band_model.getRMSE() / total_time)
            total_slope_eval += slope_val
        
            start_val = np.absolute(band_model.getBandData()[0] - band_model.getPredicted()[0]) / (3 * band_model.getRMSE())
            total_start_eval += start_val
            
            end_val = np.absolute(band_model.getBandData()[num_data_points-1] - band_model.getPredicted()[num_data_points-1]) / (3 * band_model.getRMSE())
            total_end_eval += end_val
 
        if((total_slope_eval / num_bands) > 1 or (total_start_eval / num_bands) > 1 or (total_end_eval / num_bands) > 1):
            num_iters += 1
            curr_obs_list = pixel_data[0+num_iters:init_obs+num_iters,:] # Shift along 1 row
        
        else:
            model_init = True
            init_end = init_obs + num_iters + 1
            #print("Model initialized. Iterations needed: {}".format(num_iters))

    return curr_obs_list, init_end

def findChange(pixel_data, change_file, num_bands, init_obs, args):
    
    """Continues to add data points to the model until either a new breakpoint is detected, or there
        are not enough observations remaining."""

    try:
        model_data, next_obs = initModel(pixel_data, num_bands, init_obs)
    except TypeError:
        return []

    # Detect change
    change_flag = 0
    change_start_time = None
    
    num_new_obs = 0

    while((next_obs+1) <= len(pixel_data)):

        change_eval = 0
        
        new_obs = pixel_data[next_obs,]
        new_date = new_obs[0]
        
        for i in range(1, num_bands+1):    # For each band
            new_ref_obs = new_obs[i]
            residual_val = np.absolute(new_ref_obs - model_list[i-1].getPrediction(new_date)[0]) / (2 * model_list[i-1].getRMSE())
            change_eval += residual_val
            
        if((change_eval / num_bands) <= 1):
            #print("Adding new data point")
            model_data = np.append(model_data, [new_obs], axis=0)
            
            num_new_obs += 1
            
            if(num_new_obs == args.re_init):
                setupModels(model_data, num_bands, init_obs)
                num_new_obs = 0
                
            change_flag = 0 # Reset change flag because we have an inlier

        else:
            change_flag += 1 # Don't add the new pixel to the model

            if(change_flag == 1): # If this is the first observed possible change point
                change_start_time = new_date
    
        if(change_flag == 6):
            #print("Change detected!")
            
            if(args.outtype == 'plot'):
                addChangeMarker(num_bands, change_start_time, new_date, pixel_data)

            else:
               with open(change_file, 'a') as output_file:
                  writer = csv.writer(output_file)
                  writer.writerow([datetime.fromordinal(int(change_start_time)).strftime('%d/%m/%Y'), datetime.fromordinal(int(new_date)).strftime('%d/%m/%Y')])
            
            # Pickle current models
            if(args.save_models):
                for model_num, model in enumerate(model_list):
                    pkl_file = "{}_{}_{}_{}.pkl".format(change_file.rsplit('.', 1)[0], model.getMinDate(), model.getMaxDate(), model_num)
                    joblib.dump(model, pkl_file) 
                    
            
            return pixel_data[next_obs:,]
        
        # Need to get the next observation
        next_obs += 1
    
    # No change detected, end of data reached
    return []

def runCCDC(input_data, change_file, num_bands, args):

    """The main function which runs the CCDC algorithm. Loops until there are not enough observations
        left after a breakpoint to attempt to initialize a new model."""
    
    num_changes = 0
        
    # Get the number of years covered by the dataset
    num_years = getNumYears(input_data[:,0])
    
    # The algorithm needs at least 1 year of data (after any screening)
    if(num_years > 0 and len(input_data) >= 12):
                              
        if(args.outtype == 'plot'):
            
            change_file = change_file + ".png"

            fig = plt.figure(figsize=(20, 10))
            
            # Set up plots with original data and screened data
            for i in range(num_bands):
                plt_list.append(fig.add_subplot(num_bands, 1, i+1))
                plt_list[i].plot(input_data[:,0], input_data[:,i+1], 'o', color='c', label='Original data', markersize=4)
                myFmt = mdates.DateFormatter('%m/%Y') # Format dates as month/year rather than ordinal dates
                plt_list[i].xaxis.set_major_formatter(myFmt)
                #plt_list[i].set_ylim(bottom=0, top=500)

        else:
              change_file = change_file + ".csv"
              with open(change_file, 'w') as output_file:
                 writer = csv.writer(output_file)
                 writer.writerow(['start_change', 'end_change'])
                 
        # We need at least 12 clear observations (6 + 6 to detect change)
        while(len(input_data) >= 12):
            if(getNumYears(input_data[:,0]) > 0):

                # Get total number of clear observations in the dataset
                num_clear_obs = len(input_data)
        
                if(num_clear_obs < 18):
                    # Use simple model with initialization period of 6 obs
                    input_data = findChange(input_data, change_file, num_bands, 6, args)
                
                elif(num_clear_obs >= 18 and num_clear_obs < 24):
                    # Use simple model with initialization period of 12 obs
                    input_data = findChange(input_data, change_file, num_bands, 12, args)

                elif(num_clear_obs >= 24 and num_clear_obs < 30):
                    # Use advanced model with initialization period of 18 obs
                    input_data = findChange(input_data, change_file, num_bands, 18, args)
                
                elif(num_clear_obs >= 30):
                    # Use full model with initialisation period of 24 obs
                    input_data = findChange(input_data, change_file, num_bands, 24, args)

                if(len(input_data) > 0):
                    num_changes = num_changes + 1
                
            else:
                #print("Less than 1 year of observations remaining.")
                break                               

        #print("Ran out of observations.")

        if(args.outtype == 'plot'):

            for i in range(num_bands):
                interp = interp1d(model_list[i].getDateTimes(), model_list[i].getPredicted(), kind='cubic')
                xnew = np.linspace(model_list[i].getDateTimes().min(), model_list[i].getDateTimes().max(), 500)
                plt_list[i].plot(xnew, interp(xnew), 'm-', linewidth=2) # Plot fitted model

            # Plot empty datasets so start/end of change is included in legend
            plt.plot([], [], 'r', label='Start change')
            plt.plot([], [], 'y', label='End change')
            plt.plot([], [], 'm', label='Fitted model')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(change_file)
            plt.close(fig)
         
        # Save final models if requested
        if(args.save_models):
            for model_num, model in enumerate(model_list):
                pkl_file = "{}_{}_{}_{}.pkl".format(change_file.rsplit('.', 1)[0], model.getMinDate(), model.getMaxDate(), model_num)
                joblib.dump(model, pkl_file) 
                             

def runOnSubset(num_bands, args):

    """If the user chooses to run the algorithm on a random subsample of the data, this function is called.
        This function creates a polygon shape from the lat/long points provided by the user. It then selects
        random points from within this shape and runs the algorithm on those points. This is quite slow because 
        each point has to be loaded seperately."""
        
    # Calculate the right number of columns to be returned from the data cube
    input_num_cols = num_bands + 1
    
    input_products = args.input_products

    # Set some spatial boundaries for the data 
    boundary = ogr.Geometry(ogr.wkbLinearRing)
    boundary.AddPoint(args.lowerlat, args.lowerlon) 
    boundary.AddPoint(args.lowerlat, args.upperlon)
    boundary.AddPoint(args.upperlat, args.upperlon)
    boundary.AddPoint(args.upperlat, args.lowerlon)
    boundary.AddPoint(args.lowerlat, args.lowerlon)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(boundary)

    envelope = poly.GetEnvelope()

    min_lat, max_lat, min_long, max_long = envelope[0], envelope[1], envelope[2], envelope[3]

    num_points = args.num_points # Defaults to 100

    curr_points = 0

    for i in range(num_points):
        
        while(curr_points < num_points):
            
            new_point = ogr.Geometry(ogr.wkbPoint)
            new_point_lat = uniform(min_lat, max_lat)
            new_point_long = uniform(min_long, max_long)
        
            new_point.AddPoint(new_point_lat, new_point_long)

            if(new_point.Within(poly)):
               
                input_ds = []
                tmask_ds = []

                dc = datacube.Datacube()

                for product in input_products:
                    dataset = dc.load(product=product, measurements=args.bands, lat=(new_point_lat), lon=(new_point_long))

                    if(dataset.variables):
                        input_ds.append(dataset)  
                        
                if(args.tmask_products):
                    
                    for product in args.tmask_products:
                        dataset = dc.load(product=product, measurements=['green', 'nir', 'swir1'], lat=(new_point_lat), lon=(new_point_long))

                        if(dataset.variables):
                            tmask_ds.append(dataset)
                              
                dc.close()
                        
                if(len(input_ds) == len(input_products)):
                    
                    # Tidy up input data
                    input_data = xr.concat(input_ds, dim='time')
                    input_data = mask_invalid_data(input_data)
                    
                    # Do the same for TOA data if present - tmask_ds will be empty if no TOA data sets were specified
                    if(len(tmask_ds) > 0):
                
                        tmask_data = xr.concat(tmask_ds, dim='time')
                        tmask_data = mask_invalid_data(tmask_data)
                
                    input_data = transformToArray(input_data)
                    
                    # Remove NaNs
                    input_nan_mask = np.any(np.isnan(input_data), axis=1)
                    input_data = input_data[~input_nan_mask]
                
                    # Sort by date
                    input_data = input_data[np.argsort(input_data[:,0])]

                    if(input_data.shape[1] == input_num_cols):
                        
                        if(len(tmask_ds) > 0):                    
                        
                            tmask_data = transformToArray(tmask_data)
                        
                            if(tmask_data.shape[1] == 4): # Tmask data should always have 4 columns
                            
                                input_data = doTmask(input_data, tmask_data)
                           
                        x_coord = "{0:.6f}".format(new_point.GetX())
                        y_coord = "{0:.6f}".format(new_point.GetX())
                        
                        change_file = "{}{}_{}".format(args.outdir, x_coord, y_coord)
                        
                        runCCDC(input_data, change_file, num_bands, args)
                        curr_points += 1

def runOnArea(num_bands, args):

    """If the user chooses to run the algorithm on the whole of the specified area, this function is called.
    This function will load the whole area specified by the user, and run the algorithm on each pixel."""
    
    # Calculate the right number of columns to be returned from the data cube
    input_num_cols = num_bands + 1
    
    input_products = args.input_products
    
    input_ds = []
    tmask_ds = []
    
    num_processes = args.num_procs
    
    processes = [] 

    dc = datacube.Datacube()

    for product in input_products:
        dataset = dc.load(product=product, measurements=args.bands, lat=(args.lowerlat, args.upperlat), lon=(args.lowerlon, args.upperlon))
        
        if(dataset.variables):
            input_ds.append(dataset)
            
    if(args.tmask_products):
        
        for product in args.tmask_products:
            dataset = dc.load(product=product, measurements=['green', 'nir', 'swir1'], lat=(args.lowerlat, args.upperlat), lon=(args.lowerlon, args.upperlon))
        
            if(dataset.variables):
                tmask_ds.append(dataset)
            
    dc.close()
    
    if(len(input_ds) == len(input_products)):      
        
        # Tidy up input data
        input_data = xr.concat(input_ds, dim='time')
        input_data = mask_invalid_data(input_data)
        
        # Do the same for TOA data if present - tmask_ds will be empty if no TOA data sets were specified
        if(len(tmask_ds) > 0):
                
            tmask_data = xr.concat(tmask_ds, dim='time')
            tmask_data = mask_invalid_data(tmask_data)

        for i in range(len(input_data.x)):
            for j in range(len(input_data.y)):
                
                input_ts = input_data.isel(x=i, y=j)
                
                x_val = float(input_ts.x)
                y_val = float(input_ts.x)
       
                input_ts = transformToArray(input_ts)
                
                # Remove NaNs
                input_nan_mask = np.any(np.isnan(input_ts), axis=1)
                input_ts = input_ts[~input_nan_mask]
                
                # Sort by date
                input_ts = input_ts[np.argsort(input_ts[:,0])]
    
                if(input_ts.shape[1] == input_num_cols):
                    
                    if(len(tmask_ds) > 0):
                    
                        tmask_ts = tmask_data.isel(x=i, y=j)
                        
                        tmask_ts = transformToArray(tmask_ts)
                        
                        if(tmask_ts.shape[1] == 4): # Tmask data should always have 4 columns
                            
                           input_ts = doTmask(input_ts, tmask_ts)
                        
                    change_file = "{}{}_{}".format(args.outdir, str(x_val), str(y_val))
                    
                    # Block until a core becomes available
                    while(True):

                        p_done = []

                        for index, p in enumerate(processes):
                                
                            if(not p.is_alive()):
                                p_done.append(index)

                        if(p_done):
                            for index in sorted(p_done, reverse=True): # Need to delete in reverse order to preserve indexes
                                del(processes[index])

                        if(len(processes) < num_processes):
                            break
                                    
                    process = multiprocessing.Process(target=runCCDC, args=(input_ts, change_file, num_bands, args))
                    processes.append(process)
                    process.start()

    # Keep running until all processes have finished
    for p in processes:
        p.join()

def runByTile(key, num_bands, args):

    """Lets you process data using cell keys and x/y extent.
       A key represent one cell/area. Each cell has a tile for each time point. The x and y values define the extent of
       the tile that should be loaded and processed."""
       
    # Calculate the right number of columns to be returned from the data cube
    input_num_cols = num_bands + 1
    
    num_processes = args.num_procs
    processes = []
    
    input_products = args.input_products
       
    min_x, max_x = args.tile_x_min, args.tile_x_max
    
    min_y, max_y = args.tile_y_min, args.tile_y_max

    input_ds = []
    tmask_ds = []

    for product in input_products:

        dc = datacube.Datacube()

        # Create the GridWorkflow object for this product
        curr_gw = GridWorkflow(dc.index, product=product)

        # Get the list of tiles (one for each time point) for this product
        tile_list = curr_gw.list_tiles(product=product, cell_index=key)

        dc.close()

        # Retrieve the specified pixel for each tile in the list
        for tile_index, tile in tile_list.items():
            dataset = curr_gw.load(tile[0:1, min_x:max_x, min_y:max_y], measurements=args.bands)

            if(dataset.variables):
                input_ds.append(dataset)
                
    if(args.tmask_products): # If tmask should be used to screen for outliers
        
        for product in args.tmask_products:
            
            dc = datacube.Datacube()
            
            # Create the GridWorkflow object for this product
            curr_gw = GridWorkflow(dc.index, product=product)
            
            # Get the list of tiles (one for each time point) for this product
            tile_list = curr_gw.list_tiles(product=product, cell_index=key)
            
            dc.close()
            
            # Retrieve the specified pixel for each tile in the list
            for tile_index, tile in tile_list.items():
                dataset = curr_gw.load(tile[0:1, min_x:max_x, min_y:max_y], measurements=['green', 'nir', 'swir1'])
                
                if(dataset.variables):
                    tmask_ds.append(dataset)
                
    if(len(input_ds) > 0): # Check that there is actually some input data
                
        # Tidy up input data
        input_data = xr.concat(input_ds, dim='time')
        input_data = mask_invalid_data(input_data)
        
        # Do the same for TOA data if present - tmask_ds will be empty if no TOA data sets were specified
        if(len(tmask_ds) > 0):
                
            tmask_data = xr.concat(tmask_ds, dim='time')
            tmask_data = mask_invalid_data(tmask_data)
                    
        for i in range(len(input_data.x)):
            for j in range(len(input_data.y)):
                
                input_ts = input_data.isel(x=i, y=j)
                
                x_val = float(input_ts.x)
                y_val = float(input_ts.y)
                
                input_ts = transformToArray(input_ts)
                
                # Remove NaNs
                input_nan_mask = np.any(np.isnan(input_ts), axis=1)
                input_ts = input_ts[~input_nan_mask]
                
                # Sort by date
                input_ts = input_ts[np.argsort(input_ts[:,0])]
            
                if(input_ts.shape[1] == input_num_cols):
                    
                    if(len(tmask_ds) > 0):
                    
                        tmask_ts = tmask_data.isel(x=i, y=j)
                        
                        tmask_ts = transformToArray(tmask_ts)
                        
                        if(tmask_ts.shape[1] == 4): # Tmask data should always have 4 columns
                            
                           input_ts = doTmask(input_ts, tmask_ts)
                                                                                              
                    change_file = "{}{}_{}".format(args.outdir, str(x_val), str(y_val))
                    
                    # Block until a core becomes available
                    while(True):
    
                        p_done = []
    
                        for index, p in enumerate(processes):
                                    
                            if(not p.is_alive()):
                                p_done.append(index)
    
                        if(p_done):
                            for index in sorted(p_done, reverse=True): # Need to delete in reverse order to preserve indexes
                                del(processes[index])
    
                        if(len(processes) < num_processes):
                            break
                                        
                    process = multiprocessing.Process(target=runCCDC, args=(input_ts, change_file, num_bands, args))
                    processes.append(process)
                    process.start()

    # Keep running until all processes have finished
    for p in processes:
        p.join()

def runAll(num_bands, args):

    """Run on all tiles in the specified datasets/area. Keys are based on the last dataset listed."""

    # Calculate the right number of columns to be returned from the data cube
    input_num_cols = num_bands + 1
    
    num_processes = args.num_procs
    processes = []
    
    input_products = args.input_products

    dc = datacube.Datacube()

    # Create Gridworkflow object for most recent dataset
    gw = GridWorkflow(dc.index, product=input_products[-1])

    # Get list of cell keys for most recent dataset
    keys = list(gw.list_cells(product=input_products[-1], lat=(args.lowerlat, args.upperlat), lon=(args.lowerlon, args.upperlon)).keys())

    for key in keys:

        input_ds = []
        tmask_ds = []

        for product in input_products:

            gw = GridWorkflow(dc.index, product=product)

            # Get the list of tiles (one for each time point) for this product
            tile_list = gw.list_tiles(product=product, cell_index=key)

            # Load all tiles
            for tile_index, tile in tile_list.items():
                dataset = gw.load(tile, measurements=args.bands)

                if(dataset.variables):
                    input_ds.append(dataset)
                    
        if(args.tmask_products):
            
            for product in args.tmask_products:
                
                gw = GridWorkflow(dc.index, product=product)

                # Get the list of tiles (one for each time point) for this product
                tile_list = gw.list_tiles(product=product, cell_index=key)
    
                # Load all tiles
                for tile_index, tile in tile_list.items():
                    dataset = gw.load(tile, measurements=['green', 'nir', 'swir1'])
    
                    if(dataset.variables):
                        tmask_ds.append(dataset)
                       
        dc.close()
        
        if(len(input_ds) > 0): # Check that there is actually some input data
            
            # Tidy up input data
            input_data = xr.concat(input_ds, dim='time')
            input_data = mask_invalid_data(input_data)
            
            # Do the same for TOA data if present - tmask_ds will be empty if no TOA data sets were specified
            if(len(tmask_ds) > 0):
                
                tmask_data = xr.concat(tmask_ds, dim='time')
                tmask_data = mask_invalid_data(tmask_data)
                       
            # We want to process each pixel seperately
            for i in range(len(input_data.x)):
                for j in range(len(input_data.y)):

                    input_ts = input_data.isel(x=i, y=j) # Get just one pixel
                    
                    x_val = float(input_ts.x)
                    y_val = float(input_ts.y)

                    input_ts = transformToArray(input_ts) # Transform the time series into a numpy array
                    
                    # Remove NaNs
                    input_nan_mask = np.any(np.isnan(input_ts), axis=1)
                    input_ts = input_ts[~input_nan_mask]
                    
                    # Sort by date
                    input_ts = input_ts[np.argsort(input_ts[:,0])]
                                              
                    if(input_ts.shape[1] == input_num_cols):
                        
                        if(len(tmask_ds) > 0):
                    
                            tmask_ts = tmask_data.isel(x=i, y=j)
                            
                            tmask_ts = transformToArray(tmask_ts)
                            
                            if(tmask_ts.shape[1] == 4): # Tmask data should always have 4 columns
                                
                               input_ts = doTmask(input_ts, tmask_ts)
                                   
                        change_file = "{}{}_{}".format(args.outdir, str(x_val), str(y_val))

                        # Block until a core becomes available
                        while(True):
    
                            p_done = []
    
                            for index, p in enumerate(processes):
                                    
                                if(not p.is_alive()):
                                    p_done.append(index)
    
                            if(p_done):
                                for index in sorted(p_done, reverse=True): # Need to delete in reverse order to preserve indexes
                                    del(processes[index])
    
                            if(len(processes) < num_processes):
                                break
                                        
                        process = multiprocessing.Process(target=runCCDC, args=(input_ts, change_file, num_bands, args))
                        processes.append(process)
                        process.start()

    # Keep running until all processes have finished
    for p in processes:
        p.join()
  
def main(args):
    
    """Program runs from here"""

    if(not os.path.isdir(args.outdir)):
       print("Output directory does not exist.")
       sys.exit()
       
    num_bands = len(args.bands)
    
    global model_list 
    model_list = [None for i in range(num_bands)] # Set up list of models

    if(args.lowerlat is not None and args.upperlat is not None and args.lowerlon is not None and args.upperlon is not None):

        if(args.mode == "subsample"):
            runOnSubset(num_bands, args)

        elif(args.mode == "area"):
            runOnArea(num_bands, args)
            
        elif(args.mode == "all"):
            runAll(num_bands, args)

        else:
            print("Lat/long boundaries were provided, but mode was not subsample, area, or all.")

    elif(len(args.key) > 1 and args.tile_x_min is not None and args.tile_x_max is not None and args.tile_y_min is not None and args.tile_y_max is not None):

        if(args.mode == "tile"):
            key = tuple(args.key)
            runByTile(key, num_bands, args)

        else:
            print("Key/pixel details were provided, but mode was not single pixel.")

    else:
        print("Please provide either lat/long boundaries or pixel coordinates and cell key.")
        

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Run CCDC algorithm using Data Cube.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--outdir', default="./", help="The output directory for any produced csv files/images/plots.")
    parser.add_argument('-llat', '--lowerlat', type=float, default=None, help="The lower latitude boundary of the area to be processed. Required if using area or subsample modes.")
    parser.add_argument('-ulat', '--upperlat', type=float, default=None, help="The upper latitude boundary of the area to be processed. Required if using area or subsample modes.")
    parser.add_argument('-llon', '--lowerlon', type=float, default=None, help="The lower longitude boundary of the area to be processed. Required if using area or subsample modes.")
    parser.add_argument('-ulon', '--upperlon', type=float, default=None, help="The upper longitude boundary of the area to be processed. Required if using area or subsample modes.")
    parser.add_argument('-k', '--key', type=int, nargs='*', default=[], help="The key for the cell to be processed. Must be entered as two separate integers, e.g. 6 -27. Required if using tile mode.")
    parser.add_argument('-t_xmin', '--tile_x_min', type=int, default=None, help="The minimum/starting x value of the area of the tile you want to process. Required if using tile mode.")
    parser.add_argument('-t_xmax', '--tile_x_max', type=int, default=None, help="The maximum/ending x value of the area of the tile you want to process. Required if using tile mode.")
    parser.add_argument('-t_ymin', '--tile_y_min', type=int, default=None, help="The minimum/starting y value of the area of the tile you want to process. Required if using tile mode.")
    parser.add_argument('-t_ymax', '--tile_y_max', type=int, default=None, help="The maximum/ending y value of the area of the tile you want to process. Required if using tile mode.")
    parser.add_argument('-m', '--mode', choices=['area','subsample', 'tile', 'all'], default='all', help="Whether the algorithm should be run on a specified area, a subsample of a (specified) area, a specific tile, or all available data.")
    parser.add_argument('-num', '--num_points', type=int, default=100, help="Specifies the number of subsamples to take if a random subsample is being processed.")
    parser.add_argument('-ot', '--outtype', choices=['plot', 'csv'], default='csv', help="Specifies the format of the output data. Either a plot or a CSV file will be produced for each pixel.")
    parser.add_argument('-ip', '--input_products', nargs='+', required=True, help="The product(s) to use for change detection.")    
    parser.add_argument('-tp', '--tmask_products', nargs='+', help="The top-of-atmosphere reflectance product(s) to use for Tmask screening. If no products are specified no screening will be applied.")
    parser.add_argument('-i', '--re_init', type=int, default=1, help="The number of new observations added to a model before the model is refitted.")
    parser.add_argument('-p', '--num_procs', type=int, default=1, help="The number of processes to use.")
    parser.add_argument('-s', '--save_models', type=bool, default=False, help="Whether models should be pickled.")
    parser.add_argument('-b', '--bands', nargs='+', required=True, help="List of band names to use in the analysis.")
    
    args = parser.parse_args()
    
    main(args)








