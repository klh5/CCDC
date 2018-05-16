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

# Set up list of models
model_list = [None for i in range(5)]     # List of models, one for each band
plt_list = []                             # List of plots, one for each band

def add_change_marker(num_bands, start_change, end_change, obs_data, plot_data):

    """ Adds a vertical line to each plot every time change is detected """
    T = 365
    pi_val_simple = (2 * np.pi) / T
    pi_val_advanced = (4 * np.pi) / T
    pi_val_full = (6 * np.pi) / T
   
    for i in range(num_bands):
        y_min = np.amin(obs_data.iloc[:,i+1])
        y_max = np.amax(obs_data.iloc[:,i+1])

        plt_list[i].plot([start_change, start_change], [y_min, y_max], 'r', linewidth=1, label="Start change")
        plt_list[i].plot([end_change, end_change], [y_min, y_max], 'y', linewidth=1, label="End change")

        num_coeffs = model_list[i].get_num_coeffs()

        if(num_coeffs == 4):
            interp = interp1d(plot_data.datetime, model_list[i].get_coefficients()[0] + (model_list[i].get_coefficients()[1]*(np.cos(pi_val_simple * plot_data.datetime))) + (model_list[i].get_coefficients()[2]*(np.sin(pi_val_simple * plot_data.datetime))) + model_list[i].get_coefficients()[3]*plot_data.datetime, kind='linear')

        elif(num_coeffs == 6):
            interp = interp1d(plot_data.datetime, model_list[i].get_coefficients()[0] + (model_list[i].get_coefficients()[1]*(np.cos(pi_val_simple * plot_data.datetime))) + (model_list[i].get_coefficients()[2]*(np.sin(pi_val_simple * plot_data.datetime))) + (model_list[i].get_coefficients()[3]*(np.cos(pi_val_advanced * plot_data.datetime))) + (model_list[i].get_coefficients()[4]*(np.sin(pi_val_advanced * plot_data.datetime))) + model_list[i].get_coefficients()[5]*plot_data.datetime, kind='linear')

        else:
            interp = interp1d(plot_data.datetime, model_list[i].get_coefficients()[0] + (model_list[i].get_coefficients()[1]*(np.cos(pi_val_simple * plot_data.datetime))) + (model_list[i].get_coefficients()[2]*(np.sin(pi_val_simple * plot_data.datetime))) + (model_list[i].get_coefficients()[3]*(np.cos(pi_val_advanced * plot_data.datetime))) + (model_list[i].get_coefficients()[4]*(np.sin(pi_val_advanced * plot_data.datetime))) + (model_list[i].get_coefficients()[5]*(np.sin(pi_val_full * plot_data.datetime))) + (model_list[i].get_coefficients()[6]*(np.cos(pi_val_full * plot_data.datetime))) + model_list[i].get_coefficients()[7]*plot_data.datetime, kind='linear')

        plt_list[i].plot(plot_data.datetime, interp(plot_data.datetime), 'b', linewidth=1, label="Lasso fit")

def setupModels(all_band_data, num_bands, init_obs):
    
    """Creates a model for each band and stores it in model_list"""
    
    # Create a model for each band and store it in model_list
    for i in range(num_bands):
        
        band_data = pd.DataFrame({'datetime': all_band_data['datetime'], 'reflectance': all_band_data.iloc[:,i+1]})
        
        ccdc_model = MakeCCDCModel(band_data)
        
        ccdc_model.fit_model(init_obs)
        
        model_list[i] = ccdc_model

def getNumYears(date_list):

    """Get number of years (from Python/Rata Die date)"""
    
    last_date = datetime.fromordinal(np.amax(date_list)).strftime('%Y')
    first_date = datetime.fromordinal(np.amin(date_list)).strftime('%Y')
        
    num_years = int(last_date) - int(first_date)

    return num_years

def transformToDf(dataset_to_transform):

    """Transforms xarray Dataset object into a Pandas dataframe"""

    new_df = pd.DataFrame()
        
    new_df['datetime'] = dataset_to_transform.time.data
    new_df['datetime'] = new_df['datetime'].apply(lambda x: x.toordinal())

    for name, var in dataset_to_transform.data_vars.items():
        new_df[name] = np.reshape(var.data, -1)
            
    # Points at the edge of the image could return empty arrays (all 0's) - this will remove any columns to which this applies
    new_df = new_df.dropna(axis=1, how='all')

    return new_df

def init_model(pixel_data, num_bands, init_obs):

    """Finds a sequence of 6/12/18/24 consecutive clear observations without any change, to initialize the model"""

    # Subset first n clear observations for model initialisation
    curr_obs_list = pixel_data.iloc[0:init_obs,:]
    
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
        
        # Get cumulative time
        total_time = curr_obs_list['datetime'].sum()
        
        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0
        
        # Check for change during the initialization period. We need 12 observations with no change
        for band_model in model_list: # For each model
            
            for row in band_model.get_band_data().iterrows():
                slope_val = np.absolute(((band_model.get_coefficients()['datetime']) * row[0])) / 3 * (band_model.get_rmse() / total_time)
                total_slope_eval += slope_val
        
            start_val = np.absolute((band_model.get_band_data()['reflectance'].iloc[0] - band_model.get_band_data()['predicted'].iloc[0])) / (3 * band_model.get_rmse())
            total_start_eval += start_val
            
            end_val = np.absolute((band_model.get_band_data()['reflectance'].iloc[num_data_points-1] - band_model.get_band_data()['predicted'].iloc[num_data_points-1])) / (3 * band_model.get_rmse())
            total_end_eval += end_val
        
        if(total_slope_eval > 1 or total_start_eval > 1 or total_end_eval > 1):
            num_iters += 1
            curr_obs_list = pixel_data.iloc[0+num_iters:init_obs+num_iters,:] # Shift along 1 row
        
        else:
            model_init = True
            init_end = init_obs + num_iters + 1
            #print("Model initialized. Iterations needed: {}".format(num_iters))

    return curr_obs_list, init_end

def findChange(pixel_data, change_file, num_bands, init_obs, args):
    
    """Continues to add data points to the model until either a new breakpoint is detected, or there
        are not enough observations remaining."""
    
    try:
        model_data, next_obs = init_model(pixel_data, num_bands, init_obs)
    except TypeError:
        return []

    # Detect change
    change_flag = 0
    change_start_time = None

    while((next_obs+1) <= len(pixel_data)):

        change_eval = 0
        
        new_obs = pixel_data.iloc[next_obs]
        new_date = new_obs[0]
        
        for model_num, band_model in enumerate(model_list):    # For each band
            new_ref_obs = new_obs[model_num+1]
            residual_val = np.absolute((new_ref_obs - band_model.get_prediction(new_date)[0])) / (2 * band_model.get_rmse())
            change_eval += residual_val
        
        if(change_eval <= 1):
            #print("Adding new data point")
            model_data.append(new_obs, ignore_index=True)
            setupModels(model_data, num_bands, init_obs)
            change_flag = 0 # Reset change flag because we have an inlier

        else:
            change_flag += 1 # Don't add the new pixel to the model
            if(change_flag == 1): # If this is the first observed possible change point
                change_start_time = new_date
    
        if(change_flag == 6):
            #print("Change detected!")
            
            if(args.outtype == 'plot'):
                add_change_marker(num_bands, change_start_time, new_date, pixel_data, model_data)

            else:
               with open(change_file, 'a') as output_file:
                  writer = csv.writer(output_file)
                  writer.writerow([datetime.fromordinal(int(change_start_time)).strftime('%d/%m/%Y'), datetime.fromordinal(int(new_date)).strftime('%d/%m/%Y')])
                
            return pixel_data.iloc[next_obs:,]
        
        # Need to get the next observation
        next_obs += 1
    
    # No change detected, end of data reached
    return []

def runCCDC(sref_data, toa_data, change_file, x_val, y_val, return_list, args):

    """The main function which runs the CCDC algorithm. Loops until there are not enough observations
        left after a breakpoint to attempt to initialize a new model."""

    # Drop any rows which have NaN values for one or more bands (we need data for all bands to be complete)
    sref_data.dropna(axis=0, how='any', inplace=True)
    toa_data.dropna(axis=0, how='any', inplace=True)

    num_bands = 5
    num_changes = 0

    # Sort data by date
    sref_data.sort_values(by=['datetime'], inplace=True)
    toa_data.sort_values(by=['datetime'], inplace=True)

    # Very important that both datasets contain the same number of observations
    if(len(sref_data) == len(toa_data)):
        
        # Get the number of years covered by the dataset
        num_years = getNumYears(sref_data['datetime'])
        
        # The algorithm needs at least 1 year of data
        if(getNumYears(sref_data['datetime']) > 0):
            
            # Screen for outliers
            robust_outliers = RLMRemoveOutliers()
            outlier_list = robust_outliers.findOutliers(toa_data, num_years)

            ts_data = sref_data.drop(outlier_list)
            ts_data.reset_index(drop=True, inplace=True)

            # Update num_years now outliers have been removed
            num_years = getNumYears(ts_data['datetime'])

            if(args.outtype == 'plot'):

                fig = plt.figure(figsize=(20, 10))

                # Set up plots with original data and screened data
                for i in range(num_bands):
                    plt_list.append(fig.add_subplot(num_bands, 1, i+1))
                    band_col = ts_data.columns[i+1]
                    plt_list[i].plot(sref_data['datetime'], sref_data.iloc[:,i+1], 'o', color='0.5', label='Original data', markersize=3)
                    plt_list[i].plot(ts_data['datetime'], ts_data.iloc[:,i+1], 'o', color='k', label='Data after RIRLS', markersize=3)
                    plt_list[i].set_ylabel(band_col)
                    myFmt = mdates.DateFormatter('%m/%Y') # Format dates as month/year rather than ordinal dates
                    plt_list[i].xaxis.set_major_formatter(myFmt)

            else:
                  change_file = change_file + ".csv"
                  with open(change_file, 'w') as output_file:
                     writer = csv.writer(output_file)
                     writer.writerow(['start_change', 'end_change'])

            # We need at least 12 clear observations (6 + 6 to detect change)
            while(len(ts_data) >= 12):
                if(getNumYears(ts_data['datetime']) > 0):

                    
                    # Get total number of clear observations in the dataset
                    num_clear_obs = len(ts_data)
            
                    if(num_clear_obs < 18):
                        # Use simple model with initialization period of 6 obs
                        ts_data = findChange(ts_data, change_file, num_bands, 6, args)
                    
                    elif(num_clear_obs >= 18 and num_clear_obs < 24):
                        # Use simple model with initialization period of 12 obs
                        ts_data = findChange(ts_data, change_file, num_bands, 12, args)

                    elif(num_clear_obs >= 24 and num_clear_obs < 30):
                        # Use advanced model with initialization period of 18 obs
                        ts_data = findChange(ts_data, change_file, num_bands, 18, args)
                    
                    elif(num_clear_obs >= 30):
                        # Use full model with initialisation period of 24 obs
                        ts_data = findChange(ts_data, change_file, num_bands, 24, args)

                    if(len(ts_data) > 0):
                        num_changes = num_changes + 1
                    
                else:
                    #print("Less than 1 year of observations remaining.")
                    break                               

            #print("Ran out of observations.")

            if(args.outtype == 'plot'):

                # Once there is no more data to process, plot the results
                plt.legend(['Original data', 'Data after RIRLS', 'Start change', 'End change', 'Lasso fit'])
                plt.tight_layout()
                change_file = change_file + ".svg"
                plt.savefig(change_file)
                plt.close(fig)

            return_list.append({'x': x_val, 'y': y_val, 'num_changes': num_changes})
                  
    #else:
        #print('SREF and TOA data not the same length. Check indexing/ingestion.')

def runOnSubset(sref_products, toa_products, args):

    """If the user chooses to run the algorithm on a random subsample of the data, this function is called.
        This function creates a polygon shape from the lat/long points provided by the user. It then selects
        random points from within this shape and runs the algorithm on those points."""

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
               
                sref_ds = []
                toa_ds = []

                dc = datacube.Datacube()

                for product in sref_products:
                    dataset = dc.load(product=product, measurements=['red', 'green', 'nir', 'swir1', 'swir2'], lat=(new_point_lat), lon=(new_point_long))

                    if(dataset.notnull()):
                        sref_ds.append(dataset)
            
                for product in toa_products:
                    dataset = dc.load(product=product, measurements=['green', 'nir', 'swir1'], lat=(new_point_lat), lon=(new_point_long))

                    if(dataset.notnull()):
                        toa_ds.append(dataset)
                        
                dc.close()
                        
                if(len(sref_ds) == len(sref_products) and len(toa_ds) == len(toa_products)):
                    sref = xr.concat(sref_ds, dim='time')
                    toa = xr.concat(toa_ds, dim='time')
                
                    # Change nodata values (0's) to NaN
                    sref = mask_invalid_data(sref)
                    toa = mask_invalid_data(toa)
                
                    sref_data = transformToDf(sref)
                    toa_data = transformToDf(toa)

                    if(sref_data.shape[1] == 6 and toa_data.shape[1] == 4):
                        dc.close()
                        change_file = args.outdir + str(new_point.GetX()) + "_" + str(new_point.GetY())
                        runCCDC(sref_data, toa_data, change_file, args)
                        curr_points += 1

def runOnArea(sref_products, toa_products, args):

    """If the user chooses to run the algorithm on the whole of the specified area, this function is called.
    This function will load the whole area specified by the user, and run the algorithm on each pixel one by one."""

    sref_ds = []
    toa_ds = []

    dc = datacube.Datacube()

    for product in sref_products:
        dataset = dc.load(product=product, measurements=['red', 'green', 'nir', 'swir1', 'swir2'], lat=(args.lowerlat, args.upperlat), lon=(args.lowerlon, args.upperlon))
        
        if(dataset.notnull()):
            sref_ds.append(dataset)

    for product in toa_products:
        dataset = dc.load(product=product, measurements=['green', 'nir', 'swir1'], lat=(args.lowerlat, args.upperlat), lon=(args.lowerlon, args.upperlon))
        
        if(dataset.notnull()):
            toa_ds.append(dataset)
            
    dc.close()
    
    if(len(sref_ds) == len(sref_products) and len(toa_ds) == len(toa_products)):      
        sref = xr.concat(sref_ds, dim='time')
        toa = xr.concat(toa_ds, dim='time')

        # Change nodata values (0's) to NaN
        sref = mask_invalid_data(sref)
        toa = mask_invalid_data(toa)

        for i in range(len(sref.x)):
            for j in range(len(sref.y)):
                
                sref_ts = sref.isel(x=i, y=j)
                toa_ts = toa.isel(x=i, y=j)
       
                sref_data = transformToDf(sref_ts)
                toa_data = transformToDf(toa_ts)
    
                if(sref_data.shape[1] == 6 and toa_data.shape[1] == 4):
                    dc.close()
                    change_file = args.outdir + str(int(sref_ts.x)) + "_" + str(int(sref_ts.y))
                    runCCDC(sref_data, toa_data, change_file, args)

def runOnPixel(sref_products, toa_products, key, args):

    """A key represent one cell/area. Each cell has a tile for each time point. The x and y values define the extent of
   the tile that should be loaded and processed."""

    x_val = args.pixel_x
    y_val = args.pixel_y

    sref_ds = []
    toa_ds = []

    for product in sref_products:

        dc = datacube.Datacube()

        # Create the GridWorkflow object for this product
        curr_gw = GridWorkflow(dc.index, product=product)

        # Get the list of tiles (one for each time point) for this product
        tile_list = curr_gw.list_tiles(product=product, cell_index=key)

        dc.close()

        # Retrieve the specified pixel for each tile in the list
        for tile_index, tile in tile_list.items():
            dataset = curr_gw.load(tile[0:1, x_val:x_val+1, y_val:y_val+1], measurements=['red', 'green', 'nir', 'swir1', 'swir2'])

            if(dataset.notnull()):
                sref_ds.append(dataset)

    for product in toa_products:

        dc = datacube.Datacube()

        # Create the GridWorkflow object for this product
        curr_gw = GridWorkflow(dc.index, product=product)

        # Get the list of tiles (one for each time point) for this product
        tile_list = curr_gw.list_tiles(product=product, cell_index=key)

        dc.close()

        # Retrieve the specified pixel for each tile in the list
        for tile_index, tile in tile_list.items():
            dataset = curr_gw.load(tile[0:1, x_val:x_val+1, y_val:y_val+1], measurements=['green', 'nir', 'swir1'])

            if(dataset.notnull()):
                toa_ds.append(dataset)
    
    # Check that both datasets are the same length
    if(len(sref_ds) == len(toa_ds) and len(sref_ds) > 0 and len(toa_ds) > 0):

        sref = xr.concat(sref_ds, dim='time')
        toa = xr.concat(toa_ds, dim='time')

        # Change nodata values (0's) to NaN
        sref = mask_invalid_data(sref)
        toa = mask_invalid_data(toa)
        
        sref_data = transformToDf(sref)
        toa_data = transformToDf(toa)

        if(sref_data.shape[1] == 6 and toa_data.shape[1] == 4):
            dc.close()
            change_file = args.outdir + str(int(sref.x)) + "_" + str(int(sref.y))
            runCCDC(sref_data, toa_data, change_file, args)

def runAll(sref_products, toa_products, args):

    """Run on all tiles in the specified datasets. Keys are based on the most recent dataset."""

    num_cores = multiprocessing.cpu_count() - 1
    
    processes = []

    # Set up list to enable all processes to send their results back
    manager = multiprocessing.Manager()
    return_list = manager.list()

    dc = datacube.Datacube()

    # Create Gridworkflow object for most recent dataset
    gw = GridWorkflow(dc.index, product=sref_products[-1])

    # Get list of cell keys for most recent dataset
    keys = list(gw.list_cells(product=sref_products[-1]).keys())

    keys = [(5, -28)]

    for key in keys:

        sref_ds = []
        toa_ds = []

        for product in sref_products:

            gw = GridWorkflow(dc.index, product=product)

            # Get the list of tiles (one for each time point) for this product
            tile_list = gw.list_tiles(product=product, cell_index=key)

            # Load all tiles
            for tile_index, tile in tile_list.items():
                dataset = gw.load(tile[0:1, 1500:1501, 0:1], measurements=['red', 'green', 'nir', 'swir1', 'swir2'])

                if(dataset.notnull()):
                    sref_ds.append(dataset)

        for product in toa_products:

            gw = GridWorkflow(dc.index, product=product)

            # Get the list of tiles (one for each time point) for this product
            tile_list = gw.list_tiles(product=product, cell_index=key)

            # Load all tiles
            for tile_index, tile in tile_list.items():
                dataset = gw.load(tile[0:1, 1500:1501, 0:1], measurements=['green', 'nir', 'swir1'])

                if(dataset.notnull()):
                    toa_ds.append(dataset)

        dc.close()

        # Check that both datasets are the same length
        if(len(sref_ds) == len(toa_ds) and len(sref_ds) > 0 and len(toa_ds) > 0):

            sref = xr.concat(sref_ds, dim='time')
            toa = xr.concat(toa_ds, dim='time')

            # Change nodata values (0's) to NaN
            sref = mask_invalid_data(sref)
            toa = mask_invalid_data(toa)
            
            # We want to process each pixel seperately
            for i in range(len(sref.x)):
                for j in range(len(sref.y)):

                    print("{}, {}".format(i, j))

                    sref_ts = sref.isel(x=i, y=j)
                    toa_ts = toa.isel(x=i, y=j)
       
                    sref_data = transformToDf(sref_ts)
                    toa_data = transformToDf(toa_ts)
    
                    if(sref_data.shape[1] == 6 and toa_data.shape[1] == 4):

                        x_val = float(sref_ts.x)
                        y_val = float(sref_ts.y)
                                    
                        change_file = args.outdir + str(x_val) + "_" + str(y_val)

                        # Block until a core becomes available
                        while(True):

                            p_done = []

                            for index, p in enumerate(processes):
                                p.join(timeout=0)
                                
                                if(not p.is_alive()):
                                    p_done.append(index)

                            if(p_done):
                                for index in sorted(p_done, reverse=True): # Need to delete in reverse order to preserve indexes
                                    del(processes[index])

                            if(len(processes) < num_cores):
                                break
                                    
                        process = multiprocessing.Process(target=runCCDC, args=(sref_data, toa_data, change_file, x_val, y_val, return_list, args))
                        processes.append(process)
                        process.start()

    # Keep running until all processes have finished
    for p in processes:
        p.join()

    # Pandas doesn't recognise Manager lists, so we need to convert it back to an ordinary list
    rows = return_list[0:len(return_list)]

    to_df = pd.DataFrame(rows).set_index(['y', 'x'])
    
    dataset = xr.Dataset.from_dataframe(to_df)

    dataset.attrs['crs'] = 'PROJCS["WGS 84 / UTM zone 55N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",147],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32655"]]'

    change_img = args.outdir + "change_map.nc"
    dataset.to_netcdf(change_img, encoding={'num_changes': {'dtype': 'uint16', '_FillValue': 9999}})
    
def main(args):
    
    """Program runs from here"""

    sref_products = []
    toa_products = []

    if(not os.path.isdir(args.outdir)):
       print("Output directory does not exist.")
       sys.exit()
        
    # Products are always added to both lists in the same order
    if('ls5' in args.platform):
        sref_products.append('ls5_arcsi_sref_ingested')
        toa_products.append('ls5_arcsi_toa_ingested')

    if('ls7' in args.platform):
        sref_products.append('ls7_arcsi_sref_ingested')
        toa_products.append('ls7_arcsi_toa_ingested')

    if('ls8' in args.platform):
        sref_products.append('ls8_arcsi_sref_ingested')
        toa_products.append('ls8_arcsi_toa_ingested')

    if(args.lowerlat > -1 and args.upperlat > -1 and args.lowerlon > -1 and args.upperlon > -1):

        if(args.mode == "subsample"):
            runOnSubset(sref_products, toa_products, args)

        elif(args.mode == "whole_area"):
            runOnArea(sref_products, toa_products, args)

        else:
            print("Lat/long boundaries were provided, but mode was not subsample or whole_area.")

    elif(len(args.key) > 1 and args.pixel_x > -1 and args.pixel_y > -1):

        if(args.mode == "by_pixel"):
            key = tuple(args.key)
            runOnPixel(sref_products, toa_products, key, args)

        else:
            print("Key/pixel details were provided, but mode was not by_pixel.")

    elif(args.mode == "all"):
        runAll(sref_products, toa_products, args)

    else:
        print("Please provide either lat/long boundaries or pixel coordinates and cell key, or set mode to 'all'.")
        


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Run CCDC algorithm using Data Cube.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--outdir', default="./", help="The output directory for any produced csv files/images/plots.")
    parser.add_argument('-llat', '--lowerlat', type=float, default=-1, help="The lower latitude boundary of the area to be processed. Required if using whole_area or subsample arguments.")
    parser.add_argument('-ulat', '--upperlat', type=float, default=-1, help="The upper latitude boundary of the area to be processed. Required if using whole_area or subsample arguments.")
    parser.add_argument('-llon', '--lowerlon', type=float, default=-1, help="The lower longitude boundary of the area to be processed. Required if using whole_area or subsample arguments.")
    parser.add_argument('-ulon', '--upperlon', type=float, default=-1, help="The upper longitude boundary of the area to be processed. Required if using whole_area or subsample arguments.")
    parser.add_argument('-k', '--key', type=int, nargs='*', default=[], help="The key for the cell to be processed. Must be a tuple of two integers, e.g. 6, -27. Required if using by_pixel argument.")
    parser.add_argument('-x', '--pixel_x', type=int, default=-1, help="The x value of the pixel to be processed within the specified tile. Required if using by_pixel argument.")
    parser.add_argument('-y', '--pixel_y', type=int, default=-1, help="The y value of the pixel to be processed within the specified tile. Required if using by_pixel argument.")
    parser.add_argument('-p', '--platform', choices=['ls5', 'ls7', 'ls8'], nargs='+', default=['ls5', 'ls7', 'ls8'], help="The platforms to be included.")
    parser.add_argument('-m', '--mode', choices=['whole_area','subsample', 'by_pixel', 'all'], default='subsample', help="Whether the algorithm should be run on a whole (specified) area, a subsample of a (specified) area, a specific pixel, or all available data.")
    parser.add_argument('-num', '--num_points', type=int, default=100, help="Specifies the number of subsamples to take if a random subsample is being processed.")
    parser.add_argument('-ot', '--outtype', choices=['plot', 'csv'], default='csv', help="Specifies the format of the output data. Either a plot or a CSV file will be produced for each pixel.")
    args = parser.parse_args()
    
    main(args)








