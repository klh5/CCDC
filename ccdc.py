import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import datacube
import xarray as xr
from datacube.storage.masking import mask_invalid_data
from makeModel import MakeCCDCModel
from removeOutliers import RLMRemoveOutliers
from datetime import datetime
from osgeo import ogr
from random import uniform


# Set up list of models
model_list = [None for i in range(5)]     # List of models, one for each band

def add_change_marker(plot_list, num_bands, change_point, obs_data):

   """ Adds a vertical line to each plot every time change is detected """
   
   for i in range(num_bands):
       y_min = np.amin(obs_data.iloc[:,i+1])
       y_max = np.amax(obs_data.iloc[:,i+1])

       plot_list[i].plot([change_point, change_point], [y_min, y_max], 'r', linewidth=1, label="Change point")

def setupModels(all_band_data, num_bands, init_obs):
    
    """Creates a model for each band and stores it in model_list"""
    
    # Create a model for each band and store it in model_list
    for i in range(num_bands):
        
        band_data = pd.DataFrame({'time': all_band_data['time'], 'reflectance': all_band_data.iloc[:,i+1]})
        
        ccdc_model = MakeCCDCModel(band_data)
        
        ccdc_model.fit_model(init_obs)
        
        model_list[i] = ccdc_model

def getNumYears(date_list):

    """Get number of years (from Python/Rata Die date)"""
    
    last_date = datetime.fromordinal(np.amax(date_list)).strftime('%Y')
    first_date = datetime.fromordinal(np.amin(date_list)).strftime('%Y')
        
    num_years = int(last_date) - int(first_date)

    return num_years

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
            print("Not enough data points left to initialize model.")
            return None
    
        # Re-initialize the models
        setupModels(curr_obs_list, num_bands, init_obs)
        
        # Get cumulative time
        total_time = curr_obs_list['time'].sum()
        
        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0
        
        # Check for change during the initialization period. We need 12 observations with no change
        for band_model in model_list: # For each model
            
            for row in band_model.get_band_data().iterrows():
                slope_val = np.absolute(((band_model.get_coefficients()['time']) * row[0])) / 3 * (band_model.get_rmse() / total_time)
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
            print("Model initialized. Iterations needed: {}".format(num_iters))

    return curr_obs_list, init_end

def findChange(pixel_data, figures, num_bands, init_obs):
    
    """Continues to add data points to the model until either a new breakpoint is detected, or there
        are not enough observations remaining."""
    
    try:
        model_data, next_obs = init_model(pixel_data, num_bands, init_obs)
    except TypeError:
        return []

    # Detect change
    change_flag = 0

    while((next_obs+1) <= len(pixel_data)):

        change_eval = 0
        
        new_date = pixel_data.iloc[next_obs][0]
        
        for model_num, band_model in enumerate(model_list):    # For each band
            new_ref_obs = pixel_data.iloc[next_obs][model_num+1]
            residual_val = np.absolute((new_ref_obs - band_model.get_prediction(new_date)[0])) / (band_model.get_rmse() * 2)
            change_eval += residual_val
    
        new_obs = pixel_data.iloc[next_obs]
        
        if(change_eval <= 1):
            print("Adding new data point")
            model_data.append(new_obs, ignore_index=True)
            setupModels(model_data, num_bands, init_obs)
            change_flag = 0 # Reset change flag because we have an inlier

        else:
            change_flag += 1 # Don't add the new pixel to the model
    
        if(change_flag == 6):
            print("Change detected!")
            add_change_marker(figures, num_bands, new_obs[0], pixel_data)
            return pixel_data.iloc[next_obs:,]
        
        # Need to get the next observation
        next_obs += 1
    
    # No change detected, end of data reached
    return []

def main():
    
    """Program runs from here"""
    
    if(len(sys.argv) > 1):
        num_points = int(sys.argv[1])
    else:
        print("Number of points to analyze was not specified.")
        sys.exit()

    dc = datacube.Datacube()

    # Set some spatial boundaries for the data
    BLLat = -27.0
    BLLon = 146.0
    BRLat = -27.0
    BRLon = 149.0
    TLLat = -25.0
    TLLon = 146.0
    TRLat = -25.0
    TRLon = 149.0

    boundary = ogr.Geometry(ogr.wkbLinearRing)
    boundary.AddPoint(BLLat, BLLon)
    boundary.AddPoint(BRLat, BRLon)
    boundary.AddPoint(TRLat, TRLon)
    boundary.AddPoint(TLLat, TLLon)
    boundary.AddPoint(BLLat, BLLon)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(boundary)

    envelope = poly.GetEnvelope()

    min_lat = envelope[0]
    max_lat = envelope[1]
    min_long = envelope[2]
    max_long = envelope[3]

    curr_points = 0

    for i in range(num_points):

        while curr_points < num_points:
        
            new_point = ogr.Geometry(ogr.wkbPoint)
            new_point_lat = uniform(min_lat, max_lat)
            new_point_long = uniform(min_long, max_long)
        
            new_point.AddPoint(new_point_lat, new_point_long)

            if new_point.Within(poly):
            
                sref = dc.load(product='ls8_arcsi_sref_ingested', measurements=['red', 'green', 'blue', 'nir', 'swir1', 'swir2'], lat=(new_point_lat), lon=(new_point_long))
                sref = mask_invalid_data(sref)

                if(sref.notnull()):
                
                    data = pd.DataFrame()
                    data['time'] = sref.time.data
                    data['time'] = data['time'].apply(lambda x: x.toordinal())
                
                    for name, var in sref.data_vars.items():
                        data[name] = np.reshape(var.data, -1)
        
                    data = data.dropna(axis=1, how='all')

                    if(data.shape[1] == 7):

                        # Get number of bands, which will be the number of columns - 2
                        num_bands = len(data.columns) - 2

                        # One figure for each band - makes the plots much simpler
                        plt_list = []
    
                        # Sort data by date
                        data = data.sort_values(by=['time'])
    
                        # Get the number of years covered by the dataset
                        num_years = getNumYears(data['time'])
    
                        # Screen for outliers
                        #robust_outliers = RLMRemoveOutliers()
                        #next_data = robust_outliers.clean_data(data, num_years)
                        next_data = data
                        # Update num_years now outliers have been removed
                        num_years = getNumYears(next_data['time'])
    
                        fig = plt.figure()

                        # Set up basic plots with original data
                        for i in range(num_bands):
                            plt_list.append(fig.add_subplot(num_bands, 1, i+1))
                            band_col = next_data.columns[i+1]
                            plt_list[i].plot(data['time'], data.iloc[:,i+1], 'o', color='blue', label='Original data', markersize=2)
                            plt_list[i].plot(next_data['time'], next_data.iloc[:,i+1], 'o', color='black', label='Data after RIRLS', markersize=3)
                            plt_list[i].set_ylabel(band_col)
                            myFmt = mdates.DateFormatter('%Y') # Format dates as year rather than ordinal dates
                            plt_list[i].xaxis.set_major_formatter(myFmt)
    
                        # We need at least 12 clear observations (6 + 6 to detect change)
                        while(len(next_data) >= 12):
        
                            if(getNumYears(next_data['time']) > 0):
            
                                # Get total number of clear observations in the dataset
                                num_clear_obs = len(next_data)
    
                                if(num_clear_obs >= 12 and num_clear_obs < 18):
                                # Use simple model with initialization period of 6 obs
                                    next_data = findChange(next_data, plt_list, num_bands, 6)
            
                                elif(num_clear_obs >= 18 and num_clear_obs < 24):
                                    # Use simple model with initialization period of 12 obs
                                    next_data = findChange(next_data, plt_list, num_bands, 12)

                                elif(num_clear_obs >= 24 and num_clear_obs < 30):
                                    # Use advanced model with initialization period of 18 obs
                                    next_data = findChange(next_data, plt_list, num_bands, 18)
            
                                elif(num_clear_obs >= 30):
                                    # Use full model with initialisation period of 24 obs
                                    next_data = findChange(next_data, plt_list, num_bands, 24)
            
                            else:
                                break

                        # Once there is no more data to process, plot the results
                        plt.legend(['Original data', 'Data after RIRLS', 'Change point'])
                        plt.tight_layout()
                        plt.show()


if __name__ == "__main__":
    main()








