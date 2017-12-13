import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import datacube
import xarray as xr
import argparse
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
            print("Could not find a period of no change for model initialization.")
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
    change_time = None

    while((next_obs+1) <= len(pixel_data)):

        change_eval = 0
        
        new_obs = pixel_data.iloc[next_obs]
        new_date = new_obs[0]
        
        for model_num, band_model in enumerate(model_list):    # For each band
            new_ref_obs = new_obs[model_num+1]
            residual_val = np.absolute((new_ref_obs - band_model.get_prediction(new_date)[0])) / (band_model.get_rmse() * 2)
            change_eval += residual_val
        
        if(change_eval <= 1):
            print("Adding new data point")
            model_data.append(new_obs, ignore_index=True)
            setupModels(model_data, num_bands, init_obs)
            change_flag = 0 # Reset change flag because we have an inlier

        else:
            change_flag += 1 # Don't add the new pixel to the model
            if(change_flag == 1): # If this is the first observed possible change point
                change_time = new_date
    
        if(change_flag == 6):
            print("Change detected!")
            add_change_marker(figures, num_bands, change_time, pixel_data)
            return pixel_data.iloc[next_obs:,]
        
        # Need to get the next observation
        next_obs += 1
    
    # No change detected, end of data reached
    return []

def transform_to_df(dataset_to_transform):

    """Transforms xarray Dataset object into a Pandas dataframe"""

    new_df = pd.DataFrame()
        
    new_df['datetime'] = dataset_to_transform.time.data
    new_df['datetime'] = new_df['datetime'].apply(lambda x: x.toordinal())

    for name, var in dataset_to_transform.data_vars.items():
        new_df[name] = np.reshape(var.data, -1)
            
    # Points at the edge of the image could return empty arrays (all 0's) - this will remove any columns to which this applies
    new_df = new_df.dropna(axis=1, how='all')

    return new_df

def main(args):
    
    """Program runs from here"""

    dc = datacube.Datacube()

    # Set some spatial boundaries for the data
    #lower_lat = -27.0
    #upper_lat = -25.0
    #left_long = 146.0
    #right_long = 149.0

    boundary = ogr.Geometry(ogr.wkbLinearRing)
    boundary.AddPoint(lower_lat, left_long)
    boundary.AddPoint(lower_lat, right_long)
    boundary.AddPoint(upper_lat, right_long)
    boundary.AddPoint(upper_lat, left_long)
    boundary.AddPoint(lower_lat, left_long)

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
            
                sref = dc.load(product='ls8_arcsi_sref_ingested', measurements=['red', 'green', 'nir', 'swir1', 'swir2'], lat=(new_point_lat), lon=(new_point_long))
                toa = dc.load(product='ls8_arcsi_toa_ingested', measurements=['red', 'green', 'nir', 'swir1', 'swir2'], lat=(new_point_lat), lon=(new_point_long))
                
                # Change nodata values (0's) to NaN
                sref = mask_invalid_data(sref)
                toa = mask_invalid_data(toa)

                if(sref.notnull() and toa.notnull()):
                
                    sref_data = transform_to_df(sref)
                    toa_data = transform_to_df(toa)

                    if(sref_data.shape[1] == 6 and sref_data.shape[1] == 6):
                    
                        # Drop any rows which have NaN values for one or more bands (we need data for all bands to be complete)
                        sref_data = sref_data.dropna(axis=0, how='any')
                        toa_data = toa_data.dropna(axis=0, how='any')

                        num_bands = 5

                        # One figure for each band - makes the plots much simpler
                        plt_list = []
    
                        # Sort data by date
                        sref_data = sref_data.sort_values(by=['datetime'])
                        toa_data = toa_data.sort_values(by=['datetime'])
                        
                        if(len(sref_data) == len(toa_data)):
    
                            # Get the number of years covered by the dataset
                            num_years = getNumYears(sref_data['datetime'])
                            
                            # The algorithm needs at least 1 year of data
                            if(getNumYears(sref_data['datetime']) > 0):
    
                                # Screen for outliers
                                robust_outliers = RLMRemoveOutliers()
                                outlier_list = robust_outliers.clean_data(toa_data, num_years)
                        
                                next_data = sref_data.drop(outlier_list)
                                next_data = next_data.reset_index(drop=True)

                                # Update num_years now outliers have been removed
                                num_years = getNumYears(next_data['datetime'])
    
                                fig = plt.figure(figsize=(20, 10))
                        
                                curr_points += 1

                                # Set up basic plots with original data
                                for i in range(num_bands):
                                    plt_list.append(fig.add_subplot(num_bands, 1, i+1))
                                    band_col = next_data.columns[i+1]
                                    plt_list[i].plot(sref_data['datetime'], sref_data.iloc[:,i+1], 'o', color='blue', label='Original data', markersize=2)
                                    plt_list[i].plot(next_data['datetime'], next_data.iloc[:,i+1], 'o', color='black', label='Data after RIRLS', markersize=3)
                                    plt_list[i].set_ylabel(band_col)
                                    myFmt = mdates.DateFormatter('%m/%Y') # Format dates as year rather than ordinal dates
                                    plt_list[i].xaxis.set_major_formatter(myFmt)
    
                                # We need at least 12 clear observations (6 + 6 to detect change)
                                while(len(next_data) >= 12):
        
                                    if(getNumYears(next_data['datetime']) > 0):
            
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
                                        print("Less than 1 year of observations remaining.")
                                        break
                                
                            
                                print("Ran out of observations.")

                                # Once there is no more data to process, plot the results
                                plt.legend(['Original data', 'Data after RIRLS', 'Change point'])
                                plt.tight_layout()
                                plt_name = "/Users/Katie/CCDC/plots/" + str(new_point.GetX()) + "_" + str(new_point.GetY()) + ".png"
                                plt.savefig(plt_name)
                                plt.clf()

                        else:
                            print("SREF and TOA data not the same length. Check indexing/ingestion.")



if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='Run CCDC algorithm using Data Cube.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lowerlat', '--llat', type=float, help='The lower latitude boundary of the area to be processed.')
    parser.add_argument('-upperlat', '--ulat', type=float, help='The upper latitude boundary of the area to be processed.')
    parser.add_argument('-lowerlon', '--llon', type=float, help='The lower longitude boundary of the area to be processed.')
    parser.add_argument('-upperlon', '--ulon', type=float, help='The upper longitude boundary of the area to be processed.')
    parser.add_argument('-mode', '--m', choices=['whole','sub'], default='sub', help='Specifies whether the entire area should be processed, or a random subsample.')
    parser.add_argument('-num_points', '--num', type=int, default=100, help='Specifies the number of subsamples to take if a random subsample is being processed.')
    args = parser.parse_args()
    
    main(args)








