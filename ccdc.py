import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from makeModel import MakeCCDCModel
from removeOutliers import RLMRemoveOutliers
from datetime import datetime

# Set up list of models
model_list = [None for i in range(5)]     # List of models, one for each band

def add_change_marker(plot_list, num_bands, change_point, obs_data):

   """ Adds a vertical line to each plot every time change is detected """
   
   for i in range(num_bands):
       y_min = np.amin(obs_data.iloc[:,i+1])
       y_max = np.amax(obs_data.iloc[:,i+1])

       plot_list[i].plot([change_point, change_point], [y_min, y_max], 'r', linewidth=1)

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
    
    last_date = datetime.fromordinal(int(np.amax(date_list))).strftime('%Y')
    first_date = datetime.fromordinal(int(np.amin(date_list))).strftime('%Y')
        
    num_years = int(last_date) - int(first_date)

    return num_years

def findChange(pixel_data, figures, num_bands, num_years, init_obs):
    
    """This function does two things: 
        1. Finds the next set of 6/12/18/24 observations without change from which the models can be initialized.
        2. Continues to add data points to the model until either a new breakpoint is detected, or there
        are not enough observations remaining."""

    # Subset first n clear observations for model initialisation
    model_data = pixel_data.iloc[0:init_obs,:]

    # Start off with the model uninitialized
    model_init = False
    num_iters = 0
    
    # The next observation to be added to the model to detect change
    next_obs = None

    # Model initialization sequence
    while(model_init == False):
        
        num_data_points = len(model_data)
        
        if(num_data_points < init_obs):
            print("Not enough data points left to initialize model.")
            return []
    
        # Re-initialize the models
        setupModels(model_data, num_bands, init_obs)

        # Get cumulative time
        total_time = model_data['datetime'].sum()

        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0

        # Check for change during the initialization period. We need 12 observations with no change
        for band_model in model_list: # For each model

            for row in band_model.get_band_data().iterrows():
                slope_val = ((band_model.get_coefficients()['datetime']) * row[0]) / 3 * (band_model.get_rmse() / total_time)
                total_slope_eval += slope_val

            start_val = (band_model.get_band_data()['reflectance'].iloc[0] - band_model.get_band_data()['predicted'].iloc[0]) / (3 * band_model.get_rmse())
            total_start_eval += start_val

            end_val = (band_model.get_band_data()['reflectance'].iloc[num_data_points-1] - band_model.get_band_data()['predicted'].iloc[num_data_points-1]) / (3 * band_model.get_rmse())
            total_end_eval += end_val

        if(total_slope_eval > 1 or total_start_eval > 1 or total_end_eval > 1):
            num_iters += 1
            model_data = pixel_data.iloc[0+num_iters:init_obs+num_iters,:] # Shift up 1 row

        else:
            model_init = True
            next_obs = init_obs + num_iters + 1
            print("Model initialized. Iterations needed: {}".format(num_iters))

    # Detect change
    change_flag = 0

    while((next_obs+1) <= len(pixel_data)):

        change_eval = 0
        
        new_date = pixel_data.iloc[next_obs][0]
        
        for model_num, band_model in enumerate(model_list):    # For each band
            new_ref_obs = pixel_data.iloc[next_obs][model_num+1]
            residual_val = (new_ref_obs - band_model.get_prediction(new_date)[0]) / (band_model.get_rmse() * 2)
            change_eval += residual_val
        
        if(change_eval <= 1):
            print("Adding new data point")
            new_obs = pixel_data.iloc[next_obs]
            pd.concat([model_data, new_obs])
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
        data_in = pd.read_csv(sys.argv[1])
    else:
        print("No data file specified. Exiting")
        sys.exit()

    # Get number of bands, which will be the number of columns - 2
    num_bands = len(data_in.columns) - 2

    # One figure for each band - makes the plots much simpler
    plt_list = []
    
    # Sort data by date
    data_in = data_in.sort_values(by=['datetime'])
    
    # Only select clear pixels (0 is clear land; 1 is clear water)
    next_data = data_in[data_in.qa < 2]
    
    # Get the number of years covered by the dataset
    num_years = getNumYears(next_data.datetime)
    
    # Screen for outliers
    robust_outliers = RLMRemoveOutliers()
    next_data = robust_outliers.clean_data(next_data, num_years)
    
    fig = plt.figure()

    # Set up basic plots with original data    
    for i in range(num_bands):
        plt_list.append(fig.add_subplot(num_bands, 1, i+1))
        band_col = "band_" + str(i+1)
        plt_list[i].plot(next_data['datetime'], next_data.iloc[:,i+1], 'o', color='black', label='Original data', markersize=2)
        plt_list[i].set_ylabel(band_col)
    
    # We need at least 12 clear observations (6 + 6 to detect change)
    while(len(next_data) >= 12):
        
        if(getNumYears(next_data['datetime']) > 0):
            
            # Get total number of clear observations in the dataset
            num_clear_obs = len(next_data)
    
            if(num_clear_obs >= 12 and num_clear_obs < 18):
            # Use simple model with initialization period of 6 obs
               next_data = findChange(next_data, plt_list, num_bands, num_years, 6)
            
            elif(num_clear_obs >= 18 and num_clear_obs < 24):
               # Use simple model with initialization period of 12 obs
               next_data = findChange(next_data, plt_list, num_bands, num_years, 12)

            elif(num_clear_obs >= 24 and num_clear_obs < 30):
               # Use advanced model with initialization period of 18 obs
               next_data = findChange(next_data, plt_list, num_bands, num_years, 18)
            
            elif(num_clear_obs >= 30):
               # Use full model with initialisation period of 24 obs
               next_data = findChange(next_data, plt_list, num_bands, num_years, 24)
            
        else:
            break

    # Once there is no more data to process, plot the results
    plt.legend(['Original data', 'Change point'])
    plt.show()


if __name__ == "__main__":
    main()








