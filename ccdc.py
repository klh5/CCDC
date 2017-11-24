import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from makeModel import MakeCCDCModel
from removeOutliers import RLMRemoveOutliers
from datetime import datetime

# Set up list of models
model_list = [None for i in range(7)]     # List of models, one for each band

def add_change_marker(plot_list, num_bands, change_point, obs_data):

   """ Adds a vertical line to each plot every time change is detected """

   for i in range(num_bands):
       y_min = np.amin(obs_data[:,i+1])
       y_max = np.amax(obs_data[:,i+1])

       plot_list[i].plot([change_point, change_point], [y_min, y_max], 'r', linewidth=1)

def setupModels(data_all_bands, num_bands):
    
    """Creates a model for each band and stores it in model_list"""

    # Get julian dates
    julian_dates = data_all_bands[:,0]
    
    # Create the model
    for i in range(num_bands):
        
        reflectance = data_all_bands[:,i+1]
        
        model = MakeCCDCModel()
        
        model.fit_model(reflectance, julian_dates)
        
        model_list[i] = model

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
    model_data = pixel_data[0:init_obs,:]

    # Start off with the model uninitialized
    model_init = False
    num_iters = 0

    model_end = None # Store the index of the end date of the current model period

    # Model initialization sequence
    while(model_init == False):
        
        num_data_points = len(model_data)
        
        if(num_data_points < init_obs):
            print("Not enough data points left to initialize model.")
            return []
    
        # Re-initialize the models
        setupModels(model_data, num_bands)

        # Get cumulative time
        total_time = 0

        for row in model_data:
            total_time += row[0]

        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0

        # Check for change during the initialization period. We need 12 observations with no change
        for i in range(num_bands): # For each Landsat band
            
            for row in model_data:
                slope_val = ((model_list[i].get_coefficients()[3]) * row[0]) / 3 * (model_list[i].get_rmse() / total_time)
                total_slope_eval += slope_val
    
            start_val = (model_data[0, i+1] - model_list[i].get_predicted(model_data[0, 0])) / (3 * model_list[i].get_rmse())
            total_start_eval += start_val

            end_val = (model_data[num_data_points-1, i+1] - model_list[i].get_predicted(model_data[num_data_points-1, 0])) / (3 * model_list[i].get_rmse())
            total_end_eval += end_val

        if(total_slope_eval > 1 or total_start_eval > 1 or total_end_eval > 1):
            num_iters += 1
            model_data = pixel_data[0+num_iters:init_obs+num_iters,:] # Shift up 1 row

        else:
            model_init = True
            model_end = init_obs + num_iters + 1
            print("Model initialized. Iterations needed: {}".format(num_iters))

    # Detect change
    change_flag = 0

    while((model_end+1) <= len(pixel_data)):

        new_obs = pixel_data[model_end] # Get next observation

        change_eval = 0 

        for i in range(num_bands):    # For each band
            residual_val = (new_obs[i+1] - model_list[i].get_predicted(new_obs[0])) / (model_list[i].get_rmse()*2)
            change_eval += residual_val

        if(change_eval <= 1):
            print("Adding new data point")
            model_data = np.vstack((model_data, new_obs))
            setupModels(model_data, num_bands)
            change_flag = 0 # Reset change flag because we have an inlier

        else:
            change_flag += 1 # Don't add the new pixel to the model
    
        if(change_flag == 6):
            print("Change detected!")
            add_change_marker(figures, num_bands, new_obs[0], pixel_data)
            return pixel_data[model_end:,]
        
        # Need to get the next observation
        model_end += 1
    
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
    
    # Get total number of clear observations in the dataset
    num_clear_obs = len(next_data)
    
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
        
        if(getNumYears(next_data[:,0]) > 0):
            
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








