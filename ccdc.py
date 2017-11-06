import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
from makeModel import MakeCCDCModel
from removeOutliers import RLMRemoveOutliers

# Set up list of models
model_list = [None for i in range(0,7)]     # List of models, one for each band

def plot_data(data_to_plot, figures, num_bands):
    
    """Creates seperate plots for each of the bands"""

    two_pi_div_T = (2 * np.pi) / 365
    
    for i in range(0, num_bands):
        ax2 = figures[i].add_subplot(111)
        # Plot the model
        f = interp1d(data_to_plot[:,0], model_list[i].get_coefficients()[0] + (model_list[i].get_coefficients()[1]*(np.cos(two_pi_div_T * data_to_plot[:,0]))) + (model_list[i].get_coefficients()[2]*(np.sin(two_pi_div_T * data_to_plot[:,0]))) + (model_list[i].get_coefficients()[3]*data_to_plot[:,0]), kind='cubic')
        xnew = np.linspace(data_to_plot[:,0].min(), data_to_plot[:,0].max(), 200)
        ax2.plot(xnew, f(xnew), 'green', linewidth=1)

def setupModels(data_all_bands, num_bands):
    
    """Creates a model for each band and stores it in model_list"""

    # Get julian dates
    julian_dates = data_all_bands[:,0]
    
    # Create the model
    for i in range(0, num_bands):
        
        reflectance = data_all_bands[:,i+1]
        
        model = MakeCCDCModel()
        
        model.fit_model(reflectance, julian_dates)
        
        model_list[i] = model

def findChange(pixel_data, figures, num_bands):
    
    """This function does two things: 
        1. Finds the next set of at least 6 clear observations from which the models can be initialized.
        2. Continues to add data points to the model until either a new breakpoint is detected, or there
        are not enough observations remaining."""

    # Subset first 12 clear pixels for model initialisation
    model_data = pixel_data[0:12,:]

    # Start off with the model uninitialized
    model_init = False
    num_iters = 0

    model_end = None # Store the index of the end date of the current model period
    
    robust_outliers = RLMRemoveOutliers()

    # Model initialization sequence
    while(model_init == False):
        
        # Get rid of any obvious outliers
        model_data = robust_outliers.clean_data(model_data)
    
        # This will later be adjusted so that the number of model terms is adjusted to the number of clear observations available
        if(len(model_data) < 6):
            print("Not enough data left after removing outliers")
            return []
        
        num_data_points = len(model_data)
    
        # Re-initialize the models
        setupModels(model_data, num_bands)

        # Get cumulative time
        total_time = 0

        for row in model_data:
            total_time += row[0]

        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0

        # Check for change during the initialization period. We need 12 clear observations with no change
        for i in range(0, num_bands): # For each Landsat band
            
            for row in model_data:
                slope_val = (np.absolute(model_list[i].get_coefficients()[3]) * row[0]) / 3 * model_list[i].get_rmse() / total_time
                total_slope_eval += slope_val
    
            start_val = np.absolute(model_data[0, i+1] - model_list[i].get_predicted(model_data[0, 0])) / (3 * model_list[i].get_rmse())
            total_start_eval += start_val

            end_val = np.absolute(model_data[num_data_points-1, i+1] - model_list[i].get_predicted(model_data[num_data_points-1, 0])) / (3 * model_list[i].get_rmse())
            total_end_eval += end_val

        if(total_slope_eval > 1 or total_start_eval > 1 or total_end_eval > 1):
            num_iters += 1
            model_data = pixel_data[0+num_iters:12+num_iters,:] # Shift up 1 row

        else:
            model_init = True
            model_end = 12 + num_iters
            print("Model initialized. Iterations needed: ", num_iters)

    # Detect change
    while((model_end+1) <= len(pixel_data)):

        new_data = pixel_data[model_end:model_end+6,:] # Get next six data points

        # Data point can be flagged as being potential change
        change_flag = 0
    
        for index, row in enumerate(new_data):        # For each new data point
            change_eval = 0
        
            for i in range(0,num_bands):    # For each band
                residual_val = np.absolute(row[i+1] - model_list[i].get_predicted(row[0])) / (model_list[i].get_rmse()*2)
                change_eval += residual_val

            if(change_eval <= 1):
                print("Adding new data point")
                model_data = np.vstack((model_data, row))
            else:
                change_flag += 1
    
        if(change_flag == 6):
            print("Change detected!")
            plot_data(model_data, figures, num_bands)
            return pixel_data[model_end:,]
    
        else:
            print("Re-initializing models with new data point(s)")
            setupModels(model_data, num_bands)
        
        # Need to get the next three pixels, whether or not the model has been updated
        model_end += 6
    
    # No change detected, end of data reached
    plot_data(model_data, figures, num_bands)
    return []

def main():
    
    """Program runs from here"""
    
    if(len(sys.argv) > 1):
        next_data = np.loadtxt(sys.argv[1], delimiter = ',')
    else:
        print("No data file specified. Exiting")
        sys.exit()

    # Get number of bands, which will be the number of columns - 2
    num_bands = next_data.shape[1] - 2

    # One figure for each band - makes the plots much simpler
    fig_list = []
    
    # Sort data by date
    next_data = next_data[np.argsort(next_data[:,0])]
    
    # Only select clear pixels (0 is clear land; 1 is clear water)
    next_data = next_data[next_data[:,num_bands+1] < 2]
    
    for i in range(0, num_bands):
        fig_list.append(plt.figure("Band" + str(i+1)))
        ax1 = fig_list[i].add_subplot(111)
        ax1.plot(next_data[:,0], next_data[:,i+1], 'o', color='black', label='Original data', markersize=2)

    # We need at least 15 observations determined as clear by Fmask to proceed
    while(len(next_data) > 0):
        
        if(len(next_data) >= 15):
            next_data = findChange(next_data, fig_list, num_bands)

        else:
            break

    # Once there is no more data to process, plot the results
    plt.legend(['Original data', 'Lasso fit'])
    plt.show()


if __name__ == "__main__":
    main()










