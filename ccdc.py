import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
from makeModel import MakeCCDCModel
from removeOutliers import RLMRemoveOutliers

# Set up list of models
model_list = [None for i in range(0,7)]     # List of models, one for each band

def plot_data(data_to_plot, figures):

    two_pi_div_T = (2 * np.pi) / 365
    
    for i in range(0,7):
        ax1 = figures[i].add_subplot(111)
        ax1.plot(data_to_plot[:,0], data_to_plot[:,i+1], 'o', color='black', label='Original data', markersize=2)
        
        # Plot the model
        ax2 = figures[i].add_subplot(111)
        f = interp1d(data_to_plot[:,0], model_list[i].get_coefficients()[0] + (model_list[i].get_coefficients()[1]*(np.cos(two_pi_div_T * data_to_plot[:,0]))) + (model_list[i].get_coefficients()[2]*(np.sin(two_pi_div_T * data_to_plot[:,0]))) + (model_list[i].get_coefficients()[3]*data_to_plot[:,0]), kind='cubic')
        xnew = np.linspace(data_to_plot[:,0].min(), data_to_plot[:,0].max(), 200)
        ax2.plot(xnew, f(xnew), 'green', linewidth=1)

        #figname = "/Users/Katie/band" + str(i+1) + ".png"

        #plt.legend()
        #plt.savefig(figname)
        #plt.show()
        #plt.close(fig)

def setupModels(data_all_bands):

    # Get julian dates
    julian_dates = data_all_bands[:,0]
    
    # Create the model
    for i in range(0, 7):
        
        reflectance = data_all_bands[:,i+1]
        
        model = MakeCCDCModel()
        
        model.fit_model(reflectance, julian_dates)
        
        model_list[i] = model

def findChange(pixel_data, figures):

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
    
        if(len(model_data) < 10):
            print("Not enough data left after removing outliers")
            return []
        
        num_data_points = len(model_data)
    
        # Re-initialize the models
        setupModels(model_data)

        # Get cumulative time
        total_time = 0

        for row in model_data:
            total_time += row[0]

        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0

        # Check for change during the initialization period. We need 12 clear observations with no change
        for i in range(0, 7): # For each Landsat band
            
            for row in model_data:
                slope_val = np.absolute(model_list[i].get_coefficients()[3] * row[0]) / model_list[i].get_multiplied_rmse() / total_time
                total_slope_eval += slope_val
    
            start_val = np.absolute(model_data[0, i+1] - model_list[i].get_predicted(model_data[0, 0])) / (model_list[i].get_multiplied_rmse())
            total_start_eval += start_val

            end_val = np.absolute(model_data[num_data_points-1, i+1] - model_list[i].get_predicted(model_data[num_data_points-1, 0])) / (model_list[i].get_multiplied_rmse())
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

        new_data = pixel_data[model_end:model_end+3,:] # Get next three data points

        # Data point can be flagged as being potential change
        change_flag = 0
    
        for index, row in enumerate(new_data):        # For each new data point
            change_eval = 0
        
            for i in range(0,7):    # For each band
                residual_val = np.absolute(row[i+1] - model_list[i].get_predicted(row[0])) / model_list[i].get_multiplied_rmse()
                change_eval += residual_val

            if(change_eval < 1):
                print("Adding new data point")
                model_data = np.vstack((model_data, row))
            else:
                change_flag += 1
    
        if(change_flag == 3):
            print("Change detected!")
            plot_data(model_data, figures)
            return pixel_data[model_end:,] # CHECK
    
        else:
            print("Re-initializing models with new data point(s)")
            setupModels(model_data)
        
        # Need to get the next three pixels, whether or not the model has been updated
        model_end += 3
    
    # No change detected, end of data reached
    plot_data(model_data, figures)
    return []

def main():
    
    # One figure for each band - makes the plots much simpler
    fig_list = [plt.figure("Band" + str(i+1)) for i in range(0, 7)]
    
    if(len(sys.argv) > 1):
        next_data = np.genfromtxt(sys.argv[1], delimiter = ',')
    else:
        print("No data file specified. Exiting")
        sys.exit()

    next_data = next_data[np.argsort(next_data[:,0])]        # Sort data by DOY
    
    # Only select clear pixels
    next_data = next_data[next_data[:,8] < 2]

    # We need at least 15 observations determined as clear by Fmask to proceed
    while(len(next_data) > 0):
        
        if(len(next_data) >= 15):
            next_data = findChange(next_data, fig_list)

    plt.legend(['Original data', 'OLS model'])
    plt.show()


if __name__ == "__main__":
    main()










