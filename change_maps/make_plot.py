'''
Takes the CSV files output from CCDC, and outputs a change intensity map.
'''

import pandas as pd
import os
import fnmatch
import sys
import csv
import xarray as xr
from osgeo import gdal
from matplotlib import pyplot as plt

def main():

    if(len(sys.argv) > 1):
        file_path = sys.argv[1]

        print(file_path)
        
        if(os.path.isdir(file_path)):

            rows = []
        
            # For every file in the directory
            for change_file in os.listdir(file_path):

                if fnmatch.fnmatch(change_file, '*.csv'): # Check if it's a CSV file

                    sep_filename = change_file.replace('.', '_').split('_')

                    x_val = float(sep_filename[0])
                    y_val = float(sep_filename[2])

                    csv_file_path = os.path.join(file_path, change_file)

                    with open(csv_file_path, "r") as data_file: 
                       
                        file_reader = csv.reader(data_file)

                        num_changes = sum(1 for row in file_reader) - 1
                            
                        row = {'x': x_val, 'y': y_val, 'num_changes': num_changes}
                        rows.append(row)

        to_df = pd.DataFrame(rows).set_index(['y', 'x'])
    
        dataset = xr.Dataset.from_dataframe(to_df)
        
        img_name = "new_map.nc"

        dataset.to_netcdf(img_name, encoding={'num_changes': {'dtype': 'uint16', '_FillValue': 9999}})

        # Set CRS
        change_map = gdal.Open(img_name)
        change_map = gdal.Translate('change_map_crs.nc', change_map, outputSRS = 'wkt.txt')
        os.remove(img_name) # Remove change map with no CRS
        
        dataset['num_changes'].plot(cmap=plt.cm.OrRd)
        
        plt.tight_layout()
        plt.savefig("change_map_plot.png")
                     
                    
if __name__ == "__main__":

	main()



