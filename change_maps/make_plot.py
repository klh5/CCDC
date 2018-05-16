import pandas as pd
import os
import fnmatch
import sys
import csv
import xarray as xr
import numpy as np
import re
from osgeo import gdal

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
						num_changes = num_changes * 100 # Scale up
                            
						row = {'x': x_val, 'y': y_val, 'num_changes': num_changes}
						rows.append(row)

		to_df = pd.DataFrame(rows).set_index(['y', 'x'])
    
		dataset = xr.Dataset.from_dataframe(to_df)

		dataset.to_netcdf('new_map.nc', encoding={'num_changes': {'dtype': 'uint16', '_FillValue': 9999}})

		# Set CRS
		change_map = gdal.Open('new_map.nc')
		change_map = gdal.Translate('new_map.nc', change_map, outputSRS = 'wkt.txt')
                     
                    
if __name__ == "__main__":

	main()



