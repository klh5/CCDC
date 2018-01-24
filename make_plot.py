import pandas as pd
import os
import fnmatch
import sys
import csv
import xarray as xr
import numpy as np

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
                    y_val = float(sep_filename[1])

                    csv_file_path = os.path.join(file_path, change_file)

                    with open(csv_file_path, "r") as data_file:
                        file_reader = csv.reader(data_file)

                        num_changes = sum(1 for row in file_reader) - 1
                            
                        row = {'x': x_val, 'y': y_val, 'num_changes': num_changes}
                        rows.append(row)

    to_df = pd.DataFrame(rows).set_index(['x', 'y'])
    
    dataset = xr.Dataset.from_dataframe(to_df)

    dataset.attrs['crs'] = 'PROJCS["WGS 84 / UTM zone 55N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",147],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32655"]]'

    print(dataset)

    dataset.to_netcdf('saved_on_disk.nc', encoding={'num_changes': {'dtype': 'uint16', '_FillValue': 9999}})
                     
                    
if __name__ == "__main__":

    main()



