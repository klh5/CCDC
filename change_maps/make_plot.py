'''
Takes the CSV files output from CCDC, and outputs a change intensity map.
'''

import pandas as pd
import os
import fnmatch
import sys
import csv
import xarray as xr
import subprocess
from osgeo import gdal
from matplotlib import pyplot as plt

def main():

    if(len(sys.argv) > 1):
        csv_dir = sys.argv[1]
        img_dir = sys.argv[2]
        
        csv_dir = "/media/hdd_1/output/"
        img_dir = "ls8_img.kea"

        print("CSV file directory:", csv_dir)
        print("Image file directory:", img_dir)
        
        if(os.path.isdir(csv_dir)):

            rows = []
        
            # For every file in the directory
            for change_file in os.listdir(csv_dir):

                if fnmatch.fnmatch(change_file, '*.csv'): # Check if it's a CSV file

                    sep_filename = change_file.replace('.', '_').split('_')

                    x_val = float(sep_filename[0])
                    y_val = float(sep_filename[2])

                    csv_file_path = os.path.join(csv_dir, change_file)

                    with open(csv_file_path, "r") as data_file: 
                       
                        file_reader = csv.reader(data_file)

                        num_changes = sum(1 for row in file_reader) - 1
                            
                        row = {'x': x_val, 'y': y_val, 'num_changes': num_changes}
                        rows.append(row)

        to_df = pd.DataFrame(rows).set_index(['y', 'x'])
    
        dataset = xr.Dataset.from_dataframe(to_df)
        
        temp_img = "change_temp.nc"
        change_map_crs = "change_map_crs.nc"

        dataset.to_netcdf(temp_img, encoding={'num_changes': {'dtype': 'uint16', '_FillValue': 9999}})

        # Set CRS
        change_map = gdal.Open(temp_img)
        change_map = gdal.Translate(change_map_crs, change_map, outputSRS = 'wkt.txt')
        os.remove(temp_img) # Remove change map with no CRS
        
        # Create plot
        dataset['num_changes'].plot(cmap=plt.cm.OrRd)
        
        plt.tight_layout()
        plt.savefig("change_map_plot.png")
        
        # Create shapefile
        shapefile = "change_shapefile.shp"
        
        gen_shapefile = subprocess.call(["gdaltindex", shapefile, change_map_crs])
        
        # Crop provided image to fit the analysed area
        crop_name = img_dir.replace(".kea", "_cropped.kea")
        
        cut_to_shape = subprocess.call(["gdalwarp", "-of", "kea", "-cutline", shapefile, "-crop_to_cutline", img_dir, crop_name])
        
        # Create false colour image to compare with change map
        orig_img = gdal.Open(img_dir)
        
        band4 = orig_img.GetRasterBand(4)
        band5 = orig_img.GetRasterBand(5)
        band6 = orig_img.GetRasterBand(6)
        
        band4_stats = band4.GetStatistics(True, True)
        band5_stats = band5.GetStatistics(True, True)
        band6_stats = band6.GetStatistics(True, True)
        
        band4_min = str(band4_stats[0])
        band4_max = str(band4_stats[1])
        
        band5_min = str(band5_stats[0])
        band5_max = str(band5_stats[1])
        
        band6_min = str(band6_stats[0])
        band6_max = str(band6_stats[1])
        
        false_colour = img_dir.replace(".kea", "_false_colour.pdf")
        make_false_colour = subprocess.call(["gdal_translate", crop_name, "-b", "6", "-b", "5", "-b", "4", "-of", "pdf", "-ot", "byte", "-scale_1", band6_min, band6_max, "-scale_2", band5_min, band5_max, "-scale_3", band4_min, band4_max, "-a_nodata", "255", false_colour])

                    
if __name__ == "__main__":

	main()



