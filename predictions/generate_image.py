'''
Takes the CSV files output from CCDC prediction mode, and outputs a predicted image.
'''

import pandas as pd
import os
import fnmatch
import csv
import xarray as xr
import subprocess
import argparse
import numpy as np
from osgeo import gdal, osr
from matplotlib import pyplot as plt

def main(args):

    csv_dir = args.csv_dir

    print("CSV file directory:", csv_dir)
    
    if(os.path.isdir(csv_dir)):
        
        print("Gathering prediction data...")

        bands = []

        rows = []
        csv_count = 1
    
        # For every file in the directory
        for pixel_file in os.listdir(csv_dir):

            if fnmatch.fnmatch(pixel_file, '*.csv'): # Check if it's a CSV file
                
                print("Processing file: {} Count: {}".format(pixel_file, csv_count))
                
                sep_filename = pixel_file.replace('.', '_').split('_')

                x_val = float(sep_filename[0])
                y_val = float(sep_filename[2])

                csv_file_path = os.path.join(csv_dir, pixel_file)

                with open(csv_file_path, "r") as data_file: 
                   
                    file_reader = csv.reader(data_file)
                    
                    row = {'x': x_val, 'y': y_val}
                    
                    headers = next(file_reader)
                    
                    if(not bands):
                        bands = headers
                    
                    try:
                        band_data = next(file_reader)
                        
                        for band, value in zip(headers, band_data):
                                band_value = float(value)
                                
                                if(band_value < 0):
                                    band_value = args.nodata_val
                                    
                                row[band] = band_value
                        
                        rows.append(row)
                        
                    except:
                        print("No predictions present in file {}".format(pixel_file))

                    csv_count += 1
  
    # Index has to be y,x so that the numpy arrays are the right shape
    # If index is x,y the numpy arrays will have x rows and y columns, wheras actually we have y rows and x columns
    to_df = pd.DataFrame(rows).set_index(['y', 'x'])

    dataset = xr.Dataset.from_dataframe(to_df)

    print("Generating output...")
    
    # Create multi-band KEA file from the predictions
    
    # Taken from https://stackoverflow.com/questions/32609570/how-to-set-the-band-description-option-tag-of-a-geotiff-file-using-gdal-gdalw
    multi_band = "{}.kea".format(csv_dir.split('/')[-1])
    x_size = len(dataset.x)
    y_size = len(dataset.y)
    cell_size = 30
    x_min = min(dataset.x)
    y_max = max(dataset.y)
    num_bands = len(dataset.data_vars)
    
    geo_transform = (x_min, cell_size, 0, y_max, 0, -cell_size)
    
    driver = gdal.GetDriverByName('KEA')
    new_raster = driver.Create(multi_band, x_size, y_size, num_bands, 2) # Datatype = 2 same as gdal.GDT_UInt16
    
    if(args.real_img):
        print("Image file directory:", args.real_img)
        orig_img = gdal.Open(args.real_img)
        srs = orig_img.GetProjection()
        orig_img = None
        
    elif(args.coord_system):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(args.coord_system)
        srs = srs.ExportToWkt()
        
    new_raster.SetProjection(srs)
    new_raster.SetGeoTransform(geo_transform)
    
    for band_num, var in enumerate(dataset.data_vars):      
        raster_band = new_raster.GetRasterBand(band_num+1)
        raster_band.SetNoDataValue(0)
        raster_band.SetDescription(var)
        dataset[var] = dataset[var].fillna(0)
        band_min = np.amin(dataset[var].values)
        band_max = np.amax(dataset[var].values)
        band_mean = np.mean(dataset[var].values)
        band_sd = np.std(dataset[var].values)
        raster_band.SetStatistics(band_min, band_max, band_mean, band_sd)
        raster_band.WriteArray(dataset[var].values)
        
    new_raster = None
        
    if(args.real_img and args.make_comparison):
        
        # Check format of real image
        # If KEA, carry on
        # If HDF (MODIS), convert to KEA
        
        # Name shapefile
        shapefile = "predicted.shp"
        
        # Create shapefile
        subprocess.call(["gdaltindex", shapefile, multi_band])
        
        # Crop provided image to fit the analysed area
        crop_name = args.real_img.replace(".kea", "_cropped.kea")       
        subprocess.call(["gdalwarp", "-of", "kea", "-cutline", shapefile, "-crop_to_cutline", args.real_img, crop_name])
       
        real_img = gdal.Open(crop_name)
        predicted_img = gdal.Open(multi_band)
        
        for pred_band_num in range(predicted_img.RasterCount):
            pred_srcband = predicted_img.GetRasterBand(pred_band_num+1)
            pred_desc = pred_srcband.GetDescription()
            
            for real_band_num in range(real_img.RasterCount):
                real_srcband = real_img.GetRasterBand(real_band_num+1)
                
                if(real_srcband.GetDescription().lower() == pred_desc):
                    pred_data = np.array(pred_srcband.ReadAsArray())
                    real_data = np.array(real_srcband.ReadAsArray())
                    difference = np.subtract(real_data.astype(float), pred_data.astype(float))
                    
                    band_diff_name = "{}_difference".format(pred_desc)
                    dataset[band_diff_name] = (('y', 'x'), difference)
                    
                    # Make difference plots
                    dataset[band_diff_name].plot()
                    plt.gca().invert_yaxis()
                    
                    diff_plot_name = "{}_diff.png".format(pred_desc)
                    plt.savefig(diff_plot_name)
                    plt.close()
                    
                    # Print RMSE
                    band_rmse = np.sqrt(np.mean((dataset[band_diff_name].values) ** 2))
                    print("{} RMSE is: {}".format(pred_desc, band_rmse))
        
        real_img = None
        predicted_img = None
                 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate output from predicted values produced by CCDC.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-csv', '--csv_dir', required=True, help="The directory containing the CSV files to be processed.")
    parser.add_argument('-crs', '--coord_system', help="EPSG code for the CRS of the input data. Enter as a number, e.g. 32655")
    parser.add_argument('-img', '--real_img', help="Location of corresponding real image. Can be used to make a comparison or to get CRS.")
    parser.add_argument('-comp', '--make_comparison', type=bool, help="Whether or not to output RMSE for the predicted image.")
    parser.add_argument('-nodat', '--nodata_val', type=float, help="Value for pixels with invalid/no data.")

    args = parser.parse_args()
    
    main(args)



