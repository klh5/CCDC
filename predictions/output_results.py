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
import datacube
from osgeo import gdal, osr
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta

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
    
    dataset = dataset.sortby("y", ascending=False)
    
    print("Generating output...")
    
    # Create multi-band KEA file from the predictions
    
    # Taken from https://stackoverflow.com/questions/32609570/how-to-set-the-band-description-option-tag-of-a-geotiff-file-using-gdal-gdalw
    date = csv_dir.split('/')[-1]
    multi_band = "{}.kea".format(date)
    x_size = len(dataset.x.values)
    y_size = len(dataset.y.values)
    cell_size = args.cell_size
    x_min = np.amin(dataset.x.values)
    x_max = np.amax(dataset.x.values)
    y_min = np.amin(dataset.y.values)
    y_max = np.amax(dataset.y.values)
    num_bands = len(dataset.data_vars)

    geo_transform = (x_min, cell_size, 0.0, y_max, 0.0, -cell_size)
    
    driver = gdal.GetDriverByName('KEA')
    pred_raster = driver.Create(multi_band, x_size, y_size, num_bands, 2)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(args.coord_system)
    srs = srs.ExportToWkt()
        
    pred_raster.SetProjection(srs)
    pred_raster.SetGeoTransform(geo_transform)
    
    for band_num, var in enumerate(dataset.data_vars):      
        raster_band = pred_raster.GetRasterBand(band_num+1)
        raster_band.SetNoDataValue(0)
        raster_band.SetDescription(var)
        dataset[var] = dataset[var].fillna(args.nodata_val)
        band_min = np.amin(dataset[var].values)
        band_max = np.amax(dataset[var].values)
        band_mean = np.mean(dataset[var].values)
        band_sd = np.std(dataset[var].values)
        raster_band.SetStatistics(band_min, band_max, band_mean, band_sd)
        raster_band.WriteArray(dataset[var].values)
        
    pred_raster = None
        
    if(args.make_comparison):
        
        # Open predicted KEA image for comparison
        predicted_img = gdal.Open(multi_band)
        
        aspect = x_size / y_size # Get aspect ratio for plots, since area is not always square
        
        plt.rcParams.update({'font.size': 22})
        
        if(args.product):
            
            # User has provided a Data Cube product which the real image belongs to

            start_date = datetime.strptime(date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1)
           
            epsg_code = "EPSG:{}".format(args.coord_system)
    
            dc = datacube.Datacube()
           
            real_data = dc.load(product=args.product, measurements=bands, time=(start_date, end_date), x=(x_min, x_max), y=(y_min, y_max), crs=epsg_code, output_crs=epsg_code, resolution=(-args.cell_size, args.cell_size))
    
            if(real_data.variables):
               
                # Output KEA file for real data
                real_file_name = "{}_real.kea".format(date)
               
                real_img = driver.Create(real_file_name, x_size, y_size, num_bands, 2)
               
                real_img.SetProjection(srs)
                real_img.SetGeoTransform(geo_transform)
               
                for band_num, var in enumerate(dataset.data_vars):      
                    raster_band = real_img.GetRasterBand(band_num+1)
                    raster_band.SetNoDataValue(0)
                    raster_band.SetDescription(var)
                    real_data[var] = real_data[var].fillna(0)
                    band_min = np.amin(real_data[var].values)
                    band_max = np.amax(real_data[var].values)
                    band_mean = np.mean(real_data[var].values)
                    band_sd = np.std(real_data[var].values)
                    raster_band.SetStatistics(float(band_min), float(band_max), float(band_mean), float(band_sd))
                    raster_band.WriteArray(real_data[var].values.reshape(y_size, -1))
                   
                real_img = None
                          
                # Create difference plots and output RMSE values
               
                real_img = gdal.Open(real_file_name)
                 
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
                             dataset[band_diff_name].plot(aspect=aspect, size=20)
                             plt.xticks(rotation='vertical')
                             plt.tight_layout()
                             diff_plot_name = "{}_diff.png".format(pred_desc)
                             plt.savefig(diff_plot_name)
                             plt.close()
                        
                             # Print RMSE
                             band_rmse = np.sqrt(np.mean((dataset[band_diff_name].values) ** 2))
                             print("{} RMSE is: {}".format(pred_desc, band_rmse))
                                   
            else:
               print("No matching image could be found from the provided product.")
               
        elif(args.real_img):
            
            # Name shapefile
            shapefile = "predicted.shp"
            
            # Create shapefile
            subprocess.call(["gdaltindex", shapefile, multi_band])
            
            # Crop provided image to fit the analysed area
            crop_name = args.real_img.replace(".kea", "_cropped.kea")       
            subprocess.call(["gdalwarp", "-of", "kea", "-cutline", shapefile, "-crop_to_cutline", args.real_img, crop_name])
           
            real_img = gdal.Open(crop_name)
            
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
                        dataset[band_diff_name].plot(aspect=aspect, size=20)
                        plt.xticks(rotation='vertical')
                        plt.tight_layout()
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
    parser.add_argument('-crs', '--coord_system', type=int, required=True, help="EPSG code for the CRS of the data. Enter as a number, e.g. 32655")
    parser.add_argument('-comp', '--make_comparison', type=bool, help="Whether or not to output RMSE for the predicted image.")
    parser.add_argument('-nodat', '--nodata_val', type=float, required=True, help="Value for pixels with invalid/no data.")
    parser.add_argument('-cell', '--cell_size', required=True, type=int, help="Spatial resolution, e.g. 30 for Landsat.")
    parser.add_argument('-p', '--product', help="Product name for comparison dataset. The actual comparison image will be found based on the date and spatial extent of the analysed data.")
    parser.add_argument('-img', '--real_img', help="Location of corresponding real image. Can be used instead of fetching real image from the Data Cube.")
    
    args = parser.parse_args()
    
    main(args)



