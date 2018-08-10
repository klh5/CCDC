'''
Takes the CSV files output from CCDC, and outputs a change intensity map.
'''

import pandas as pd
import os
import re
import fnmatch
import csv
import xarray as xr
import subprocess
import argparse
import numpy as np
from osgeo import gdal, osr, ogr
from matplotlib import pyplot as plt

def main(args):

    if(args.csv_dir):

        print("CSV file directory:", args.csv_dir)
        
        if(os.path.isdir(args.csv_dir)):
            
            print("Gathering change data...")

            rows = []
            csv_count = 1
        
            # For every file in the directory
            for change_file in os.listdir(args.csv_dir):

                if fnmatch.fnmatch(change_file, '*.csv'): # Check if it's a CSV file
                    
                    print("Processing file: {} Count: {}".format(change_file, csv_count))
                    
                    sep_filename = change_file.replace('.', '_').split('_')

                    x_val = float(sep_filename[0])
                    y_val = float(sep_filename[2])

                    csv_file_path = os.path.join(args.csv_dir, change_file)

                    with open(csv_file_path, "r") as data_file: 
                       
                        file_reader = csv.reader(data_file)

                        num_changes = sum(1 for row in file_reader) - 1
                            
                        row = {'x': x_val, 'y': y_val, 'num_changes': num_changes}
                        rows.append(row)
                        
                        csv_count += 1

        to_df = pd.DataFrame(rows).set_index(['y', 'x'])
    
        dataset = xr.Dataset.from_dataframe(to_df)
        
        dataset = dataset.sortby("y", ascending=False)
        
        print("Generating plot...")
        
        x_size = len(dataset.x.values)
        y_size = len(dataset.y.values)
        
        aspect = x_size / y_size # Get aspect ratio for plots, since area is not always square
        
        plt.rcParams.update({'font.size': 22})

        # Create plot
        dataset['num_changes'].plot(cmap=plt.cm.OrRd, aspect=aspect, size=20)        
        
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig("change_intensity_map.png")
        
        print("Generating KEA file...")
        
        kea_change_map = "intensity_map.kea"
        x_min = np.amin(dataset.x.values)
        y_max = np.amax(dataset.y.values)
    
        geo_transform = (x_min, args.cell_size, 0.0, y_max, 0.0, -args.cell_size)
        
        driver = gdal.GetDriverByName('KEA')
        pred_raster = driver.Create(kea_change_map, x_size, y_size, 1, 2) # Only one band
            
        if(args.ref_img):
            
            print("Reference image: {}".format(args.ref_img))
            
            img_name = args.ref_img
        
            # Get CRS of the input image
            orig_img = gdal.Open(img_name)
            srs = orig_img.GetProjection()
            
            print("CRS from input image is {}".format(srs))
            
            orig_img = None
            
        elif(args.coord_system):
            
            print("CRS is {}".format(args.coord_system))
            
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(args.coord_system)
            srs = srs.ExportToWkt()
            
        else:
            print("No CRS specified.")
            srs = None
                
        pred_raster.SetProjection(srs)
        pred_raster.SetGeoTransform(geo_transform)
            
        raster_band = pred_raster.GetRasterBand(1)
        raster_band.SetNoDataValue(0)
        raster_band.SetDescription("num_changes")
        dataset['num_changes'] = dataset['num_changes'].fillna(args.nodata_val)
        band_min = np.amin(dataset['num_changes'].values)
        band_max = np.amax(dataset['num_changes'].values)
        band_mean = np.mean(dataset['num_changes'].values)
        band_sd = np.std(dataset['num_changes'].values)
        raster_band.SetStatistics(band_min, band_max, band_mean, band_sd)
        raster_band.WriteArray(dataset['num_changes'].values)
            
        pred_raster = None
        
        if(args.ref_img):
            
            print("Processing reference image...")
 
            # Name shapefile
            shapefile = "change_shapefile.shp"
            
            # Create shapefile
            subprocess.call(["gdaltindex", shapefile, kea_change_map])
            
            # Crop provided image to fit the analysed area
            crop_name = img_name.replace(".kea", "_cropped.kea")       
            subprocess.call(["gdalwarp", "-of", "kea", "-cutline", shapefile, "-crop_to_cutline", img_name, crop_name])
             
            # Create false colour image to compare with change map
            orig_img = gdal.Open(img_name)
            
            band4 = orig_img.GetRasterBand(4)
            band5 = orig_img.GetRasterBand(5)
            band6 = orig_img.GetRasterBand(6)
                 
            band4_stats = band4.GetStatistics(True, True)
            band5_stats = band5.GetStatistics(True, True)
            band6_stats = band6.GetStatistics(True, True)
            
            # Close image
            orig_img = None
            
            band4_min = str(band4_stats[0])
            band4_max = str(band4_stats[1])
            
            band5_min = str(band5_stats[0])
            band5_max = str(band5_stats[1])
            
            band6_min = str(band6_stats[0])
            band6_max = str(band6_stats[1])
            
            # Name cropped false colour file
            false_colour_cropped = img_name.replace(".kea", "_false_colour_cropped.png")
            
            subprocess.call(["gdal_translate", crop_name, "-b", "6", "-b", "5", "-b", "4", "-of", "png", "-ot", "byte", "-scale_1", band6_min, band6_max, "-scale_2", band5_min, band5_max, "-scale_3", band4_min, band4_max, "-a_nodata", "255", false_colour_cropped])
    
            # Draw area on to full size false colour image
            cropped_img = gdal.Open(crop_name)
            ulx, xres, xskew, uly, yskew, yres  = cropped_img.GetGeoTransform()
            
            lrx = ulx + (cropped_img.RasterXSize * xres)
            lry = uly + (cropped_img.RasterYSize * yres)
            
            # Close image
            cropped_img = None
            
            # create ogr geometry
            linelyr = ogr.Geometry(ogr.wkbLinearRing)
            linelyr.AddPoint(ulx, lry)
            linelyr.AddPoint(lrx, lry)
            linelyr.AddPoint(lrx, uly)
            linelyr.AddPoint(ulx, uly)
            linelyr.AddPoint(ulx, lry)
            
            outline_shape = "outline.shp"
            
            # Create the shapefile
            driver = ogr.GetDriverByName("Esri Shapefile")
            ds = driver.CreateDataSource(outline_shape)
            
            layr1 = ds.CreateLayer('', None, ogr.wkbLineString)
                     
            # create the field
            layr1.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
            
            # Create the feature and set values
            defn = layr1.GetLayerDefn()
            feat = ogr.Feature(defn)
            feat.SetField('id', 1)
            feat.SetGeometry(linelyr)
            layr1.CreateFeature(feat)
                        
            # close the shapefile
            ds.Destroy()
            
            buffered = "buffered_lines.shp"
            
            # Buffer shapefile to make it more visible 
            subprocess.call(["ogr2ogr", "-dialect", "SQLite", "-sql", "SELECT ST_Buffer(geometry,360) FROM outline", buffered, outline_shape])
            
            # Create name for full size false colour image
            false_colour = img_name.replace(".kea", "_false_colour.jpg")
            
            subprocess.call(["gdal_rasterize", "-burn", "0", "-at", "-b", "6", "-b", "5", "-b", "4", buffered, img_name])
    
            subprocess.call(["gdal_translate", img_name, "-b", "6", "-b", "5", "-b", "4", "-of", "jpeg", "-ot", "byte", "-scale_1", band6_min, band6_max, "-scale_2", band5_min, band5_max, "-scale_3", band4_min, band4_max, "-a_nodata", "255", "-outsize", "50%", "50%", false_colour])
           
            print("Cleaning up...")
            
            # Clean up shape files
            crop_files  = shapefile.replace("shp", "*")
            outline_files = outline_shape.replace("shp", "*")
            buffered_files = buffered.replace("shp", "*")
    
            for file in os.listdir("."):
                if re.search(crop_files, file) or re.search(outline_files, file) or re.search(buffered_files, file):
                    os.remove(file)                         
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate change map from CCDC output.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-csv', '--csv_dir', required=True, help="The directory containing the CSV files to be processed.")
    parser.add_argument('-ref', '--ref_img', required=True, help="Reference image to show the area that has been processed. Assumed to be in the same CRS as the change output.")
    parser.add_argument('-nodat', '--nodata_val', type=float, required=True, help="Value for pixels with invalid/no data.")
    parser.add_argument('-cell', '--cell_size', required=True, type=int, help="Spatial resolution, e.g. 30 for Landsat.")
    parser.add_argument('-crs', '--coord_system', type=int, help="EPSG code for the CRS of the data. Enter as a number, e.g. 32655")

    args = parser.parse_args()
    
    main(args)



