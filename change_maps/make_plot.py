'''
Takes the CSV files output from CCDC, and outputs a change intensity map.
'''

import pandas as pd
import os
import re
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
        img_name = sys.argv[2]

        print("CSV file directory:", csv_dir)
        print("Image file directory:", img_name)
        
        if(os.path.isdir(csv_dir)):
            
            print("Gathering change data...")

            rows = []
            csv_count = 1
        
            # For every file in the directory
            for change_file in os.listdir(csv_dir):

                if fnmatch.fnmatch(change_file, '*.csv'): # Check if it's a CSV file
                    
                    print("Processing file: {} Count: {}".format(change_file, csv_count))
                    
                    sep_filename = change_file.replace('.', '_').split('_')

                    x_val = float(sep_filename[0])
                    y_val = float(sep_filename[2])

                    csv_file_path = os.path.join(csv_dir, change_file)

                    with open(csv_file_path, "r") as data_file: 
                       
                        file_reader = csv.reader(data_file)

                        num_changes = sum(1 for row in file_reader) - 1
                            
                        row = {'x': x_val, 'y': y_val, 'num_changes': num_changes}
                        rows.append(row)
                        
                        csv_count += 1

        to_df = pd.DataFrame(rows).set_index(['y', 'x'])
    
        dataset = xr.Dataset.from_dataframe(to_df)
        
        temp_img = "change_temp.nc" # Temporary file to hold change map without CRS
        change_map_crs = "change_map_crs.nc" # File name for change map with CRS
        
        print("Generating output...")

        # Create NetCDF file of the change map
        dataset.to_netcdf(temp_img, encoding={'num_changes': {'dtype': 'uint16', '_FillValue': 9999}})
        
        # Get CRS of the input image
        orig_img = gdal.Open(img_name)
        wkt = orig_img.GetProjection()
        
        print("CRS is {}".format(wkt))
        
        orig_img = None

        # Set CRS
        change_map = gdal.Open(temp_img)
        change_map = gdal.Translate(change_map_crs, change_map, outputSRS = wkt)
        os.remove(temp_img) # Remove change map with no CRS
        
        # Create plot
        dataset['num_changes'].plot(cmap=plt.cm.OrRd, figsize=(12,6))
        
        plt.tight_layout()
        plt.savefig("change_map_plot.png")
        
        # Name shapefile
        shapefile = "change_shapefile.shp"
        
        # Create shapefile
        gen_shapefile = subprocess.call(["gdaltindex", shapefile, change_map_crs])
        
        # Crop provided image to fit the analysed area
        crop_name = img_name.replace(".kea", "_cropped.kea")       
        cut_to_shape = subprocess.call(["gdalwarp", "-of", "kea", "-cutline", shapefile, "-crop_to_cutline", img_name, crop_name])
         
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
        
        make_false_colour_cropped = subprocess.call(["gdal_translate", crop_name, "-b", "6", "-b", "5", "-b", "4", "-of", "png", "-ot", "byte", "-scale_1", band6_min, band6_max, "-scale_2", band5_min, band5_max, "-scale_3", band4_min, band4_max, "-a_nodata", "255", false_colour_cropped])

        # Draw area on to full size false colour image
        cropped_img = gdal.Open(crop_name)
        ulx, xres, xskew, uly, yskew, yres  = cropped_img.GetGeoTransform()
        
        lrx = ulx + (cropped_img.RasterXSize * xres)
        lry = uly + (cropped_img.RasterYSize * yres)
        
        # Close image
        cropped_img = None
        
        from osgeo import ogr

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
        layr1 = ds.CreateLayer('',None, ogr.wkbLineString)
        
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
        buffer_shapefile = subprocess.call(["ogr2ogr", "-dialect", "SQLite", "-sql", "SELECT ST_Buffer(geometry,360) FROM outline", buffered, outline_shape])
        
        # Create name for full size false colour image
        false_colour = img_name.replace(".kea", "_false_colour.jpg")
        
        draw_outline = subprocess.call(["gdal_rasterize", "-burn", "0", "-at", "-b", "6", "-b", "5", "-b", "4", buffered, img_name])

        make_false_colour = subprocess.call(["gdal_translate", img_name, "-b", "6", "-b", "5", "-b", "4", "-of", "jpeg", "-ot", "byte", "-scale_1", band6_min, band6_max, "-scale_2", band5_min, band5_max, "-scale_3", band4_min, band4_max, "-a_nodata", "255", "-outsize", "50%", "50%", false_colour])
       
        print("Cleaning up...")
        
        # Clean up shape files
        crop_files  = shapefile.replace("shp", "*")
        outline_files = outline_shape.replace("shp", "*")
        buffered_files = buffered.replace("shp", "*")

        for file in os.listdir("."):
            if re.search(crop_files, file) or re.search(outline_files, file) or re.search(buffered_files, file):
                os.remove(file)
        
        
if __name__ == "__main__":

	main()



