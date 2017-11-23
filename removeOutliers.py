import numpy as np
import statsmodels.api as sm
from datetime import datetime
import statsmodels.formula.api as smf
import pandas as pd

class RLMRemoveOutliers(object):

    def __init__(self):
        
        self.T = 365
        self.pi_val = (2 * np.pi) / self.T
        self.pi_val_change = None
        self.band2_model = None
        self.band4_model = None
        self.band5_model = None
    
    def clean_data(self, pixel_data, num_years):
        
        """Extracts the Band 2, band 4, and Band 5 data, which are used to build RLMs for detecting cloud and snow outliers"""
        self.pi_val_change = (2 * np.pi) / (num_years * self.T)

        # Get coefficients for the three models
        self.band2_model = self.makeRLMModel(pixel_data[['datetime', 'band_2']])
        self.band4_model = self.makeRLMModel(pixel_data[['datetime', 'band_4']])
        self.band5_model = self.makeRLMModel(pixel_data[['datetime', 'band_5']])

        #predictions = fcoeffs.predict()
        #print(predictions, band2_only)

        #print(fcoeffs.params)
        #print(self.band2_coeffs)

        #outliers = self.removeBadPixels(outlier_bands)
    
        # Remove outliers from data
        #pixel_data = np.delete(pixel_data, outliers, axis=0)
    
        return pixel_data
    

    def makeRLMModel(self, band_data):
        
        """Builds the model and stores the coefficients"""
        
        band_data.columns = ['datetime', 'reflectance']

        rlm_model = smf.rlm('reflectance ~ 1 + (np.cos(self.pi_val * datetime) + np.sin(self.pi_val * datetime)) + (np.cos(self.pi_val_change * datetime)) + (np.sin(self.pi_val_change * datetime))', band_data)
        rlm_result = rlm_model.fit(maxiter=5)
        
        return rlm_result
    
    def removeBadPixels(self, band_data):
        
        """Goes through each observation and makes a list of outliers"""
        
        outliers = []
    
        for index, row in enumerate(band_data):

            # Get B2 delta
            b2_predicted = self.band2_coeffs[0] + (self.band2_coeffs[1] * np.cos(self.pi_val * row[0])) + (self.band2_coeffs[2] * np.sin(self.pi_val * row[0])) + (self.band2_coeffs[3] * np.cos(self.pi_val_change * row[0])) + (self.band2_coeffs[4] * np.sin(self.pi_val_change * row[0]))

            b2_delta = row[1] - b2_predicted

            if(b2_delta > 400):
                outliers.append(index)

            else:
                # Get B4 delta
                b4_predicted = self.band4_coeffs[0] + (self.band4_coeffs[1] * np.cos(self.pi_val * row[0])) + (self.band4_coeffs[2] * np.sin(self.pi_val * row[0])) + (self.band4_coeffs[3] * np.cos(self.pi_val_change * row[0])) + (self.band4_coeffs[4] * np.sin(self.pi_val_change * row[0]))

                b4_delta = row[2] - b4_predicted

                if(b4_delta < -400):

                    # Get B5 delta
                    b5_predicted = self.band5_coeffs[0] + (self.band5_coeffs[1] * np.cos(self.pi_val * row[0])) + (self.band5_coeffs[2] * np.sin(self.pi_val * row[0])) + (self.band5_coeffs[3] * np.cos(self.pi_val_change * row[0])) + (self.band5_coeffs[4] * np.sin(self.pi_val_change * row[0]))

                    b5_delta = row[3] - b5_predicted

                    if(b5_delta < -400):
                        outliers.append(index)

        return outliers











