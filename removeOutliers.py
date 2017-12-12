import numpy as np
from datetime import datetime
import statsmodels.formula.api as smf
import statsmodels.api as sm
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
        self.band2_model = self.makeRLMModel(pixel_data[['time', 'green']])
        self.band4_model = self.makeRLMModel(pixel_data[['time', 'nir']])
        self.band5_model = self.makeRLMModel(pixel_data[['time', 'swir1']])

        outliers = self.removeBadPixels(pixel_data)
    
        # Remove outliers from data
        pixel_data = pixel_data.drop(outliers)
        pixel_data = pixel_data.reset_index(drop=True)
        
        return pixel_data
    
    def makeRLMModel(self, band_data):
        
        """Builds the model and stores the coefficients"""
        
        band_data.columns = ['time', 'reflectance']
        
        rlm_model = smf.rlm('reflectance ~ np.cos(self.pi_val * time) + np.sin(self.pi_val * time) + np.cos(self.pi_val_change * time) + np.sin(self.pi_val_change * time)', band_data, M=sm.robust.norms.TukeyBiweight(c=0.4685))
        rlm_result = rlm_model.fit(maxiter=5)
        
        return rlm_result
    
    def removeBadPixels(self, band_data):
        
        """Goes through each observation and makes a list of outliers"""
        
        outliers = []
        
        band_data['band_2_pred'] = self.band2_model.predict()
        band_data['band_4_pred'] = self.band4_model.predict()
        band_data['band_5_pred'] = self.band5_model.predict()
    
        for index, row in band_data.iterrows():

            # Get B2 delta
            b2_delta = row['green'] - row['band_2_pred']
            
            if(b2_delta > 400):
                outliers.append(index)

            else:
                # Get B4 delta
                b4_delta = row['nir'] - row['band_4_pred']
                
                if(b4_delta < -400):
                    # Get B5 delta
                    b5_delta = row['swir1'] - row['band_5_pred']
                    
                    if(b5_delta < -400):
                        outliers.append(index)

        return outliers











