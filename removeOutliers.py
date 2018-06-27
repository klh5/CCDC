'''
Implements Tmask according to the paper: Automated cloud, cloud shadow, and snow detection in multitemporal Landsat data: An algorithm designed specifically for monitoring land cover change (2014) by Z. Zhu and C. Woodcock.
'''

import numpy as np
import statsmodels.formula.api as smf
import statsmodels as sm

class RLMRemoveOutliers(object):

    def __init__(self):
        
        self.T = 365.25
        self.pi_val = (2 * np.pi) / self.T
        self.pi_val_change = None
        self.green_model = None
        self.nir_model = None
        self.swir1_model = None
    
    def findOutliers(self, pixel_data, num_years):
        
        """Extracts the Band 2, band 4, and Band 5 data, which are used to build RLMs for detecting cloud and snow outliers"""
        self.pi_val_change = (2 * np.pi) / (num_years * self.T)
        start_date = pixel_data.datetime.min()
        
        pixel_data['rescaled'] = pixel_data.datetime - start_date

        # Get coefficients for the three models
        self.green_model = self.makeRLMModel(pixel_data[['rescaled', 'green']])
        self.nir_model = self.makeRLMModel(pixel_data[['rescaled', 'nir']])
        self.swir1_model = self.makeRLMModel(pixel_data[['rescaled', 'swir1']])

        outliers = self.getOutlierList(pixel_data)
    
        return outliers
    
    def makeRLMModel(self, band_data):
        
        """Builds the model and stores the coefficients"""
        
        band_data.columns = ['datetime', 'reflectance']
        
        rlm_model = smf.rlm('reflectance ~ np.cos(self.pi_val * datetime) + np.sin(self.pi_val * datetime) + np.cos(self.pi_val_change * datetime) + np.sin(self.pi_val_change * datetime)', band_data, M=sm.robust.norms.TukeyBiweight(c=0.4685))
        rlm_result = rlm_model.fit(maxiter=5)
        
        return rlm_result
    
    def getOutlierList(self, band_data):
        
        """Goes through each observation and makes a list of outliers"""

        outliers = []
        
        band_data['b2_delta'] = band_data.green - self.green_model.predict()
        band_data['b4_delta'] = band_data.nir - self.nir_model.predict()
        band_data['b5_delta'] = band_data.swir1 - self.swir1_model.predict()
    
        for index, row in band_data.iterrows():
            
            if(row.b2_delta > 40.0):
                outliers.append(index)
            
            else:
                if(row.b4_delta < -40.0 and row.b5_delta < -40.0):
                    outliers.append(index)

        return outliers










