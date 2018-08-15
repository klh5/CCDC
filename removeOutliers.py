'''
Implements Tmask according to the paper: Automated cloud, cloud shadow, and snow detection in multitemporal Landsat data: An algorithm designed specifically for monitoring land cover change (2014) by Z. Zhu and C. Woodcock.
'''

import numpy as np
import statsmodels.api as sm

class RLMRemoveOutliers(object):

    def __init__(self, toa_data, sref_data):
        
        self.T = 365.25
        self.pi_val = (2 * np.pi) / self.T
        self.toa_data = toa_data
        self.sref_data = sref_data
        
        self.pi_val_change = None
        self.green_model = None
        self.nir_model = None
        self.swir1_model = None
        self.b2_delta = None
        self.b4_delta = None
        self.b5_delta = None
    
    def cleanData(self, num_years):
        
        """Uses the Band 2, band 4, and Band 5 data to build RLMs for detecting cloud and snow outliers"""
        self.pi_val_change = (2 * np.pi) / (num_years * self.T)
        start_date = self.toa_data[:,0].min()
        
        rescaled = self.toa_data[:,0] - start_date

        # Get coefficients for the three models
        self.green_model, self.b2_delta = self.makeRLMModel(rescaled, self.toa_data[:,1])
        self.nir_model, self.b4_delta = self.makeRLMModel(rescaled, self.toa_data[:,2])    
        self.swir1_model, self.b5_delta = self.makeRLMModel(rescaled, self.toa_data[:,3])             
        
        self.toa_data = np.hstack((self.toa_data, self.b2_delta.reshape(-1, 1)))
        self.toa_data = np.hstack((self.toa_data, self.b4_delta.reshape(-1, 1)))
        self.toa_data = np.hstack((self.toa_data, self.b5_delta.reshape(-1, 1)))
        
        self.dropOutliers()
    
        return self.sref_data
    
    def makeRLMModel(self, datetimes, band_data):
        
        """Builds the model and stores the coefficients"""
        
        x = np.array([np.ones_like(datetimes), # Add constant 
                      np.cos(self.pi_val * datetimes),
                      np.sin(self.pi_val * datetimes),
                      np.cos(self.pi_val_change * datetimes),
                      np.sin(self.pi_val_change * datetimes)]).T       
        
        rlm_model = sm.RLM(band_data, x, M=sm.robust.norms.TukeyBiweight(c=0.4685))
        rlm_result = rlm_model.fit(maxiter=5)
        
        delta = band_data - rlm_result.predict(x)
               
        return rlm_result, delta
    
    def dropOutliers(self):
              
        self.sref_data = self.sref_data[(self.toa_data[:,4] < 40) & ((self.toa_data[:,5] > -40) | (self.toa_data[:,6] > -40))]
