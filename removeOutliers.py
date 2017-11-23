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
        self.band2_coeffs = None
        self.band4_coeffs = None
        self.band5_coeffs = None
    
    def clean_data(self, pixel_data, num_years):
        
        """Extracts the Band 2, band 4, and Band 5 data, which are used to build RLMs for detecting cloud and snow outliers"""
        self.pi_val_change = (2 * np.pi) / (num_years * self.T)
        
        # Extract list of dates
        julian_dates = pixel_data[:,0]

        # Extract band 2 data
        band2_only = pixel_data[:,1]

        # Extract band 4 data
        band4_only = pixel_data[:,3]

        # Extract band 5 data
        band5_only = pixel_data[:,4]

        # Get coefficients for the three models
        self.band2_coeffs = self.makeRLMModel(band2_only, julian_dates)
        self.band4_coeffs = self.makeRLMModel(band4_only, julian_dates)
        self.band5_coeffs = self.makeRLMModel(band5_only, julian_dates)

        outlier_bands = np.column_stack((julian_dates, band2_only, band4_only, band5_only))
        
        test_fmodel = np.column_stack((julian_dates, band2_only))
        p_test_fmodel = pd.DataFrame(data=test_fmodel[0:,0:], index=test_fmodel[:,0], columns=['julian_date', 'band2'])

        fmodel = smf.rlm('band2 ~ 1 + (np.cos(self.pi_val * julian_date) + np.sin(self.pi_val * julian_date)) + (np.cos(self.pi_val_change * julian_date)) + (np.sin(self.pi_val_change * julian_date))', p_test_fmodel)

        fcoeffs = fmodel.fit(maxiter=5)
        predictions = fcoeffs.predict()
        print(predictions, band2_only)

        print(fcoeffs.params)
        print(self.band2_coeffs)

        outliers = self.removeBadPixels(outlier_bands)
    
        # Remove outliers from data
        pixel_data = np.delete(pixel_data, outliers, axis=0)
    
        return pixel_data
    

    def makeRLMModel(self, band_data, julian_dates):
        
        """Builds the model and stores the coefficients"""

        terms = []

        a0 = band_data / band_data
        terms.append(a0)

        a1i = np.cos(self.pi_val * julian_dates)
        terms.append(a1i)
    
        b1i = np.sin(self.pi_val * julian_dates)
        terms.append(b1i)
    
        a2i = np.cos(self.pi_val_change * julian_dates)
        terms.append(a2i)
        
        b2i = np.sin(self.pi_val_change * julian_dates)
        terms.append(b2i)
        
        terms = np.array(terms).T
            
        rlm_model = sm.RLM(band_data, terms, M=sm.robust.norms.TukeyBiweight())
        
        # Paper suggests a maximum of 5 iterations
        coeff_list = rlm_model.fit(maxiter=5).params
        
        return coeff_list
    
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











