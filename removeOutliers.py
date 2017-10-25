import numpy as np
import statsmodels.api as sm
from jdutil import jd_to_date

class RLMRemoveOutliers(object):

    def __init__(self):
        
        self.T = 365
        self.two_pi_div_T = (2 * np.pi) / self.T
    
    def clean_data(self, pixel_data):
    
        # Extract list of dates
        julian_dates = pixel_data[:,0]
        
        # Get number of years
        num_years = jd_to_date(np.amax(julian_dates))[0] - jd_to_date(np.amin(julian_dates))[0]
        if(num_years == 0):
            num_years = 1

        # Extract band 2 data
        band2_ref = pixel_data[:,2]

        # Model band 2 data
        band2_coeffs = self.makeRLMModel(band2_ref, julian_dates, num_years)
        
        # Find band 2 outliers
        band2_data = np.column_stack((julian_dates, band2_ref))
        band2_outliers = self.removeBadPixels(band2_data, band2_coeffs, num_years, 2)
        
        # Extract band 5 data
        band5_ref = pixel_data[:,5]
        
        # Model band 5 data
        band5_coeffs = self.makeRLMModel(band5_ref, julian_dates, num_years)
        
        # Find band 5 outliers
        band5_data = np.column_stack((julian_dates, band5_ref))
        band5_outliers = self.removeBadPixels(band5_data, band5_coeffs, num_years, 5)
    
        # Remove outliers from data
        pixel_data = np.delete(pixel_data, band2_outliers, axis=0)
        pixel_data = np.delete(pixel_data, band5_outliers, axis=0)
    
        return pixel_data
    

    def makeRLMModel(self, band_data, julian_dates, N):

        terms = []
        
        two_pi_div_NT = (2 * np.pi) / (N * self.T)

        a0 = band_data / band_data
        terms.append(a0)

        a1i = np.cos(self.two_pi_div_T * julian_dates)
        terms.append(a1i)
    
        b1i = np.sin(self.two_pi_div_T * julian_dates)
        terms.append(b1i)
    
        a2i = np.cos(two_pi_div_NT * julian_dates)
        terms.append(a2i)
        
        b2i = np.sin(two_pi_div_NT * julian_dates)
        terms.append(b2i)
        
        terms = np.array(terms).T
            
        rlm_model = sm.RLM(band_data, terms, M=sm.robust.norms.HuberT())
        coeff_list = rlm_model.fit().params
        
        return coeff_list
    
    def removeBadPixels(self, band_data, coefficients, N, band_num):
        
        two_pi_div_NT = (2 * np.pi) / (N * self.T)
        outliers = []
    
        for index, row in enumerate(band_data):

            predicted_value = coefficients[0] + (coefficients[1] * np.cos(self.two_pi_div_T * row[0])) + (coefficients[2] * np.sin(self.two_pi_div_T * row[0])) + (coefficients[3] * np.cos(two_pi_div_NT * row[0])) + (coefficients[4] * np.sin(two_pi_div_NT * row[0]))
            
            difference = row[1] - predicted_value

            if(band_num == 2):
                if(difference > 400):
                    outliers.append(index)
        
            else:
                if(difference < -400):
                    outliers.append(index)
    
        return outliers











