import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.api as sm

class MakeCCDCModel(object):

    def __init__(self):
        
        self.coefficients = None
        self.T = 365
        self.two_pi_div_T = (2 * np.pi) / self.T
        self.three_times_RMSE = None

    def fit_model(self, reflectance, julian_dates):
        
        """Finds the coefficients by fitting an OLS model to the data"""

        terms = []

        a0i = reflectance / reflectance
        terms.append(a0i)

        a1i = np.cos(self.two_pi_div_T * julian_dates)
        terms.append(a1i)
    
        b1i = np.sin(self.two_pi_div_T * julian_dates)
        terms.append(b1i)
    
        c1i = (julian_dates - julian_dates) + julian_dates
        terms.append(c1i)
    
        terms = np.array(terms).T
        
        ols_model = np.linalg.lstsq(terms, reflectance)
        self.coefficients = ols_model[0]

        RMSE = np.sqrt(ols_model[1])
        
        self.three_times_RMSE = RMSE * 3
        
    def get_predicted(self, julian_date):
        
        """Returns the predicted value for a given julian date based on the model coefficients from OLS"""
        
        new_pixel = self.coefficients[0] + (self.coefficients[1]*(np.cos(self.two_pi_div_T * julian_date))) + (self.coefficients[2]*(np.sin(self.two_pi_div_T * julian_date))) + (self.coefficients[3]*julian_date)
            
        return new_pixel
    
    def get_coefficients(self):
        
        """Returns the list of coefficients for this model"""
        
        if(self.coefficients.any()):
            return self.coefficients

    def get_multiplied_rmse(self):
        
        """Returns the multipled RMSE value, which is used to find change in the model"""
    
        if(self.three_times_RMSE != None):
            return self.three_times_RMSE












