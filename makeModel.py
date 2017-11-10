import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.api as sm

class MakeCCDCModel(object):

    def __init__(self):
        
        self.coefficients = None
        self.T = 365
        self.two_pi_div_T = (2 * np.pi) / self.T
        self.four_pi_div_T = (4 * np.pi) / self.T
        self.six_pi_div_T = (2 * np.pi) / self.T
        self.RMSE = None

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

        a2i = np.cos(self.four_pi_div_T * julian_dates)
        terms.append(a2i)

        b2i = np.sin(self.four_pi_div_T * julian_dates)
        terms.append(b2i)

        a3i = np.cos(self.six_pi_div_T * julian_dates)
        terms.append(a3i)

        b3i = np.sin(self.six_pi_div_T * julian_dates)
        terms.append(b3i)
    
        terms = np.array(terms).T
        
        lasso_model = sm.OLS(reflectance, terms)
        lasso_results = lasso_model.fit_regularized(method='elastic_net', alpha=0.01, L1_wt=1.0)
        self.coefficients = lasso_results.params
        
        predicted_vals = [self.get_predicted(row) for row in julian_dates]
    
        self.RMSE = np.sqrt(np.mean(((predicted_vals - reflectance) ** 2)))
        
    def get_predicted(self, julian_date):
        
        """Returns the predicted value for a given julian date based on the model coefficients from OLS"""

        new_pixel = self.coefficients[0] + (self.coefficients[1]*(np.cos(self.two_pi_div_T * julian_date))) + (self.coefficients[2]*(np.sin(self.two_pi_div_T * julian_date))) + (self.coefficients[3]*julian_date) + (self.coefficients[4]*(np.sin(self.four_pi_div_T * julian_date))) + (self.coefficients[5]*(np.sin(self.four_pi_div_T * julian_date))) + (self.coefficients[6]*(np.cos(self.six_pi_div_T * julian_date)))+ (self.coefficients[7]*(np.sin(self.six_pi_div_T * julian_date)))
            
        return new_pixel
    
    def get_coefficients(self):
        
        """Returns the list of coefficients for this model"""
        
        if(self.coefficients.any()):
            return self.coefficients

    def get_rmse(self):
        
        """Returns the RMSE value, which is used to find change in the model"""
    
        if(self.RMSE != None):
            return self.RMSE












