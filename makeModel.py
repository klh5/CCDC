import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.formula.api as smf

class MakeCCDCModel(object):

    def __init__(self, band_data):
        
        self.T = 365
        self.pi_val_simple = (2 * np.pi) / self.T
        self.pi_val_advanced = (4 * np.pi) / self.T
        self.pi_val_full = (6 * np.pi) / self.T
        self.band_data = band_data
        
        self.lasso_model = None
        self.RMSE = None
        self.coefficients = None

    def fit_model(self, model_num):
        
        """Finds the coefficients by fitting an OLS model to the data"""
        
        if(model_num == 6 or model_num == 12):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * datetime) + np.sin(self.pi_val_simple * datetime) + datetime', self.band_data)
        
        elif(model_num == 18):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * datetime) + np.sin(self.pi_val_simple * datetime) + np.cos(self.pi_val_advanced * datetime) + np.sin(self.pi_val_advanced * datetime) + datetime', self.band_data)
        
        elif(model_num == 24):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * datetime) + np.sin(self.pi_val_simple * datetime) + np.cos(self.pi_val_advanced * datetime) + np.sin(self.pi_val_advanced * datetime) + np.cos(self.pi_val_full * datetime) + np.sin(self.pi_val_full * datetime) + datetime', self.band_data)
        
        self.lasso_model = lasso_model.fit_regularized(method='elastic_net', alpha=0.01, L1_wt=1.0)
        self.band_data['predicted'] = self.lasso_model.predict()
    
        self.RMSE = np.sqrt(np.mean(((self.band_data['predicted'] - self.band_data['reflectance']) ** 2)))
        
        self.coefficients = self.lasso_model.params
        
    def get_prediction(self, date_to_predict):
    
        """Returns a predicted value for a give date based on the current model"""
    
        return self.lasso_model.predict({'datetime': [date_to_predict]})
        
    def get_coefficients(self):
        
        """Returns the list of coefficients for this model"""
        
        if(self.coefficients.any()):
            return self.coefficients

    def get_rmse(self):
        
        """Returns the RMSE value, which is used to find change in the model"""
    
        if(self.RMSE != None):
            return self.RMSE

    def get_band_data(self):

        return self.band_data












