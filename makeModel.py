import numpy as np
import statsmodels.formula.api as smf

class MakeCCDCModel(object):

    def __init__(self, band_data):
        
        self.T = 365.25
        self.pi_val_simple = (2 * np.pi) / self.T
        self.pi_val_advanced = (4 * np.pi) / self.T
        self.pi_val_full = (6 * np.pi) / self.T
        self.band_data = band_data
        
        self.lasso_model = None
        self.RMSE = None
        self.coefficients = None

    def fitModel(self, model_num):
        
        """Finds the coefficients by fitting a Lasso model to the data"""
        
        if(model_num == 6 or model_num == 12):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * datetime) + np.sin(self.pi_val_simple * datetime) + datetime', self.band_data)
        
        elif(model_num == 18):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * datetime) + np.sin(self.pi_val_simple * datetime) + np.cos(self.pi_val_advanced * datetime) + np.sin(self.pi_val_advanced * datetime) + datetime', self.band_data)
        
        elif(model_num == 24):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * datetime) + np.sin(self.pi_val_simple * datetime) + np.cos(self.pi_val_advanced * datetime) + np.sin(self.pi_val_advanced * datetime) + np.cos(self.pi_val_full * datetime) + np.sin(self.pi_val_full * datetime) + datetime', self.band_data)
        
        self.lasso_model = lasso_model.fit_regularized(alpha=0.001, maxiter=5, L1_wt = 1.0)
        self.band_data['predicted'] = self.lasso_model.predict()
    
        self.RMSE = np.sqrt(np.mean((self.band_data['predicted'] - self.band_data['reflectance']) ** 2))
        
        self.coefficients = self.lasso_model.params
        
    def getPrediction(self, date_to_predict):
    
        """Returns a predicted value for a give date based on the current model"""
    
        return self.lasso_model.predict({'datetime': [date_to_predict]})
        
    def getCoefficients(self):
        
        """Returns the list of coefficients for this model"""
        
        if(self.coefficients.any()):
            return self.coefficients

    def getRMSE(self):
        
        """Returns the RMSE value, which is used to find change in the model"""
    
        if(self.RMSE != None):
            return self.RMSE

    def getBandData(self):

        return self.band_data

    def getNumCoeffs(self):
		
        return len(self.coefficients)











