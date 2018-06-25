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
        self.start_date = self.band_data.datetime.min()

    def fitModel(self, model_num):
        
        """Finds the coefficients by fitting a Lasso model to the data"""
        
        # Rescale date so that it starts from 0
        self.band_data['rescaled'] = self.band_data.datetime - self.start_date
        
        if(model_num == 6 or model_num == 12):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * rescaled) + np.sin(self.pi_val_simple * rescaled) + rescaled', self.band_data)
        
        elif(model_num == 18):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * rescaled) + np.sin(self.pi_val_simple * rescaled) + np.cos(self.pi_val_advanced * rescaled) + np.sin(self.pi_val_advanced * rescaled) + rescaled', self.band_data)
        
        elif(model_num == 24):
            lasso_model = smf.ols('reflectance ~ np.cos(self.pi_val_simple * rescaled) + np.sin(self.pi_val_simple * rescaled) + np.cos(self.pi_val_advanced * rescaled) + np.sin(self.pi_val_advanced * rescaled) + np.cos(self.pi_val_full * rescaled) + np.sin(self.pi_val_full * rescaled) + rescaled', self.band_data)
        
        self.lasso_model = lasso_model.fit_regularized(alpha=10, L1_wt = 1.0, maxiter=50) # 50 is default for statsmodels
        self.band_data['predicted'] = self.lasso_model.predict()
    
        self.RMSE = np.sqrt(np.mean((self.band_data['reflectance'] - self.band_data['predicted']) ** 2))
        
        self.coefficients = self.lasso_model.params
 
    def getPrediction(self, date_to_predict):
    
        """Returns a predicted value for a give date based on the current model"""
        
        date_to_predict = date_to_predict - self.start_date
        
        return self.lasso_model.predict({'rescaled': [date_to_predict]})
        
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











