import numpy as np
from sklearn import linear_model

class MakeCCDCModel(object):

    def __init__(self, band_data, band):
        
        self.T = 365.25
        self.pi_val_simple = (2 * np.pi) / self.T
        self.pi_val_advanced = (4 * np.pi) / self.T
        self.pi_val_full = (6 * np.pi) / self.T
        self.band_data = band_data
        self.band = band
        
        self.lasso_model = None
        self.RMSE = None
        self.coefficients = None
        self.start_date = self.band_data.datetime.min()

    def fitModel(self, model_num):
        
        """Finds the coefficients by fitting a Lasso model to the data"""
        rescaled = self.band_data.datetime - self.start_date
        
        x = np.array([rescaled,
                      np.cos(self.pi_val_simple * rescaled),
                      np.sin(self.pi_val_simple * rescaled)])

        if(model_num >= 24):
            x = np.vstack((x, np.array([np.cos(self.pi_val_advanced * rescaled),
                      np.sin(self.pi_val_advanced * rescaled)])))
    
        if(model_num >= 30):
            x = np.vstack((x, np.array([np.cos(self.pi_val_full * rescaled),
                      np.sin(self.pi_val_full * rescaled)])))
    
        x = x.T

        clf = linear_model.Lasso(fit_intercept=True, alpha=0.001, max_iter=50)

        self.lasso_model = clf.fit(x, self.band_data.reflectance)
        
        self.band_data['predicted'] = self.lasso_model.predict(x)
    
        self.RMSE = np.sqrt(np.mean((self.band_data.reflectance - self.band_data.predicted) ** 2))

        self.coefficients = self.lasso_model.coef_ 

    def getPrediction(self, date_to_predict):
    
        """Returns the predicted value for a given date based on the current model"""
        
        # Rescale date so that it starts from 0
        date_to_predict = date_to_predict - self.start_date
        
        x = np.array([[date_to_predict],
                      [np.cos(self.pi_val_simple * date_to_predict)],
                      [np.sin(self.pi_val_simple * date_to_predict)]])

        if(self.getNumCoeffs() >= 5):
            x = np.vstack((x, np.array([[np.cos(self.pi_val_advanced * date_to_predict)],
                      [np.sin(self.pi_val_advanced * date_to_predict)]])))
    
        if(self.getNumCoeffs() >= 7):
            x = np.vstack((x, np.array([[np.cos(self.pi_val_full * date_to_predict)],
                      [np.sin(self.pi_val_full * date_to_predict)]])))
    
        x = x.T
        return self.lasso_model.predict(x.reshape(1,-1))
        
    def getCoefficients(self):
        
        """Returns the list of coefficients for this model"""
        
        if(self.coefficients.any()):
            return self.coefficients

    def getRMSE(self):
        
        """Returns the RMSE value, which is used to find change in the model"""
    
        if(self.RMSE != None):
            return self.RMSE
        
    def getReflectance(self):

        return self.band_data.reflectance

    def getPredicted(self):

        return self.band_data.predicted
    
    def getDateTimes(self):

        return self.band_data.datetime

    def getNumCoeffs(self):
		
        return len(self.coefficients)
    
    def getBandData(self):
        
        return self.band_data

    def getBandName(self):
		
        return self.band







