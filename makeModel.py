import numpy as np
from sklearn import linear_model
from datetime import datetime

class MakeCCDCModel(object):

    def __init__(self, datetimes, init_obs, band):
        
        self.T = 365.25
        self.pi_val_simple = (2 * np.pi) / self.T
        self.pi_val_advanced = (4 * np.pi) / self.T
        self.pi_val_full = (6 * np.pi) / self.T
        self.datetimes = datetimes
        self.band = band
        
        self.doy = np.array([datetime.fromordinal(x.astype(int)).timetuple().tm_yday for x in self.datetimes])
        self.lasso_model = None
        self.residuals = None
        self.RMSE = None
        self.coefficients = None
        self.predicted = None
        self.start_val = None
        self.end_val = None
        self.alpha = None
        self.init_obs = init_obs

    def fitModel(self, band_data, cv, alpha):
        
        self.start_val = band_data[0]
        self.end_val = band_data[-1]
        
        """Finds the coefficients by fitting a Lasso model to the data"""
        rescaled = self.datetimes - self.getMinDate()
        
        x = np.array([rescaled,
                      np.cos(self.pi_val_simple * rescaled),
                      np.sin(self.pi_val_simple * rescaled)])

        if(self.init_obs >= 18):
            x = np.vstack((x, np.array([np.cos(self.pi_val_advanced * rescaled),
                      np.sin(self.pi_val_advanced * rescaled)])))
    
        if(self.init_obs >= 24):
            x = np.vstack((x, np.array([np.cos(self.pi_val_full * rescaled),
                      np.sin(self.pi_val_full * rescaled)])))
    
        x = x.T

        if(cv): # If cross validation should be used to find alpha parameter
            self.lasso_model = linear_model.LassoCV(fit_intercept=True).fit(x, band_data)
            self.alpha = self.lasso_model.alpha_
        else:
            self.lasso_model = linear_model.Lasso(fit_intercept=True, alpha=alpha).fit(x, band_data)
            self.alpha = alpha
                                
        self.predicted = self.lasso_model.predict(x)
        
        self.coefficients = self.lasso_model.coef_
        
        self.residuals = band_data - self.predicted
    
        # Get overall RMSE of model
        self.RMSE = np.sqrt(np.mean(self.residuals ** 2))
            
    def getPrediction(self, date_to_predict):
    
        """Returns the predicted value for a given date based on the current model"""
        
        # Rescale date so that it starts from 0
        date_to_predict = date_to_predict - self.getMinDate()
        
        x = np.array([[date_to_predict],
                      [np.cos(self.pi_val_simple * date_to_predict)],
                      [np.sin(self.pi_val_simple * date_to_predict)]])

        if(self.init_obs >= 18):
            x = np.vstack((x, np.array([[np.cos(self.pi_val_advanced * date_to_predict)],
                      [np.sin(self.pi_val_advanced * date_to_predict)]])))
    
        if(self.init_obs >= 24):
            x = np.vstack((x, np.array([[np.cos(self.pi_val_full * date_to_predict)],
                      [np.sin(self.pi_val_full * date_to_predict)]])))
    
        x = x.T
        return self.lasso_model.predict(x.reshape(1,-1))
    
    def getAdjustedRMSE(self, curr_date):
        
        """Get adjusted RMSE for a specific DOY"""
        
        # Get DOY for current date
        curr_doy = datetime.fromordinal(curr_date.astype(int)).timetuple().tm_yday
               
        # Get absolute differences between the current DOY and all other DOY values
        differenced = np.abs(self.doy - curr_doy)

        # Sort differenced values and return index
        sorted_ix = sorted(range(len(differenced)), key=lambda k: differenced[k])

        # Get closest values by index
        closest = sorted_ix[:self.init_obs]
        
        # Subset residuals by indices of closest DOY values
        closest_residuals = self.residuals[closest]
    
        # Calculate adjusted RMSE
        adjusted_rmse = np.sqrt(np.mean(closest_residuals ** 2))
        
        return adjusted_rmse
    
    def getRMSE(self, curr_date):
        
        return self.getAdjustedRMSE(curr_date)
                  
    def getMinDate(self):
        
        return np.min(self.datetimes)
    
    def getMaxDate(self):
        
        return np.max(self.datetimes)

    def getNumCoeffs(self):
		
        return len(self.coefficients)
    








