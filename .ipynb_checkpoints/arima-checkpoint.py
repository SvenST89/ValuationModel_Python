import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

def build_sarima(df, test_method='adf', frequency=3):
    # Seasonal - fit stepwise auto-ARIMA
    sarima = pm.auto_arima(df, start_p=1, start_q=1,
                         test=test_method, # test for stationarity
                         max_p=3, max_q=3, m=frequency,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
    return sarima, sarima.summary()

def sarima_forecast(model, df, num_periods=5, company=''):
    # FORECAST
    n_periods=num_periods
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)
    
    # make series for plotting purposes
    fc_series=pd.Series(fc, index=index_of_fc)
    lower_series=pd.Series(confint[:, 0], index=index_of_fc)
    upper_series=pd.Series(confint[:, 1], index=index_of_fc)
    
    # Plot
    plt.plot(df.value)
    plt.plot(fc_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)
    plt.title(f'{company}: Final {num_periods}-year Forecast of Depreciation & Amortization')
    plt.xticks(rotation=90)
    plt.show()
    return fc_series