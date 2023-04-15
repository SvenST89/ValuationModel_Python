import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
#from statsmodels.tsa.arima.model import ARIMA #, ARMA
import pmdarima as pm
import scipy.stats as st

#============================#
# ----- LOGGING -------------'
import logging

logger=logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
#============================#


def build_sarima(df, test_method='adf', frequency=3):
    if df.shape[0]<15:
        logger.info('Calculating a simple moving average model as sample time series data is too short.')
        len_df=df.shape[0]
        global tain_data
        global test_data
        msk = np.random.rand(len(df)) < 0.8
        train_data = df[msk]
        test_data = df[~msk]
        sma_model = ARIMA(train_data.value, order=(0, 0, 1))
        model = sma_model.fit() # fit already returns ARMAResults on which we directly can call '.predict': https://www.statsmodels.org/v0.11.0/generated/statsmodels.tsa.arima_model.ARMA.fit.html
        model_summary=model.summary()
    else:
        try:
            # Seasonal - fit stepwise auto-ARIMA
            model = pm.auto_arima(df, start_p=1, start_q=1,
                                 test=test_method, # test for stationarity
                                 max_p=3, max_q=3, m=frequency,
                                 start_P=0, seasonal=True,
                                 d=None, D=1, trace=True,
                                 error_action='ignore',  
                                 suppress_warnings=True, 
                                 stepwise=True)
            model_summary=model.summary()
        except ValueError as e:
            logger.info(f'The following exception has occurred:\n{e}\n==> I will try another SARIMA Model config.')
            model = pm.auto_arima(df, start_p=1, start_q=1,
                                 test=test_method, # test for stationarity
                                 max_p=3, max_q=3, m=1,
                                 start_P=0, seasonal=True,
                                 d=None, D=0, trace=True,
                                 error_action='ignore',  
                                 suppress_warnings=True, 
                                 stepwise=True)
            model_summary=model.summary()
        
    return model, model_summary

def sarima_forecast(model, df, num_periods=5, company=''):
    n_periods=num_periods
    index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)
    if df.shape[0]<15:
        fc_df=pd.DataFrame(model.predict(start=1, end=n_periods))#, index=df_dna['date'].values)
        fc_df.set_index(index_of_fc, inplace=True)
        fc_series=fc_df.squeeze()
        # calculate confidence interval values manually and get it into a 2D array
        standard_error = st.sem(fc_series) # sample standard error
        bigl=[]
        # calculate own confidence interval values!
        for i in fc_series.tolist():
            l=[]
            # To find critical values for the t-distribution (in cases where n<30, we should not assume normality, hence go with the t-dist instead of the norm-dist and z-scores!)
            # we can use scipy: https://stackoverflow.com/questions/67340028/how-to-use-t-ppf-which-are-the-arguments
            # and https://www.geeksforgeeks.org/how-to-find-the-t-critical-value-in-python/
            # Explanation of PPF: Probability Point Function
            # The Probability Point Function or PPF is the inverse of the CDF. 
            # Specifically, the PPF returns the exact point where the probability of everything to the left is equal to y. 
            # This can be thought of as the percentile function since the PPF tells us the value of a given percentile of the data.
            t_critical_val=st.t.ppf(q=(1-.05)/2,df=n_periods-1) # assuming 95% CI, two-tailed test
            lvalue=i-(t_critical_val*standard_error)
            uvalue=i+(t_critical_val*standard_error)
            l.append(lvalue)
            l.append(uvalue)
            bigl.append(l)
        confint=np.array(bigl)
    else:
        # FORECAST
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        # make series for plotting purposes
        fc_series=pd.Series(fc, index=index_of_fc)
    lower_series=pd.Series(confint[:, 0], index=index_of_fc)
    upper_series=pd.Series(confint[:, 1], index=index_of_fc)
    
    return fc_series

def plot_autocorr(df, diff=2):
    plt.rcParams.update({'figure.figsize':(9,7)})
    
    if diff == 2:
        logger.info('You´ve chosen two orders of Differencing')
        fig, axes = plt.subplots(3, 2)
        # Original series
        axes[0,0].plot(df.value); axes[0,0].set_title('Original Series', size=10); axes[0,0].tick_params(axis='x', colors='white')
        plot_acf(df.value, ax=axes[0,1])
        
        # 1st Differencing
        axes[1,0].plot(df.value.diff()); axes[1,0].set_title('$1^{st} Order Difference$', size=10); axes[1,0].tick_params(axis='x', colors='white')
        plot_acf(df.value.diff().dropna(), ax=axes[1,1])
        
        # 2nd Differencing
        axes[2,0].plot(df.value.diff().diff()); axes[2,0].set_title('$2^{nd} Order Difference$', size=10); axes[2,0].tick_params(axis='x', colors='white')
        plot_acf(df.value.diff().diff().dropna(), ax=axes[2,1])
        
        plt.show()
        
    elif diff == 3:
        logger.info('You´ve chosen three orders of Differencing')
        fig, axes = plt.subplots(4, 2)
        # Original series
        axes[0,0].plot(df.value); axes[0,0].set_title('Original Series', size=10); axes[0,0].tick_params(axis='x', colors='white')
        plot_acf(df.value, ax=axes[0,1])
        
        # 1st Differencing
        axes[1,0].plot(df.value.diff()); axes[1,0].set_title('$1^{st} Order Difference$', size=10); axes[1,0].tick_params(axis='x', colors='white')
        plot_acf(df.value.diff().dropna(), ax=axes[1,1])
        
        # 2nd Differencing
        axes[2,0].plot(df.value.diff().diff()); axes[2,0].set_title('$2^{nd} Order Difference$', size=10); axes[2,0].tick_params(axis='x', colors='white')
        plot_acf(df.value.diff().diff().dropna(), ax=axes[2,1])
        
        # 3rd Differencing
        axes[3,0].plot(df.value.diff().diff().diff()); axes[3,0].set_title('$3^{rd} Order Difference$', size=10)
        plot_acf(df.value.diff().diff().diff().dropna(), ax=axes[3,1])
        
        plt.show()
        
    else:
        logger.info('Now, it´s getting crazy! You´ve chosen more than three orders of Differencing. I will plot differencing autocorrelation plots up until order 5. Check your time series!')
        fig, axes = plt.subplots(6, 2)
        # Original series
        axes[0,0].plot(df.value); axes[0,0].set_title('Original Series', size=10); axes[0,0].tick_params(axis='x', colors='white')
        plot_acf(df.value, ax=axes[0,1])
        
        # 1st Differencing
        axes[1,0].plot(df.value.diff()); axes[1,0].set_title('$1^{st} Order Difference$', size=10); axes[1,0].tick_params(axis='x', colors='white')
        plot_acf(df.value.diff().dropna(), ax=axes[1,1])
        
        # 2nd Differencing
        axes[2,0].plot(df.value.diff().diff()); axes[2,0].set_title('$2^{nd} Order Difference$', size=10); axes[2,0].tick_params(axis='x', colors='white')
        plot_acf(df.value.diff().diff().dropna(), ax=axes[2,1])
        
        # 3rd Differencing
        axes[3,0].plot(df.value.diff().diff().diff()); axes[3,0].set_title('$3^{rd} Order Difference$', size=10)
        plot_acf(df.value.diff().diff().diff().dropna(), ax=axes[3,1])
        
        # 4th Differencing
        axes[3,0].plot(df.value.diff().diff().diff().diff()); axes[3,0].set_title('$4^{th} Order Difference$', size=10)
        plot_acf(df.value.diff().diff().diff().diff().dropna(), ax=axes[3,1])
        
        # 5th Differencing
        axes[3,0].plot(df.value.diff().diff().diff().diff().diff()); axes[3,0].set_title('$5^{th} Order Difference$', size=10)
        plot_acf(df.value.diff().diff().diff().diff().diff().dropna(), ax=axes[3,1])
    
        plt.show()

def plot_partautocorr(df, diff=2):
    plt.rcParams.update({'figure.figsize':(9,7)})
    
    if df.shape[0]<15:
        logger.info('Dataframe is too short, i.e. sample size is too small for a partial autocorrelation plot! Continue without a plot!')
    else:
    
        if diff == 2:
            logger.info('You´ve chosen two orders of Differencing')
            fig, axes = plt.subplots(3, 2)
            # Original series
            axes[0,0].plot(df.value); axes[0,0].set_title('Original Series', size=10); axes[0,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value, ax=axes[0,1])
            
            # 1st Differencing
            axes[1,0].plot(df.value.diff()); axes[1,0].set_title('$1^{st} Order Difference$', size=10); axes[1,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value.diff().dropna(), ax=axes[1,1])
            
            # 2nd Differencing
            axes[2,0].plot(df.value.diff().diff()); axes[2,0].set_title('$2^{nd} Order Difference$', size=10); axes[2,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value.diff().diff().dropna(), ax=axes[2,1])
            
            plt.show()
            
        elif diff == 3:
            logger.info('You´ve chosen three orders of Differencing')
            fig, axes = plt.subplots(4, 2)
            # Original series
            axes[0,0].plot(df.value); axes[0,0].set_title('Original Series', size=10); axes[0,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value, ax=axes[0,1])
            
            # 1st Differencing
            axes[1,0].plot(df.value.diff()); axes[1,0].set_title('$1^{st} Order Difference$', size=10); axes[1,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value.diff().dropna(), ax=axes[1,1])
            
            # 2nd Differencing
            axes[2,0].plot(df.value.diff().diff()); axes[2,0].set_title('$2^{nd} Order Difference$', size=10); axes[2,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value.diff().diff().dropna(), ax=axes[2,1])
            
            # 3rd Differencing
            axes[3,0].plot(df.value.diff().diff().diff()); axes[3,0].set_title('$3^{rd} Order Difference$', size=10)
            plot_pacf(df.value.diff().diff().diff().dropna(), ax=axes[3,1])
            
            plt.show()
            
        else:
            logger.info('Now, it´s getting crazy! You´ve chosen more than three orders of Differencing. I will plot differencing autocorrelation plots up until order 5. Check your time series!')
            fig, axes = plt.subplots(6, 2)
            # Original series
            axes[0,0].plot(df.value); axes[0,0].set_title('Original Series', size=10); axes[0,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value, ax=axes[0,1])
            
            # 1st Differencing
            axes[1,0].plot(df.value.diff()); axes[1,0].set_title('$1^{st} Order Difference$', size=10); axes[1,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value.diff().dropna(), ax=axes[1,1])
            
            # 2nd Differencing
            axes[2,0].plot(df.value.diff().diff()); axes[2,0].set_title('$2^{nd} Order Difference$', size=10); axes[2,0].tick_params(axis='x', colors='white')
            plot_pacf(df.value.diff().diff().dropna(), ax=axes[2,1])
            
            # 3rd Differencing
            axes[3,0].plot(df.value.diff().diff().diff()); axes[3,0].set_title('$3^{rd} Order Difference$', size=10)
            plot_pacf(df.value.diff().diff().diff().dropna(), ax=axes[3,1])
            
            # 4th Differencing
            axes[3,0].plot(df.value.diff().diff().diff().diff()); axes[3,0].set_title('$4^{th} Order Difference$', size=10)
            plot_pacf(df.value.diff().diff().diff().diff().dropna(), ax=axes[3,1])
            
            # 5th Differencing
            axes[3,0].plot(df.value.diff().diff().diff().diff().diff()); axes[3,0].set_title('$5^{th} Order Difference$', size=10)
            plot_pacf(df.value.diff().diff().diff().diff().diff().dropna(), ax=axes[3,1])
        
            plt.show()

# Accuracy metrics
def performance_metrics(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    #me = np.mean(forecast - actual)             # ME
    #mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    res_dict={'mape':mape,  'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax}
    return res_dict

def hyperparam_tuning(df, d=2, max_order=3):
    try:
        # initialize a performance dataframe to store values of hyperparameter tuning
        perf_df=pd.DataFrame({'mape':0, 'AR_p':0, 'MA_q':0}, index=range(0,1))
            
        for i in range(1,int(max_order)):
            for j in range(1,int(max_order)):
                # Cross-validate manually with train/test split
                msk = np.random.rand(len(df)) < 0.8
                train = df[msk]
                test = df[~msk]
                # Build Model  
                model = ARIMA(train, order=(int(i),int(d),int(j)))
                #model.initialize_approximate_diffuse()
                fitted = model.fit()
                #print(fitted.summary())
                # Create confidence interval
                conf_int = fitted.conf_int() # default 9
                # Forecast
                fc = fitted.forecast(15)
                #i += 1
                #j += 1
                # Performance Measure
                res_dict=forecast_accuracy(fc, test.values)
                mape=res_dict['mape']
                # performance dictionary
                perf_dict={}
                perf_dict.update({'mape':mape})
                perf_dict.update({'AR_p':int(i)})
                perf_dict.update({'MA_q':int(j)})
                # Update performance dataframe
                new_row = perf_dict
                perf_df = perf_df.append(new_row, ignore_index=True)
             
        # ignore first row with zero values due to initialization
        perf_df = perf_df.iloc[1:]
        # get min mape values and related parameters
        min_mape_row=perf_df.loc[(perf_df['mape']==perf_df['mape'].min())] #perf_df[perf_df.mape == df.mape.min()]
        p=min_mape_row.iloc[:, 1].values[0]
        q=min_mape_row.iloc[:, 2].values[0]
        logger.info(f'\n Best AR parameter p: {p}\n Best MA parameter q: {q}\n')
        return min_mape_row, p, q
    except ValueError as e:
        logger.info(f'The following exception has occurred:\n{e}\n==> Dataframe is too short, i.e. sample size is too small! Continue without hyperparameter tuning. Below a simple moving average model will be calculated!')
        return None, None, None


def make_qqplot(df):
    m = df.value.mean()
    st = df.value.std()

    # Standardize the data
    df_norm=df.copy()
    for i in range(0,df.shape[0],1):
        df_norm.value.iloc[i]=(df.value.iloc[i]-m)/st

    value_min=int(df_norm['value'].min())
    value_max=int(df_norm['value'].max())
    q=[]
    j=0
    for i in range(1,df_norm.shape[0]+1,1):
        j=i/df_norm.shape[0]
        q_temp = np.quantile(df_norm['value'], j)
        q.append(q_temp)
    #fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(q,sorted(df_norm.value),'o')
    plt.plot(list(range(value_min,value_max)), list(range(value_min,value_max)), color='red')
    plt.xlabel("Theoretical Quantile of standard normal distribution")
    plt.ylabel("Sample Z-score / Quantiles")
    plt.title("Normal Q-Q Plot")
    plt.legend(loc='upper left', prop={'size': 9})
    plt.grid()

def diagnostic_plt(sarima_model, df):
    if df.shape[0]<15:
        # see arima.py: if the time series length is too short, I am buildin a simple moving average model with ARMA that has a different syntax for the residuals.
        resid=pd.DataFrame(sarima_model.resid, columns=['value'])
    else:
        # make model residuals to dataframe
        resid=pd.DataFrame(sarima_model.resid(), columns=['value'])
    try:
        sarima_model.plot_diagnostics(figsize=(7,5))
        plt.show()
    except AttributeError as a:
        print(f'The following exception has occurred:\n{a}\n==> I will make individual diagnostic plots.')
        plt.rcParams.update({'figure.figsize':(7,5)})
        # Q-Q Plot
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        fig=make_qqplot(resid)
        # seaborn histogram
        plt.subplot(1, 2, 2) # row 1, col 2 index 1
        sns.distplot(resid, hist=True, kde=True, bins=3, color = 'blue', hist_kws={'edgecolor':'black'})
        #plt.legend(prop={'size': 16}, title = 'Method')
        plt.title('Histogram plus estimated density')
        plt.ylabel('Density')
        plt.legend(loc='upper left', prop={'size': 9})
        plt.show()
    except ValueError as e:
        print(f'The following exception has occurred:\n{e}\n==> I will make individual diagnostic plots.')
        plt.rcParams.update({'figure.figsize':(7,5)})
        # Q-Q Plot
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        fig=make_qqplot(resid)
        # seaborn histogram
        plt.subplot(1, 2, 2) # row 1, col 2 index 1
        sns.distplot(resid, hist=True, kde=True, bins=3, color = 'blue', hist_kws={'edgecolor':'black'})
        #plt.legend(prop={'size': 16}, title = 'Method')
        plt.title('Histogram plus estimated density')
        plt.ylabel('Density')
        plt.legend(loc='upper left', prop={'size': 9})
        plt.show()
        