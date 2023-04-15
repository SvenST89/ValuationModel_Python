#=============SET UP LOGGING ======================#
import logging
import sys
# specifically for pyplot: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger('matplotlib').disabled = True

logger=logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)

# set logging handler in file
fileHandler=logging.FileHandler(filename="log/cvm_main.log", mode='w')
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
#===================================================#

#===================================================#
# NECESSARY IMPORTS
#===================================================#
#--- PRELIMINARY IMPORTS -------------------------------------#
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import json

# To make web requests at the ECB SDMX 2.1 RESTful web service
import requests

# For use of the ECB API
#from sdw_api import SDW_API

# Standard libs for data manipulation
import numpy as np
import scipy.stats as st
import io
import datetime
from datetime import date
import re
import ValuationModel
from ValuationModel.assist_functions import *
from ValuationModel.fmp import *
from config.api import MY_API_KEY
from ValuationModel.arima import *
#---- DATABASE MANAGEMENT TOOLS --------------#
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras as extras

#---- DATA MANIPULATION TOOLS ----------------#
import pandas_datareader as dr

#---- OWN MODULE IMPORTS --------------------#
import config.pw

#---- STATISTICAL TOOLS FOR TIME SERIES ANALYSIS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

#===================================================#
# BUILD UP MODEL: ESTIMATE INTRINSIC VALUE
#===================================================#

def build_dcf_model(company='', year=2022, tax_rate=0.15825):
    #global company
    company=company # Currently, I have just a limited amount of companies in the database. The companies are: GitLab Inc., INFINEON TECHNOLOGIES AG, E.ON SE, Palantir Technologies Inc.,
  
    companyTicker_dict = json.load(open('ticker_dict.json', 'r'))

    #global ticker
    ticker=companyTicker_dict[company]

    #=== DATABASE CONNECTION
    # Set necessary url variables for the sqlalchemy create_engine() method.
    user='svenst89' # or default user 'postgres'
    password=config.pw.password # edit the password if you switch to the default user 'postgres'; I setup different passwords.
    host='localhost'
    port='5432'
    database='fundamentalsdb'
    # Create an engine object as medium for database exchange with PostgreSQL
    global engine
    def run_engine():
        return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
    try:
        engine=run_engine()
        logger.info(f"You have successfully created an engine object for your Postgres DB at {host} for user {user}!")
    except Exception as ex:
        logger.info("Sorry your engine has not been created. Some exception has occurred. Please check and try it again!\n", ex)

    #=== MAKE DICTIONARIES FOR LATER USE
    # Retrieve company data for sector-stock matching
    sql="""SELECT * FROM company"""
    try:
        company_df=pd.read_sql(sql, engine)
        pd.set_option('display.expand_frame_repr', False)
        logger.info("You have successfully retrieved the company data!")
    except Exception as ex:
        logger.info("Sorry! Something went wrong! I guess, there is no data available. Have you stored the data in the database?\n"+ ex)


    # Creating a dictionary with the economic sectors with their respective stocks. Thereafter we'll convert it into
    # a json file so we don't need to read the data source page multiple times.
    subdf=company_df[["symbol", "sector"]]

    stocks_sectors_dict=subdf.to_dict(orient='list')
    #print(stocks_sectors_dict)

    # Implementing the economic sector names as the dictionary key and their stocks as values.
    d = {}
    for stock, sector in zip(stocks_sectors_dict['symbol'], stocks_sectors_dict['sector']):
        if sector not in d.keys():
            d[sector] = [stock]
        else:
            d[sector].append(stock)

    # Converting it into a json file
    with open("sector_stocks.json", "w") as outfile:
        json.dump(d, outfile)

    #=== Stock-currency Mapping
    subdf_curr=company_df[["symbol", "currency"]]

    stocks_curr_dict=subdf_curr.to_dict(orient='list')
    #print(stocks_sectors_dict)

    # Implementing the stock names as the dictionary key and their currency as values.
    d_curr = {}
    for stock, curr in zip(stocks_curr_dict['symbol'], stocks_curr_dict['currency']):
        if curr not in d_curr.keys():
            d_curr[curr] = [stock]
        else:
            d_curr[curr].append(stock)

    # Converting it into a json file
    with open("stocks_curr.json", "w") as outfile_curr:
        json.dump(d_curr, outfile_curr)

    #===Stock-Exchange Mapping
    subdf_stex=company_df[["symbol", "exchangeshortname"]]

    stocks_exch_dict=subdf_stex.to_dict(orient='list')
    #print(stocks_sectors_dict)

    # Implementing the stock names as the dictionary key and their currency as values.
    d_stex = {}
    for stock, exchange in zip(stocks_exch_dict['symbol'], stocks_exch_dict['exchangeshortname']):
        if exchange not in d_stex.keys():
            d_stex[exchange] = [stock]
        else:
            d_stex[exchange].append(stock)

    # Converting it into a json file
    with open("stocks_exchange.json", "w") as outfile_exchange:
        json.dump(d_stex, outfile_exchange)


    # Loading the dashboard's json files.
    global sector_stocks, stocks_curr, stocks_exch
    sector_stocks = json.load(open('sector_stocks.json', 'r'))
    stocks_curr = json.load(open('stocks_curr.json', 'r'))
    stocks_exch=json.load(open('stocks_exchange.json', 'r'))
    #======================================================================#

    #=== AT DEBT COST
    # Key List
    keys_list=pd.read_excel('assets/keys_IR.xlsx')['Keys'].tolist()
    # Set start period for time series as a string value, e.g., '2019-12-01' 
    start='2022-01-01'
    interest_df=get_ir_data(keys_list, start)
    interest_df.to_csv("data/interest_df.csv")
    debt_cost=(interest_df['OBS_VALUE'].iloc[-1])/100
    # German Corporate Tax rate (incl. solidarity surcharge) = 15.825%
    tax=tax_rate
    at_debt_cost=debt_cost*(1-tax)

    #=== MARKET RISK PREMIUM
    index_table=get_index_table()
    index_table.to_csv("data/index_list_overview.csv")
    #'^GSPC', '^FTSE', '^NDX', '^RUA', '^NYA'] # Dax Performance40, S&P500, FTSE100, Nasdaq100, Dow Jones
    # Retrieve Data on the DAX Performance Index (i.e. the 40 stocks which I have in my database)
    # Get most recent week’s minute data
    today = date.today()        # get the date of today
    today_formatted = today.strftime("%Y-%m-%d")
    dax_perf_prices = YahooFinancials('^GDAXI').get_historical_price_data('2010-01-01', end_date=today_formatted, time_interval='daily')
    #dax_perf_prices = pd.DataFrame.from_dict(pd.json_normalize(dax_perf_prices), orient="columns")
    entry=extract_json(dax_perf_prices, ['prices'])
    entry=entry[0]['prices']
    index=pd.DataFrame(entry)
    #dax=get_price_table('DAX', json_entry='historical')
    dax_perf_closep=index[['close']]
    #--- ANNUALIZE THE DAILY RETURNS --------------------------------------------------------------#
    days=len(dax_perf_closep)
    # Total Return over the period
    total_return=(dax_perf_closep.iloc[-1] - dax_perf_closep.iloc[0]) / dax_perf_closep.iloc[0]
    annualized_return=((1+total_return)**(252/days))-1
    #--- RISK-FREE RATE = ECB MAIN REFINANCING RATE -----------------------------------------------#
    # Get ECB Main refinancing rate from SDW API.
    # ECB series key: FM.B.U2.EUR.4F.KR.MRR_FR.LEV
    ecb_r_key=['FM.B.U2.EUR.4F.KR.MRR_FR.LEV']
    ecb_rate_df=get_ir_data(ecb_r_key, start)
    rfr=float(ecb_rate_df['OBS_VALUE'].iloc[-1])/100
    mrp=float(annualized_return-rfr)
    
    #=== GET DATABASE DATA
    #===OPTIONAL: INSTEAD OF USING FUNCTION 'get_database_findata_year()' use the general function retrieving all available financial data and extracting a specific year from the whole dataframe
    bs, incs, cs = get_database_findata(company, engine)
    bs_y=clearDataframes_and_get_fy(company, bs, 'bs', year)
    incs_y=clearDataframes_and_get_fy(company, incs, 'incs', year)
    cs_y=clearDataframes_and_get_fy(company, cs, 'cs', year)

    #=== CALCULATE WACC
    # The equity required rate of return is calculated and Beta is retrieved inside this function!
    wacc=get_wacc(company, year, rfr, mrp, at_debt_cost, engine)
    logger.info(f'WACC for {company} is: {wacc}')

    #=== CALCULATE NECESSARY FINANCIAL KPIS
    #=== REVENUES: Get the revenues over the last years
    rev_df=incs.loc[incs['item']=='revenue'].drop_duplicates(['date', 'value'])
    revenues=pd.DataFrame(rev_df['value'])#.values
    #date=pd.DataFrame(rev_df['date'])
    comp_revs=revenues.set_index(rev_df['date'])#.drop_duplicates(['value'])
    comp_revs.to_csv("data/revenue.csv")
    #--- REVENUES: Get CAGR
    end_val=float(rev_df['value'].iloc[0])
    beg_val=float(rev_df['value'].iloc[-1])
    n=len(rev_df['date'])
    try:
        rev_cagr=((end_val/beg_val)**(1/n))-1
        logger.info(f'Revenue CAGR for {company} is: {rev_cagr}')
    except:
        logger.info("Something went wrong with CAGR calculation! Maybe there are some negative values or zero values at beginning or end?!")
    
    # Operating Income Margin average over the last years
    oir=incs.loc[incs['item']=='operatingIncomeRatio'].drop_duplicates(['date', 'value'])
    oir_mean=pd.to_numeric(oir['value']).mean() # first convert string values in column to numeric float
    oi=(float(rev_df['value'].iloc[0]))*oir_mean
    logger.info(f'Operating Income Ratio Mean for {company} is: {oir_mean}')

    #=== Depreciation & Amortization (D&A)
    comp_dna=prepare_timeseries(incs, item='depreciationAndAmortization')
    #=== CAPEX
    comp_cex=prepare_timeseries(cs, item='capitalExpenditure')
    #=== Change in Working Capital (CWC)
    comp_cwc=prepare_timeseries(cs, item='changeInWorkingCapital')

    #===================================================#
    # STATISTICAL TIME SERIES FORECASTING OF NECESSARY INPUT VARIABLES
    #===================================================#
    #=== DEPRECIATION & AMORTIZATION / DnA
    #  Start with D&A Prediction: Prepare the dataframe with simple "value" column and index from 0 to X
    df_dna=comp_dna[['value']].sort_values('date', ascending=True).reset_index().drop(['date'], axis=1)
    dna_sarima, dna_model_summary = build_sarima(df_dna, test_method='adf', frequency=3)
    #--- Forecast
    dna_fc_series=sarima_forecast(dna_sarima, df_dna, num_periods=5, company=company)
    dna_fc_list=dna_fc_series.tolist()

    #=== CAPEX
    df_cex=comp_cex[['value']].sort_values('date', ascending=True).reset_index().drop(['date'], axis=1)
    capex_sarima, summary_cex=build_sarima(df_cex)
    #--- Forecast
    capex_fc_series=sarima_forecast(capex_sarima, df_cex, num_periods=5, company=company)
    capex_fc_list=capex_fc_series.tolist()

    #=== CHANGE IN WORKING CAPITAL / CWC
    df_cwc=comp_cwc[['value']].sort_values('date', ascending=True).reset_index().drop(['date'], axis=1)
    cwc_sarima, summary_cwc=build_sarima(df_cwc)
    #--- Forecast
    cwc_fc_series=sarima_forecast(cwc_sarima, df_cwc, num_periods=5, company=company)
    cwc_fc_list=cwc_fc_series.tolist()

    #=== NET DEBT
    mask_nd=bs_y['item'].values=='netDebt'
    net_debt=round(float(bs_y.loc[mask_nd, 'value'])/1000,0)
    #=== CASH
    mask_cash=bs_y['item'].values=='cashAndShortTermInvestments'
    cash=round(float(bs_y.loc[mask_cash, 'value'])/1000, 0)

    #===================================================#
    # BUILDING THE DCF MODEL
    #===================================================#
    # Long-term growth rate, g
    g=(wacc-0.025)
    # Now, make forecast list for UNLEVERED FREE CASHFLOW through which the random variables as defined above, 
    # i.e. the variables and their items which calculate unlevered free cashflow, will flow
    def forecast_ufcf(last_rev, rev_cagr, margin_mean, tax, dna_fc_list, capex_fc_list, cwc_fc_list, wacc, g, multiple=8):
        forecast_lst=[]
        for i in range(len(dna_fc_list)):
            if i < len(dna_fc_list)-1:
                ufcf_t=round(float(((last_rev*(1+rev_cagr)**(i+1))*margin_mean)*(1-tax)+dna_fc_list[i]/1000+capex_fc_list[i]/1000-cwc_fc_list[i]/1000)/1000,0)
                forecast_lst.append(ufcf_t)
            else :
                tv_pg=round(float(((((last_rev*(1+rev_cagr)**(i))*margin_mean)*(1-tax)+dna_fc_list[i]+capex_fc_list[i]-cwc_fc_list[i])*(1+g))/(wacc-g))/1000,0)
                ebitda_last=float(((last_rev*(1+rev_cagr)**(i))*margin_mean)+dna_fc_list[i])
                tv_multiple=round(float(ebitda_last*multiple)/1000,0)
                tv=(tv_pg+tv_multiple)/2
                forecast_lst.append(tv)
        return forecast_lst
    #=== FREE CASHFLOW FORECAST
    forecast=forecast_ufcf(end_val, rev_cagr, oir_mean, tax, dna_fc_list, capex_fc_list, cwc_fc_list, wacc, g)

    #=== INTRINSIC VALUE
    # Now calculate the Present Value of the Free Cash Flows and the terminal value
    # Function to discount the free cash flows and the terminal value
    # I changed the function a bit as compared to https://towardsdatascience.com/company-valuation-using-probabilistic-models-with-python-712e325964b7
    # because I calculated the terminal value already before in the forecast list!
    def get_pv(forecast, wacc):
        discount_lst = []
        for x,i in enumerate(forecast):
            if x < 5:
                discount_lst.append(i/(1+wacc)**(x+1))
            else:
                discount_lst.append(i*(1/(1+wacc)**5))
        return sum(discount_lst)
    intrinsic_value=round(float(get_pv(forecast, wacc)+cash-net_debt),0)

    #=== ASSESSMENT OF OUTCOME
    outstanding_shares=get_profile_data(ticker, json_entry='outstandingShares', entry_point='shares')/1000
    current_share_price=real_time_stockprice(ticker, json_entry='price')
    intrinsic_share_price=round(intrinsic_value/outstanding_shares,2)

    if current_share_price > intrinsic_share_price:
        logger.info(f'The company {company} seems to be overrated on the market. The market price exceeds the estimated intrinsic equity price per share:\n--------------------------\nMarket Price: {current_share_price}\nIntrinsic Share Price: {intrinsic_share_price}\n--------------------------')
    else:
        logger.info(f'The company {company} seems to be underrated on the market. The market price is below the estimated intrinsic equity price per share:\n--------------------------\nMarket Price: {current_share_price}\nIntrinsic Share Price: {intrinsic_share_price}\n--------------------------')

    #===================================================#
    # RUN MONTE CARLO SIMULATION
    #===================================================#
    '''----// Run simulation: Monte Carlo //----'''
    iterations=1000
    hist_lst = []
    # the rolling averages remain  the same; just change the input parameters like cagrs, operating margin mean, wacc and g (long-term growth)
    for i in range(iterations):
        cagr = np.random.normal(rev_cagr, 0.01)
        margin = np.random.normal(oir_mean, 0.005)
        long_term_rate = np.random.normal(g, 0.001)
        discount_rate = np.random.normal(wacc, 0.001)
        forecast = forecast_ufcf(end_val, cagr, margin, tax, dna_fc_list, capex_fc_list, cwc_fc_list, discount_rate, long_term_rate)
        hist_lst.append(round(float(get_pv(forecast, discount_rate)+cash-net_debt),0))
    hist_array = np.array(hist_lst)

    mean = hist_array.mean() # mean of the sampled point estimates
    standard_error = hist_array.std()/(iterations**(1/2)) # sample standard error

    lower_bound = mean-1.96*standard_error
    upper_bound = mean+1.96*standard_error

    #=== MAKE HISTOGRAM OF MC SIMULATION
    CHART_THEME='plotly_white'
    curr=[key for key, tickers_list in stocks_curr.items() if ticker in tickers_list][0]

    mc_fig=go.Figure()
    mc_fig.add_trace(go.Histogram(x=hist_array/1000000, name=f'Intrinsic Valuation: {company}',marker_color='#EB89B5'))
    mc_fig.layout.template=CHART_THEME
    mc_fig.update_layout(
        # title={
        # 'text': f'Intrinsic EV - Monte Carlo Simulation: {company}',
        # 'y':0.95,
        # 'x': 0.5,
        # 'xanchor':'center',
        # 'yanchor': 'top'
        # },
        xaxis=dict(
            title=f"Intrinsic Enterprise Value in {curr}bn",
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=14,
                color='rgb(136, 136, 138)',
            ),
        ),
        yaxis=dict(
            title="Count",
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            zeroline=False,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=14,
                color='rgb(136, 136, 138)',
            ),
        ),
        margin=dict(
            b=50,
            l=25,
            r=25,
            t=50
        ),
        # plot_bgcolor='black',
        # paper_bgcolor='black',
        # font_color='grey',
        # autosize=True
    )
    return mc_fig, intrinsic_value, intrinsic_share_price