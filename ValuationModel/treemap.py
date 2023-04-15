# libraries to check certificates for SSL connection to urls
import certifi
import ssl
# libraries for webscraping, parsing and getting stock data
from urllib.request import urlopen, Request
import requests
from bs4 import BeautifulSoup
from fmp import *

# for plotting and data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px


def make_treemap():

    #tickers=pd.read_excel("data/ticker.xlsx", sheet_name="dax")
    # Ticker list
    ticker_list=['GTLB', 'PLTR', 'IFX.DE', 'EOAN.DE', 'SHL.DE', 'PAH3', 'NEL.OL', 'PLUG', 'DIS', 'KO', 'MMM', 'MSFT', 'SQ', 'ABNB', 'AMZN', 'EBAY', 'ORCL', 'PYPL', 'QCOM', 'SNOW', 'TEAM', 'TSLA', 'DBK.DE', 'PM']

    #ticker_list=tickers['ticker'].tolist()

    tickers['pctchange']=0
    # Get Sentiment of Stock
    for ticker in ticker_list:
        tickers.loc[tickers['ticker']==ticker, ['pctchange']]=stock_pctchange(ticker)

    #print(tickers.head())

    # USE my own Financial Modeling Prep API Access for getting price, sector, market cap and industry info for each ticker
    mktCap = []
    sectors = []
    industries = []
    prices = []
    for ticker in ticker_list:
        print(ticker)
        mktCap.append(get_profile_data(ticker, json_entry='mktCap'))
        prices.append(get_profile_data(ticker, json_entry='price'))
        sectors.append(get_profile_data(ticker, json_entry='sector'))
        industries.append(get_profile_data(ticker, json_entry='industry'))

    # Combine the Information Above and the Corresponding Tickers into a DataFrame
    # dictionary {'column name': list of values for column} to be converted to dataframe
    d = {'Sector': sectors, 'Industry': industries, 'Price': prices, 'mktCap': mktCap}
    # create dataframe from 
    df_info = pd.DataFrame(data=d, index = ticker_list).reset_index()
    df_info = df_info.rename(columns={"index": "ticker"})
    #print(df_info.head())

    df = pd.merge(tickers, df_info, how='inner', on='ticker')
    df = df.dropna()
    df['mktCap']=pd.to_numeric(df['mktCap'])
    df['company_name']='0'
    for ticker in ticker_list:
         df.loc[df['ticker']==ticker, ['company_name']]=get_profile_data(ticker, json_entry='companyName')

    df.reset_index()
    #print(df)
    CHART_THEME='plotly_white'
    # Check Plotly for treemap: https://plotly.com/python/treemaps/
    # group data into sectors at the highest level, breaks it down into industry, and then ticker, specified in the 'path' parameter
    # the 'values' parameter uses the value of the column to determine the relative size of each box in the chart
    # the color of the chart follows the percentage change
    # when the mouse is hovered over each box in the chart all info will all be shown
    # the color is red (#ff0000) for negative changes, black (#000000) for 0 changes and green (#00FF00) for positive changes
    fig = px.treemap(df, path=[px.Constant("PERFORMANCE OVERVIEW (pct. change)"), 'Sector', 'Industry', 'company_name'], values='mktCap',
                    color='pctchange', hover_data=['company_name'],
                    color_continuous_scale=['#FF0000', "#FEFEFE", '#00FF00'],
                    color_continuous_midpoint=0)

    #fig.data[0].customdata = df[['company_name', 'Price', 'pctchange']].round(2) # round to 3 decimal places
    #fig.data[0].texttemplate = "%{label}<br>%{customdata[2]}"
    fig.layout.template=CHART_THEME
    fig.layout.height=500
    fig.update_traces(textposition="middle center")
    fig.update_layout(margin=dict(b=50,l=25,r=25,t=50), font_size=16)

    #plotly.offline.plot(fig, filename='stock_pctchange.html') # this writes the plot into a html file and opens it
    #fig.show()
    return fig

#make_treemap()