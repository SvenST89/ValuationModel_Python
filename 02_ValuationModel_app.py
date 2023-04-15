#=============SET UP LOGGING ======================#
import logging
import sys
# specifically for pyplot: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger('matplotlib').disabled = True

logger=logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)

# set logging handler in file
fileHandler=logging.FileHandler(filename="log/app.log", mode='w')
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
#===================================================#

#===================================================#
# NECESSARY IMPORTS
#===================================================#
#--- PRELIMINARY IMPORTS -------------------------------------#
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import json

# To make web requests at the ECB SDMX 2.1 RESTful web service
import requests

# Standard libs for data manipulation
import numpy as np
import pandas as pd
import io
import datetime
from datetime import date
import re
#---- DATABASE MANAGEMENT TOOLS --------------#
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras as extras

#---- OWN MODULE IMPORTS --------------------#
import config.pw
import ValuationModel
from ValuationModel.assist_functions import *
from ValuationModel.fmp import *
from config.api import MY_API_KEY
from ValuationModel.dcf import build_dcf_model

import warnings
warnings.filterwarnings('ignore')

#====================================================#
# INITIAL PARAMETERS & CALCULATIONS
#====================================================#
#------------------------------------------------------------------------------------#
CHART_THEME='plotly_white'
# Loading the dashboard's json files stored before with dcf.py
#global sector_stocks, stocks_curr, stocks_exch
sector_stocks = json.load(open('sector_stocks.json', 'r'))
stocks_curr = json.load(open('stocks_curr.json', 'r'))
stocks_exch=json.load(open('stocks_exchange.json', 'r'))
#global ticker_list
tickerdict=json.load(open('ticker_dict.json', 'r'))
ticker_list=list(tickerdict.values())
#ticker_list=['GTLB', 'PLTR', 'IFX.DE', 'EOAN.DE', 'SHL.DE', 'PAH3', 'NEL.OL', 'PLUG', 'DIS', 'KO', 'MMM', 'MSFT', 'SQ', 'ABNB', 'AMZN', 'EBAY', 'ORCL', 'PYPL', 'QCOM', 'SNOW', 'TEAM', 'TSLA', 'DBK.DE', 'PM']

company='The Walt Disney Company' # Currently, I have just a limited amount of companies in the database. The companies are: GitLab Inc., INFINEON TECHNOLOGIES AG, E.ON SE, Palantir Technologies Inc.,

ticker=tickerdict[company]
curr=[key for key, ticker_list in stocks_curr.items() if ticker in ticker_list][0]

d = pd.DataFrame(ticker_list)
dd_labels = [{'label': d[0].unique()[i], 'value': d[0].unique()[i]} for i in range(d[0].unique().shape[0])]
print(dd_labels)
#------------------------------------------------------------------------------------#
def variation(stk):
    df = get_price_table(stk) # from my fmp-package
    return df['changePercent'].iloc[0] # the latest percentage change value is in first row in column 'changePercent' --> FMP delivers it! Easy!

carousel_prices = {}
for stock in ticker_list:
    # Calculating the stocks' price variation and storing it in the 'carousel_prices' dictionary.
    try:
        logger.info(f'Retrieving variation data for {stock}...')
        carousel_prices[stock] = variation(stock)/100
    except IndexError as IE:
        logger.info(f"{stock} obviously has no variation data available... \n {IE}!")
        continue
    

# Turning 'carousel_prices' into a json file.
with open('carousel_prices.json', 'w') as outfile:
    json.dump(carousel_prices, outfile)

# Loading the dashboard's json files.
carousel_prices = json.load(open('carousel_prices.json', 'r'))

# The standard stock displayed when the dashboard is initialized will be Infineon, IFX.DE.
standard_disp = get_price_table(ticker)

# 'standard_disp_variation' is stored inside the card that shows the stock's current price.
standard_disp_variation = standard_disp['changePercent'].iloc[0]

#====================================================#
# STOCK KPIs: Card below Dropdown
#====================================================#
# company
#comp=company
# real time stock price from FMP API
pri = 'Real Time Price: ' + str(real_time_stockprice(ticker, json_entry='price'))
# P/E Ratio
pe = 'P/E Ratio (ttm): ' + str(pe_ttm(ticker, json_entry='peRatioTTM'))
# Sector P/E Ratio
#sec_pe='Sector P/E Ratio: ' + str(sec_per(ticker, ticker_list, sector_stocks, stocks_exch, json_entry='pe'))


#====================================================#
# PREPARE FIGURES
#====================================================#

#===== HISTOGRAM: FROM VALUATION MODEL
latest_year=get_latest_available_fy(ticker=ticker, json_entry="date")
fig_mc, intrinsic_value, intrinsic_price=build_dcf_model(company=company, year=latest_year, tax_rate=0.15825)

#===== CANDLESTICK: PRICE CHART
# 'fig' exposes a candlestick chart with the prices of the stock since 2015.
candle = go.Figure()
candle.add_trace(go.Candlestick(x=standard_disp['date'],
                             open=standard_disp['open'],
                             close=standard_disp['close'],
                             high=standard_disp['high'],
                             low=standard_disp['low'],
                             name='Stock Price'))
candle.layout.template=CHART_THEME
candle.layout.height=500
candle.update_layout(
    xaxis=dict(
        title="Date",
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
        title="Stock Price",
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
)

# Setting the graph to display the current prices in a first moment. Nonetheless,the user can also manually ajust the zoom size
# either by selecting a section of the chart or using one of the time span buttons available.
# The default
min_date = '2020-01-01'
end=date.today()
max_date = end
candle.update_xaxes(range=[min_date, end])
candle.update_yaxes(tickprefix=curr+' ')


#====================================================#
# START STYLING OF APP
#====================================================#
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_trich_components as dtc
from dash.dependencies import Output, Input

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(
    [
        #=== ROW 0: offset, for styling purposes
        dbc.Row([
            dtc.Carousel([
                html.Div([
                    # Span showing name of the stock
                    html.Span(stock, style={'margin-left':'80px'}),
                    # Now, this span shows the variation of the stock
                    html.Span('{}{:.2%}'.format('+' if carousel_prices[stock] > 0 else '', carousel_prices[stock]), style={
                        'color': 'green' if carousel_prices[stock] > 0 else 'red', 'margin-left':'20px'})
                ]) for stock in sorted(carousel_prices.keys())
            ], id='main-carousel', autoplay=True, slides_to_show=5),
        ]),
        #=== ROW 1
        dbc.Row([
            # column 1
            dbc.Col([html.H2('STOCK ASSISTANT DASHBOARD', style={'margin-top': '12px', 'margin-left': '48px', 'margin-bottom': '16px'}, className='text-center text-primary, mb-3')], width={'size': 10, 'offset': 0, 'order': 0}), # the max size of a screen is width=12!
            dbc.Col([html.Img(src="assets/ds_logo_white2.png", style={'height': '70%', 'width':'30%', 'margin-top':'12px', 'margin-left':'16px'})], width={'size': 2, 'offset': 0, 'order': 0})
        ], justify='start'),
        #=== ROW 2
        dbc.Row([
            # column 1
            dbc.Col([
                html.H5('Select the stock you want to analyse:', style={'textAlign': 'left'}),
                dcc.Dropdown(
                    id='stock-dropdown',
                    options=dd_labels,
                    value=dd_labels[0]['label'],
                    #style={'height': 550, 'margin-bottom': '14px', 'margin-left':'12px'}),
                ),
                html.Div(id='company', className='text-center mt-3 p-2'),
                html.Div(id='sec', className='text-center p-2'),
                html.Div(id='pri', className='text-center p-2'),
                html.Div(id='intrinsic_price', className='text-center p-2'),
                html.Div(id='pe', className='text-center p-2'),
                #html.Div(id='sec_pe', className='text-center p-2'),
            ], width={'size': 4, 'offset': 0, 'order': 0}),
            # column 2
            dbc.Col([
                html.H5(id='desc_candle', className='text-center'),
                dcc.Graph(
                    id='candle',
                    figure=candle,
                    style={'height': 550, 'margin-bottom': '14px', 'margin-left':'12px'}),
                html.Hr(),
            ], width={'size': 8, 'offset': 0, 'order': 0}),
        ]),
        #=== ROW 3
        dbc.Row([
            # column 1
            dbc.Col([
                html.H5(id='desc_mc', className='text-center'),
                dcc.Graph(
                    id='fig_mc',
                    figure=fig_mc,
                    style={'height': 550, 'margin-bottom': '14px', 'margin-left':'12px'}),
                html.Hr(),
            ], width={'size': 4, 'offset': 0, 'order': 0}),
            # column 2
            # dbc.Col([
            #     html.H5('Key Financial KPIs', className='text-center'),
            #     dcc.Graph(
            #         id='fin_kpis',
            #         figure=kpi_tbl,
            #         style={'height': 550, 'margin-bottom': '14px', 'margin-left':'12px'}),
            #     html.Hr(),
            # ], width={'size': 6, 'offset': 0, 'order': 0}),
        ]),
    ], fluid=True
)

#====================================================#
# MAKE APP INTERACTIVE: CALLBACKS
#====================================================#
#Output("sec_pe", "children"),
@app.callback(
    [
    Output("candle", "figure"), Output("fig_mc", "figure"), Output("company", "children"), Output("sec", "children"), Output("pri", "children"), Output("intrinsic_price", "children"),
    Output("pe", "children"), Output("desc_candle", "children"), Output("desc_mc", "children")
    ],
    Input("stock-dropdown", "value")
)

def update_figures(st):
    #=== UPDATE CANDLESTICK CHART
    company=get_profile_data(st, json_entry='companyName', entry_point='profile')
    desc_candle=f'Price Chart: {company}'
    price_table = get_price_table(st)
    candle = go.Figure()
    candle.add_trace(go.Candlestick(x=price_table['date'],
                                open=price_table['open'],
                                close=price_table['close'],
                                high=price_table['high'],
                                low=price_table['low'],
                                name='Stock Price'))
    candle.layout.template=CHART_THEME
    candle.layout.height=500
    candle.update_layout(
        # title={
        # 'text': f'Price Chart: {company}',
        # 'y':0.95,
        # 'x': 0.5,
        # 'xanchor':'center',
        # 'yanchor': 'top'
        # },
        xaxis=dict(
            title="Date",
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
            title="Stock Price",
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
    )

    #=== HISTOGRAM: FROM VALUATION MODEL
    latest_year=get_latest_available_fy(st, json_entry="date")
    fig_mc, intrinsic_value, intrinsic_price=build_dcf_model(company=company, year=latest_year, tax_rate=0.15825)
    desc_mc=f'Intrinsic EV Monte Carlo Simulation: {company}'

    #=== Update Information below dropdown field
    company='Company: ' + company
    sec='Sector: ' + get_profile_data(st, json_entry='sector', entry_point='profile')
    pri='Real Time Price: ' + str(round(real_time_stockprice(st, json_entry='price'),2))
    intrinsic_price='Intrinsic Stock Price: ' + str(intrinsic_price)
    pe='P/E Ratio (ttm): ' + str(round(pe_ttm(st, json_entry='peRatioTTM'),2))
    sector_stocks = json.load(open('sector_stocks.json', 'r'))
    stocks_exch=json.load(open('stocks_exchange.json', 'r'))
    #global ticker_list
    #tickerdict=json.load(open('ticker_dict.json', 'r'))
    #ticklist=list(tickerdict.values())
    
    #sec_pe='Sector P/E Ratio: ' + str(round(float(sec_per(st, ticklist, sector_stocks, stocks_exch, json_entry='pe')), 2))

    return candle, fig_mc, company, sec, pri, intrinsic_price, pe, desc_candle, desc_mc


if __name__=="__main__":
    # check this link for running dash app also within Jupyterlab: https://stackoverflow.com/questions/45490002/how-to-use-dash-within-jupyter-notebook-or-jupyterlab#:~:text=Following%20these%20steps%20will%20unleash%20Plotly%20Dash%20directly,on%20a%20pandas%20dataframe%20that%20expands%20every%20second.
    app.run_server(port=6040, dev_tools_ui=True, dev_tools_hot_reload=True, threaded=True)