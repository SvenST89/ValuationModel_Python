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
from pandas_ta import bbands
import io
import datetime
from datetime import date
import re
#---- WEB DEV IMPORTS ------------------------#
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_trich_components as dtc
from dash.dependencies import Output, Input
# to open up browser automatically when app is launched
import os
from threading import Timer
import webbrowser
#---- DATABASE MANAGEMENT TOOLS --------------#
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras as extras

#---- OWN MODULE IMPORTS --------------------#
import config.pw
#import build_findb
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
CHART_THEME='plotly_dark' # plotly_white
# Loading the dashboard's json files stored before with dcf.py
#global sector_stocks, stocks_curr, stocks_exch
sector_stocks = json.load(open('sector_stocks.json', 'r'))
stocks_curr = json.load(open('stocks_curr.json', 'r'))
stocks_exch=json.load(open('stocks_exchange.json', 'r'))
#global ticker_list
tickerdict=json.load(open('ticker_dict.json', 'r'))
global ticker_list
ticker_list=list(tickerdict.values())
#ticker_list=['GTLB', 'PLTR', 'IFX.DE', 'EOAN.DE', 'SHL.DE', 'PAH3', 'NEL.OL', 'PLUG', 'DIS', 'KO', 'MMM', 'MSFT', 'SQ', 'ABNB', 'AMZN', 'EBAY', 'ORCL', 'PYPL', 'QCOM', 'SNOW', 'TEAM', 'TSLA', 'DBK.DE', 'PM']

company='The Walt Disney Company' # Currently, I have just a limited amount of companies in the database. The companies are: GitLab Inc., INFINEON TECHNOLOGIES AG, E.ON SE, Palantir Technologies Inc.,

ticker=tickerdict[company]
#global curr
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
per = ratio(ticker, json_entry='priceEarningsRatio')
pe_ratio = [per['priceEarningsRatio'] for per in per][0]
pe = 'P/E Ratio: ' + str(round(pe_ratio,2))
# Sector P/E Ratio
#sec_pe='Sector P/E Ratio: ' + str(sec_per(ticker, ticker_list, sector_stocks, stocks_exch, json_entry='pe'))
# Calculate Correlation between market and stock
stock_exchange = [key for key, ticker_list in stocks_exch.items() if ticker in ticker_list][0]
if stock_exchange == 'NASDAQ':
    _index='^NDX'
elif stock_exchange == 'NYSE':
    _index='^NYA'
elif stock_exchange == 'XETRA':
    _index = '^GDAXI'
else:
    _index = '^GSPC'
m_corr = market_correlation(ticker, _index)


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
    plot_bgcolor='rgb(42, 40, 41)',
    paper_bgcolor='rgb(42, 40, 41)',
    xaxis=dict(
        title=dict(
            text ="<b>Date</b>",
            font = dict(
                family= "Century Gothic",
                size = 16,
                color='rgb(253, 251, 251)',
                )
        ),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Century Gothic',
            size=12,
            color='rgb(253, 251, 251)',
        ),
    ),
    yaxis=dict(
        title=dict(
            text ="<b>Stock Price</b>",
            font = dict(
                family= "Century Gothic",
                size = 16,
                color='rgb(253, 251, 251)',
                )
        ),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        zeroline=False,
        ticks='outside',
        tickfont=dict(
            family='Century Gothic',
            size=12,
            color='rgb(253, 251, 251)',
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
# annotations_candle = []
# # # Title
# annotations_candle.append(dict(xref="paper", yref="paper", x=1, y=1.05,
#                                xanchor='right', yanchor='bottom',
#                                 text='Data Source: https://site.financialmodelingprep.com/developer/docs/',
#                                 font=dict(family='Open Sans',
#                                             size=10,
#                                             color='rgb(150,150,150)'),
#                                 showarrow=False))
# candle.update_layout(annotations=annotations_candle)


#====================================================#
# START STYLING OF APP
#====================================================#


# Influence Styling of dbc Themes: https://hellodash.pythonanywhere.com/adding-themes/dcc-components
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])
# Set up server using waitress (For robust local hosting and not using Dash development tools)
########################################################################################################################
server = app.server  # Added so that the app can be served using waitress.

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
            dbc.Col(width=3), # fake column to keep header in center
            dbc.Col([html.H2('STOCK ASSISTANT DASHBOARD', style={'margin-top': '15px', 'margin-left': 'auto', 'margin-bottom': '16px', 'textAlign': "center"}, className='text-center text-primary, mb-3')], width={'size': 6, 'offset': 0, 'order': 0}), # the max size of a screen is width=12!
            dbc.Col([html.Img(src="assets/ds_logo_clear.png", style={'height': '70%', 'width':'30%', 'margin-left': 'auto', 'margin-top':'15px', 'margin-bottom': '16px', 'float': 'right'})], width={'size': 3, 'offset': 0, 'order': 0})
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
                    #style={'font-family': 'Open Sans', "background-color":"rgb(42, 40, 41)", "color": "MediumTurqoise"},
                ),
                html.Div(id='company', className='text-center mt-3 p-2'),
                html.Div(id='sec', className='text-center p-2'),
                html.Div(id='pri', className='text-center p-2'),
                html.Div(id='intrinsic_price', className='text-center p-2'),
                html.Div(id='pe', className='text-center p-2'),
                html.Div(id='market_corr', className='text-center p-2'),
            #html.Hr(),
                #html.Div(id='sec_pe', className='text-center p-2'),
            ], width={'size': 4, 'offset': 0, 'order': 0}),
            # column 2
            dbc.Col([
                html.H5(id='desc_candle', className='text-center'),
                dcc.Loading([
                    dcc.Graph(
                        id='candle',
                        figure=candle,
                        style={'height': 550, 'margin-left':'12px'})],
                        id='loading-price-chart', type='dot', color='#1F51FF'),
                dcc.Checklist(
                    ['Simple Moving Avrg',
                    'Exponential Moving Avrg',
                    'Bollinger Bands'],
                    inline=True,
                    inputStyle={'margin-left': '15px',
                                'margin-right': '5px'},
                    id='complements-checklist',
                    style={'margin-left':'350px'}),
            ], width={'size': 8, 'offset': 0, 'order': 0}),
        ]),
        #=== ROW 3: To have a clean break in the screen!
        dbc.Row([
            # column 1
            dbc.Col([
                html.Hr(),
            ], width={'size': 12, 'offset': 0, 'order': 0}),
        ]),
        #=== ROW 4
        dbc.Row([
            # column 1
            dbc.Col([
                html.H5(id='desc_mc', className='text-center'),
                dcc.Graph(
                    id='fig_mc',
                    figure=fig_mc,
                    style={'height': 550, 'margin-bottom': '14px', 'margin-left':'12px'}),
                #html.Hr(),
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
        #=== ROW 5 (Footnote)
        dbc.Row([
            # column 1
            dbc.Col([
                html.Hr(),
                html.Div(html.P(['Financial Data obtained from: https://site.financialmodelingprep.com/developer/docs/', html.Br(), 'Economic Data retrieval from: https://sdw-wsrest.ecb.europa.eu/help/ and https://datahelp.imf.org/knowledgebase/articles/667681-using-json-restful-web-service', html.Br(),'Created by SVEN STEINBAUER']), id='footnote', style={'margin-top': '12px', 'margin-left': '75px', 'margin-bottom': '16px', 'font-size': '12px'}, className='text-center mt-3 p-2'),
            ], width={'size': 12, 'offset': 0, 'order': 0}),
        ]),
    ], fluid=True, className="dbc"
)

#====================================================#
# MAKE APP INTERACTIVE: CALLBACKS
#====================================================#
#Output("sec_pe", "children"),
@app.callback(
    [
    Output("candle", "figure"), Output("fig_mc", "figure"), Output("company", "children"), Output("sec", "children"), Output("pri", "children"), Output("intrinsic_price", "children"),
    Output("pe", "children"), Output("market_corr", "children"), Output("desc_candle", "children"), Output("desc_mc", "children")
    ],
    Input("stock-dropdown", "value"),
    Input('complements-checklist', 'value'),
)

def update_figures(st, checklist_values):
    # currency
    stocks_curr = json.load(open('stocks_curr.json', 'r'))
    curr=[key for key, ticker_list in stocks_curr.items() if st in ticker_list][0]
    #=== UPDATE CANDLESTICK CHART
    company=get_profile_data(st, json_entry='companyName', entry_point='profile')
    desc_candle=f'Price Chart: {company}'
    price_table = get_price_table(st).sort_values("date", ascending=True)
    # Bollinger Bands (normally distributed price fluctuations)
    df_bbands = bbands(price_table['close'], length=20, std=2)
    # Measuring Rolling and Exponential Rolling Mean (moving averages!)
    price_table['Simple Moving Avrg'] = price_table['close'].rolling(window=5).mean()
    price_table['Exponential Moving Avrg'] = price_table['close'].ewm(span=5, adjust=True).mean()
    # Each metric will have its own color in the chart.
    colors = {'Simple Moving Avrg': '#02D7D7',
              'Exponential Moving Avrg': '#027517', 'Bollinger_Bands_Low': '#D70263',
              'Bollinger_Bands_AVG': 'brown',
              'Bollinger_Bands_High': '#CECCCD'}
    candle = go.Figure()
    candle.add_trace(go.Candlestick(x=price_table['date'],
                                open=price_table['open'],
                                close=price_table['close'],
                                high=price_table['high'],
                                low=price_table['low'],
                                name='Stock Price'))
    # If the user has selected any of the indicators in the checklist, we'll represent it in the chart.
    if checklist_values != None:
        for metric in checklist_values:

            # Adding the Bollinger Bands' typical three lines.
            if metric == 'Bollinger Bands':
                candle.add_trace(go.Scatter(
                    x=price_table['date'], y=df_bbands.iloc[:, 0],
                    mode='lines', name='BBand Low', line={'color': colors['Bollinger_Bands_Low'], 'width': 1}))

                candle.add_trace(go.Scatter(
                    x=price_table['date'], y=df_bbands.iloc[:, 1],
                    mode='lines', name='BBand Average', line={'color': colors['Bollinger_Bands_AVG'], 'width': 1}))

                candle.add_trace(go.Scatter(
                    x=price_table['date'], y=df_bbands.iloc[:, 2],
                    mode='lines', name='BBand Up', line={'color': colors['Bollinger_Bands_High'], 'width': 1}))

            # Plotting any of the other metrics remained, if they are chosen.
            else:
                candle.add_trace(go.Scatter(
                    x=price_table['date'], y=price_table[metric], mode='lines', name=metric, line={'color': colors[metric], 'width': 1}))
    candle.layout.template=CHART_THEME
    candle.layout.height=500
    candle.update_layout(
        plot_bgcolor='rgb(42, 40, 41)',
        paper_bgcolor='rgb(42, 40, 41)',
        # title={
        # 'text': f'Price Chart: {company}',
        # 'y':0.95,
        # 'x': 0.5,
        # 'xanchor':'center',
        # 'yanchor': 'top'
        # },
        xaxis=dict(
            title=dict(
                text ="<b>Date</b>",
                font = dict(
                    family= "Century Gothic",
                    size = 16,
                    color='rgb(253, 251, 251)',
                    )
            ),
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Century Gothic',
                size=12,
                color='rgb(253, 251, 251)',
            ),
        ),
        yaxis=dict(
            title=dict(
                text ="<b>Stock Price</b>",
                font = dict(
                    family= "Century Gothic",
                    size = 16,
                    color='rgb(253, 251, 251)',
                    )
            ),
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            zeroline=False,
            ticks='outside',
            tickfont=dict(
                family='Century Gothic',
                size=12,
                color='rgb(253, 251, 251)',
            ),
        ),
        margin=dict(
            b=50,
            l=25,
            r=25,
            t=50
        ),
    )
    candle.update_xaxes(range=[min_date, end])
    candle.update_yaxes(tickprefix=curr+' ')
    # annotations_candle = []
    # # # Title
    # annotations_candle.append(dict(xref="paper", yref="paper", x=1, y=1.05,
    #                            xanchor='right', yanchor='bottom',
    #                             text='Data Source: https://site.financialmodelingprep.com/developer/docs/',
    #                             font=dict(family='Open Sans',
    #                                         size=10,
    #                                         color='rgb(150,150,150)'),
    #                             showarrow=False))
    # candle.update_layout(annotations=annotations_candle)

    #=== HISTOGRAM: FROM VALUATION MODEL
    latest_year=get_latest_available_fy(st, json_entry="date")
    fig_mc, intrinsic_value, intrinsic_price=build_dcf_model(company=company, year=latest_year, tax_rate=0.15825)
    desc_mc=f'Intrinsic EV Monte Carlo Simulation: {company}'

    #=== Update Information below dropdown field
    company='Company: ' + company
    sec='Sector: ' + get_profile_data(st, json_entry='sector', entry_point='profile')
    pri='Real Time Price: ' + str(round(real_time_stockprice(st, json_entry='price'),2))
    intrinsic_price='Intrinsic Stock Price: ' + str(intrinsic_price)
    #---
    per = ratio(st, json_entry='priceEarningsRatio')
    pe_ratio = [per['priceEarningsRatio'] for per in per][0]
    pe = 'P/E Ratio: ' + str(round(pe_ratio,2))
    sector_stocks = json.load(open('sector_stocks.json', 'r'))
    stocks_exch=json.load(open('stocks_exchange.json', 'r'))
    #global ticker_list
    #tickerdict=json.load(open('ticker_dict.json', 'r'))
    #ticklist=list(tickerdict.values())
    
    #sec_pe='Sector P/E Ratio: ' + str(round(float(sec_per(st, ticklist, sector_stocks, stocks_exch, json_entry='pe')), 2))
    stock_exchange = [key for key, ticker_list in stocks_exch.items() if st in ticker_list][0]
    if stock_exchange == 'NASDAQ':
        _index='^NDX'
    elif stock_exchange == 'NYSE':
        _index='^NYA'
    elif stock_exchange == 'XETRA':
        _index = '^GDAXI'
    else:
        _index = '^GSPC'
    market_corr = f"Stock´s correlation with related market index: " + market_correlation(st, _index)

    return candle, fig_mc, company, sec, pri, intrinsic_price, pe, market_corr, desc_candle, desc_mc

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:6040/')


if __name__=="__main__":
    # check this link for running dash app also within Jupyterlab: https://stackoverflow.com/questions/45490002/how-to-use-dash-within-jupyter-notebook-or-jupyterlab#:~:text=Following%20these%20steps%20will%20unleash%20Plotly%20Dash%20directly,on%20a%20pandas%20dataframe%20that%20expands%20every%20second.
    Timer(1, open_browser).start()
    #app.run_server(port=6040, dev_tools_ui=True, dev_tools_hot_reload=True, threaded=True) # for dev deployment developing with Dash
    app.run_server(debug=False, port=6040) # for production deployment with waitress