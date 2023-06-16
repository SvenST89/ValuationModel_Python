#=============SET UP LOGGING ======================#
import logging
import sys
# specifically for pyplot: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger('matplotlib').disabled = True

logger=logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)

# set logging handler in file
fileHandler=logging.FileHandler(filename="log/kpi.log", mode='w')
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
#===================================================#
# Standard libs for data manipulation
import numpy as np
import pandas as pd
import json
import io
import datetime
from datetime import date
from datetime import timedelta
import re
#---- VISUALIZATION TOOLS --------------#
import plotly.graph_objects as go
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
from KPI import KpiPrep

import warnings
warnings.filterwarnings('ignore')

def make_kpi_tables(ticker=''):
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
    
    #===OPTIONAL: INSTEAD OF USING FUNCTION 'get_database_findata_year()' use the general function retrieving all available financial data and extracting a specific year from the whole dataframe
    tickerdict=json.load(open('ticker_dict.json', 'r'))
    company = [k for k, v in tickerdict.items() if v == ticker][0]
    #latest_year=get_latest_available_fy(ticker=ticker, json_entry="date")
    bs, incs, cs = get_database_findata(company, engine)
    # bs_y=clearDataframes_and_get_fy(company, bs, 'bs', latest_year)
    # incs_y=clearDataframes_and_get_fy(company, incs, 'incs', latest_year)
    # cs_y=clearDataframes_and_get_fy(company, cs, 'cs', latest_year)

    # instantiate KPI Table Object
    kpis = KpiPrep.MakeKpiDf(stock=ticker)
    #=== Income Statement Financials
    kpis_incs=['revenue', 'ebitda']
    kpi_incs_df = kpis.make_kpi_df(incs, kpis_incs, name='Items', spec='standard')
    #=== Balance Sheet Financials
    kpis_bs=['cashAndCashEquivalents', 'netDebt','propertyPlantEquipmentNet', 'intangibleAssets', 'totalAssets']
    kpi_bs_df = kpis.make_kpi_df(bs, kpis_bs, name='Items', spec='standard')
    #=== Cash Flow Financials
    kpis_cs=['operatingCashFlow', 'capitalExpenditure', 'freeCashFlow']
    kpi_cs_df = kpis.make_kpi_df(cs, kpis_cs, name='Items', spec='standard')
    #=== RATIOS ===================================#
    kpis_add_incs = ['operatingIncomeRatio', 'netIncomeRatio']
    kpi_add_df = kpis.make_kpi_df(incs, kpis_add_incs, name='Items', spec='standard')
    #---SPECIFICS
    kpis_spec = ['returnOnAssets', 'currentRatio', 'returnOnEquity', 'returnOnCapitalEmployed', 'dividendYield']
    kpi_spec_df = kpis.make_kpi_df(incs, kpis_spec, name='Items', spec='specifics')

    return kpi_incs_df, kpi_bs_df, kpi_cs_df, kpi_add_df, kpi_spec_df

def style_kpi_table(df, format='percent', CHART_THEME='plotly_dark'):
    kpi_df = df.reset_index()
    headerColor='black'
    rowEvenColor='lightgrey'
    rowOddColor='darkgrey'
    even_odd_list=[rowEvenColor, rowOddColor]*kpi_df.shape[0] # switch coloring between even and odd rows
    colvallist=[]
    for col in list(kpi_df.columns):
        colval=kpi_df[col].values
        colvallist.append(colval)

    if format == 'percent':
        formatting = ["", ".2%"]
    else:
        formatting = ["", "numeric"]

    kpi_tbl=go.Figure(data=[
    go.Table(
        header=dict(
            values=list(kpi_df.columns),
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left', 'center'],
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=colvallist,
            line_color='darkslategray',
            fill_color=[even_odd_list*2],
            align=['left', 'center'],
            font=dict(color='darkslategray', size=14),
            format=formatting,
        )
    )
    ])
    kpi_tbl.layout.template=CHART_THEME
    return kpi_tbl