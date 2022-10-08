# To make web requests at the ECB SDMX 2.1 RESTful web service
import requests

# For use of the ECB API
#from sdw_api import SDW_API

# Standard libs for data manipulation
import pandas as pd
import numpy as np
import io
import datetime
from datetime import date
import re
from sqlalchemy import create_engine
import psycopg2.extras as extras
from config.api import MY_API_KEY
import config.pw

#========================================================================================================================================#

# Function for swapping columns to fit the database schema
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

# Function to write Data to a table in database

def execute_values(conn, df, table):
  
    tuples = [tuple(x) for x in df.to_numpy()]
  
    cols = ','.join(list(df.columns))
  
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()
    
#----- INTERST RATE DATA FROM THE ECB--------------------------------#

def get_ir_data(keys_list, start):
    # Entrypoint for the ECB SDMX 2.1 RESTful web service
    entrypoint="https://sdw-wsrest.ecb.europa.eu/service/data/"

    # Pandas dataframe list
    pd_list=[]

    for i in keys_list:
        #===============================================================
        # store time series (ts) key components in a list
        keyComponent_list = re.findall(r"(\w+)\.",i)
        # access the database identifier, which is the very first component at index 0
        db_id = keyComponent_list[0]
        # remainder of key components
        keyRemainder = '.'.join(re.findall(r"\.(\w+)",i))
        # merge the request url
        requestUrl= entrypoint + db_id+ "/" + keyRemainder + "?format=genericdata"
        #--------------------------------------------------------------
    
        # Set parameters for http request get method
        today = date.today()        # get the date of today
        today_formatted = today.strftime("%Y-%m-%d") # format the date as a string 2020-10-31
        parameters = {
        'startPeriod': start,  # Start date of the time series, e.g. '2019-12-01'
        'endPeriod': today_formatted     # End of the time series
        }
        #--------------------------------------------------------------
    
        # Make the HTTP get request for each url
        response = requests.get(requestUrl, params=parameters, headers={'Accept': 'text/csv'})
        assert response.status_code == 200, f"Expected response code 200, got {response.status_code} for {requestUrl}. Check again your url!"
        #--------------------------------------------------------------

        # Read the response as a file into a Pandas Dataframe
        ir_df = pd.read_csv(io.StringIO(response.text))
        # Create a new DataFrame called 'ir_ts'; isolating Time Period and Observation Value only.
        ir_ts = ir_df.filter(['TIME_PERIOD', 'OBS_VALUE'], axis=1)
        #--------------------------------------------------------------
    
        # 'TIME_PERIOD' was of type 'object' (as can be seen in yc_df.info). Convert it to datetime first
        ir_ts['TIME_PERIOD'] = pd.to_datetime(ir_ts['TIME_PERIOD'])
        # Set 'TIME_PERIOD' to be the index
        ir_ts = ir_ts.set_index('TIME_PERIOD')
        # Append individual dataframe to pd_list
        pd_list.append(ir_ts)
        #===============================================================
    
    # Now concatenate each individual yield curve dataframe from the list of dataframes,
    # collected in the loop, into one single dataframe
    interest_df=pd.concat(pd_list, axis=1)
    return interest_df

#--- FUNCTION TO RETRIEVE DATA FOR A COMPANY FROM OUR DATABASE -------------------------------------------#

def get_database_findata_year(company, year, month, day, engine):
    """Get financial data for a company on an annual basis.
    Type in the company as a string and year/month/day as integer values"""
    #================================
    # Set necessary url variables for the sqlalchemy create_engine() method.
    user='svenst89' # or default user 'postgres'
    password=config.pw.password # edit the password if you switch to the default user 'postgres'; I setup different passwords.
    host='localhost'
    port='5433'
    database='fundamentalsdb'
    #================================
    engine=engine
    year=year
    month=month
    day=day
    dt = datetime.datetime(year, month, day)
    date_formatted = dt.strftime("%Y-%m-%d") #%H:%M:%S
    try:
        bs_y=pd.read_sql(f"SELECT * FROM balancesheet WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
        incs_y=pd.read_sql(f"SELECT * FROM incomestatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
        cs_y=pd.read_sql(f"SELECT * FROM cashflowstatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
        pd.set_option('display.expand_frame_repr', False)
        
        if len(bs_y.index) == 0:
            print(f"I guess, there is no data available for the specified date '{date_formatted}' for '{company}'. I will try 'September FY end'...")
            month=9
            day=30
            dt = datetime.datetime(year, month, day)
            date_formatted = dt.strftime("%Y-%m-%d") #%H:%M:%S
            bs_y=pd.read_sql(f"SELECT * FROM balancesheet WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
            incs_y=pd.read_sql(f"SELECT * FROM incomestatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
            cs_y=pd.read_sql(f"SELECT * FROM cashflowstatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
            pd.set_option('display.expand_frame_repr', False)
            print(f"You have successfully retrieved the financial statement data for company '{company}' for the Fiscal Year '{date_formatted}'!")
            if len(bs_y.index)==0:
                print(f"Again, there is no data available for the specified date '{date_formatted}' for '{company}'. I will try 'June FY end'...")
                month=6
                day=30
                dt = datetime.datetime(year, month, day)
                date_formatted = dt.strftime("%Y-%m-%d") #%H:%M:%S
                bs_y=pd.read_sql(f"SELECT * FROM balancesheet WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                incs_y=pd.read_sql(f"SELECT * FROM incomestatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                cs_y=pd.read_sql(f"SELECT * FROM cashflowstatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                pd.set_option('display.expand_frame_repr', False)
                print(f"You have successfully retrieved the financial statement data for company '{company}' for the Fiscal Year '{date_formatted}'!")
                if len(bs_y.index)==0:
                    print(f"Again, there is no data available for the specified date '{date_formatted}' for '{company}'. I will try 'May FY end'...")
                    month=5
                    day=31
                    dt = datetime.datetime(year, month, day)
                    date_formatted = dt.strftime("%Y-%m-%d") #%H:%M:%S
                    bs_y=pd.read_sql(f"SELECT * FROM balancesheet WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                    incs_y=pd.read_sql(f"SELECT * FROM incomestatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                    cs_y=pd.read_sql(f"SELECT * FROM cashflowstatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                    pd.set_option('display.expand_frame_repr', False)
                    print(f"You have successfully retrieved the financial statement data for company '{company}' for the Fiscal Year '{date_formatted}'!")
                    if len(bs_y.index)==0:
                        print(f"Again, there is no data available for the specified date '{date_formatted}' for '{company}'. I will try 'May FY end'...")
                        month=3
                        day=31
                        dt = datetime.datetime(year, month, day)
                        date_formatted = dt.strftime("%Y-%m-%d") #%H:%M:%S
                        bs_y=pd.read_sql(f"SELECT * FROM balancesheet WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                        incs_y=pd.read_sql(f"SELECT * FROM incomestatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                        cs_y=pd.read_sql(f"SELECT * FROM cashflowstatement WHERE shortname = '{str(company)}' AND date='{date_formatted}'", engine)
                        pd.set_option('display.expand_frame_repr', False)
                        print(f"You have successfully retrieved the financial statement data for company '{company}' for the Fiscal Year '{date_formatted}'!")
        else:
            print(f"You have successfully retrieved the financial statement data for company '{company}' for the Fiscal Year '{date_formatted}'!")
        return bs_y, incs_y, cs_y
    except Exception as ex:
        print(f"Sorry! Something went wrong! I guess, there is no data available for the specified date '{date_formatted}' for '{company}'. Have they already published or do they report somewhen in the middle of the year?\n"+ ex)
        
def get_database_findata(company, engine):
    """Get all available financial company data from our database.
    Type in the company as a string"""
    #================================
    # Set necessary url variables for the sqlalchemy create_engine() method.
    user='svenst89' # or default user 'postgres'
    password=config.pw.password # edit the password if you switch to the default user 'postgres'; I setup different passwords.
    host='localhost'
    port='5433'
    database='fundamentalsdb'
    #================================
    engine=engine
    try:
        bs=pd.read_sql(f"SELECT * FROM balancesheet WHERE shortname = '{str(company)}'", engine)
        incs=pd.read_sql(f"SELECT * FROM incomestatement WHERE shortname = '{str(company)}'", engine)
        cs=pd.read_sql(f"SELECT * FROM cashflowstatement WHERE shortname = '{str(company)}'", engine)
        pd.set_option('display.expand_frame_repr', False)
        print(f"You have successfully retrieved the financial statement data for company '{company}'!")
        return bs, incs, cs
    except Exception as ex:
        print(f"Sorry! Something went wrong! I guess, there is no data available for the '{company}'. Have they already published or do they report somewhen else?\n"+ ex)

def get_company_data(company, engine):
    """Get company profile data from our database.
    Type in the company as a string"""
    #================================
    # Set necessary url variables for the sqlalchemy create_engine() method.
    user='svenst89' # or default user 'postgres'
    password=config.pw.password # edit the password if you switch to the default user 'postgres'; I setup different passwords.
    host='localhost'
    port='5433'
    database='fundamentalsdb'
    #================================
    engine=engine
    try:
        company=pd.read_sql(f"SELECT * FROM company WHERE shortname = '{str(company)}'", engine)
        pd.set_option('display.expand_frame_repr', False)
        print(f"You have successfully retrieved the company data for company '{company}'!")
        return company
    except Exception as ex:
        print(f"Sorry! Something went wrong! I guess, there is no data available for the '{company}'. Have you stored its data in the database?\n"+ ex)

#--- FUNCTION TO CALCULATE THE WEIGHTED AVERAGE COST OF CAPITAL BASED ON OUR FINANCIAL STATEMENT DATA STORED IN POSTGRES DATABASE -------------------------------------------#
def get_wacc(company, year, month, day, rfr, mrp, at_debt_cost, engine):
    """This function calculates the weighted average cost of capital (WACC) given the capital structure of the firm.
        The WACC serves as the discounting rate for the Free Cashflows within the corporate valuation model, the DCF method."""
    #================================================================
    #engine=create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
    bs_y, incs_y, cs_y=get_database_findata_year(company, year, month, day, engine)
    bs_y=bs_y.sort_values(by=['date'], ascending=False)
    # revise balance sheet
    droprows = ['link', 'finalLink', 'calendarYear', 'period', 'symbol', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate']
    bs_y = bs_y[bs_y.item.isin(droprows) == False]
    # switch value column to float; otherwise you cannot calculate
    bs_y['value'] = bs_y['value'].astype(float)
    #================================================================
    # get 'totalAssets', 'totalEquity', 'totalLiabilities' in order to calculate weights
    mask_ta=bs_y['item'].values=='totalAssets'
    mask_te=bs_y['item'].values=='totalEquity'
    mask_tl=bs_y['item'].values=='totalLiabilities'
    totalAssets=float(bs_y.loc[mask_ta, 'value'])
    totalEquity=float(bs_y.loc[mask_te, 'value'])
    totalLiabilities=float(bs_y.loc[mask_tl, 'value'])
    # calculate weights
    w_tl=totalLiabilities / totalAssets
    w_te=totalEquity / totalAssets
    # retrieve beta from FMP
    beta=get_beta(ticker)
    # calculate the required return on equity according to CAPM
    required_return_equity=rfr + mrp*beta
    #====calculate WACC
    wacc=w_tl*at_debt_cost+w_te*required_return_equity
    return wacc

