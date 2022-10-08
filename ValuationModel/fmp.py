# To make web requests at the ECB SDMX 2.1 RESTful web service
import requests
import yfinance as yf

# For use of the ECB API
#from sdw_api import SDW_API

# Standard libs for data manipulation
import pandas as pd
import numpy as np
import io
from datetime import date
import re
import psycopg2.extras as extras
from config.api import MY_API_KEY

#========================================================================================================================================#

def retrieve_data_from_api(ticker_list):
    # https://financialmodelingprep.com/api/v3/income-statement/AAPL?limit=120&apikey=YOUR_API_KEY'
    entrypoint="https://financialmodelingprep.com/api/v3/"
    headers = {'Accept': 'application/json'}
    statement_list=["balance-sheet-statement", "income-statement", "cash-flow-statement"]
    my_api_key=MY_API_KEY
    bs_df_list=[]
    inc_df_list=[]
    cf_df_list=[]

    for ticker in ticker_list:
        for statement in statement_list:
            # merge the request url
            requestUrl= entrypoint + statement +"/" + ticker + "?limit=120" +  "&apikey=" + my_api_key #

            response = requests.get(requestUrl)#, headers=headers)
            assert response.status_code == 200, f"Expected response code 200, got {response.status_code} for {requestUrl}. Check again your url!"
            # use the .json() method offered by 'requests' package: https://datagy.io/python-requests-json/
            response_list=response.json()
            # make dataframe from list
            table=pd.DataFrame(response_list)
            # transpose, preserve the 'date' column, reset_index "shifts" 'date'-column one step left as index and item-column is named 'index'; so, rename 'index' to 'item'
            # https://stackoverflow.com/questions/39874956/transpose-table-then-set-and-rename-index
            try:
                table_T=table.set_index('date').T.reset_index().rename(columns={'index':'item'})
            except KeyError:
                print(f"{ticker} obviously has no data as column 'date' cannot be found for {statement}!")
                continue
            #table_T.columns.name=None
            # now move row 'calendarYear' and 'period' to first rows and drop duplicated rows
            target_cy=5
            target_fy=6
            idx = [target_cy] + [target_fy] + [i for i in range(len(table_T)) if i != [target_cy, target_fy] ]
            table_rev=table_T.iloc[idx].drop_duplicates()
            # through rearranging the index column will start with number 5; we have to reset the index to start at '0' again.
            final_table=table_rev.reset_index(drop=True)
            final_table['shortname']=yf.Ticker(ticker).info['shortName']
            ft_transformed=pd.melt(final_table, id_vars=["shortname", "item"], var_name="date", value_name="value")
            if statement == "balance-sheet-statement":
                bs_df_list.append(ft_transformed)
            elif statement == "income-statement":
                inc_df_list.append(ft_transformed)
            else:
                cf_df_list.append(ft_transformed)
    return bs_df_list, inc_df_list, cf_df_list

def get_index_table():
    entrypoint="https://financialmodelingprep.com/api/v3/"
    headers = {'Accept': 'application/json'}
    my_api_key=MY_API_KEY
    requestUrl='https://financialmodelingprep.com/api/v3/quotes/index?apikey='+my_api_key
    response = requests.get(requestUrl)#, headers=headers)
    assert response.status_code == 200, f"Expected response code 200, got {response.status_code} for {requestUrl}. Check again your url!"
    # use the .json() method offered by 'requests' package: https://datagy.io/python-requests-json/
    response_list=response.json()
    # make dataframe from list
    table=pd.DataFrame(response_list)[['symbol', 'name']]
    return table

def get_beta(ticker):
    my_api_key=MY_API_KEY
    entrypoint="https://financialmodelingprep.com/api/v3/profile/"
    
    # get company beta value from FMP
    # For the extract function I used the code of this colleague instead of working it out on my own (sorry! too lazy!)
    # https://python.plainenglish.io/extracting-specific-keys-values-from-a-messed-up-json-file-python-dfb671482681
    def extract(data, keys):
        out = []
        queue = [data]
        while len(queue) > 0:
            current = queue.pop(0)
            if type(current) == dict:
                for key in keys:
                    if key in current:
                        out.append({key:current[key]})
            
                for val in current.values():
                    if type(val) in [list, dict]:
                        queue.append(val)
            elif type(current) == list:
                queue.extend(current)
        return out
    # Now make request to FMP API
    requestUrl= entrypoint + ticker + "?" +  "apikey=" + my_api_key
    response = requests.get(requestUrl)#, headers=headers)
    assert response.status_code == 200, f"Expected response code 200, got {response.status_code} for {requestUrl}. Check again your url!"
    # use the .json() method offered by 'requests' package: https://datagy.io/python-requests-json/
    response_list=response.json()
    beta=extract(response_list, ["beta"])
    beta=beta[0]['beta']
    return beta