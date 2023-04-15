#=============SET UP LOGGING ======================#
import logging
import sys

logger=logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)

# set logging handler in file
fileHandler=logging.FileHandler(filename="log/db_main.log", mode='w')
fileHandler.setFormatter(formatter)
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
#===================================================#

#===================================================#
# NECESSARY IMPORTS
#===================================================#
# To make web requests
import requests
#---- DATABASE MANAGEMENT TOOLS --------------#
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras as extras

#---- DATA MANIPULATION TOOLS ----------------#
import yfinance as yf
import pandas_datareader as dr
import numpy as np
import pandas as pd
import datetime
from datetime import date

#---- OWN MODULE IMPORTS --------------------#
import config.pw
from ValuationModel.assist_functions import swap_columns
from ValuationModel.fmp import retrieve_data_from_api, get_all_company_data
from ValuationModel.assist_functions import execute_values
from config.api import MY_API_KEY

today=date.today().strftime("%Y-%m-%d")

#===================================================#
# SET UP DATABASE
#===================================================#

# Set necessary url variables for the sqlalchemy create_engine() method.
user='svenst89' # or default user 'postgres'
password=config.pw.password # edit the password if you switch to the default user 'postgres'; I setup different passwords.
host='localhost'
port='5432'
database='fundamentalsdb'

# Create an engine object as medium for database exchange with PostgreSQL
def run_engine():
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
if __name__=='__main__':
    try:
        global engine
        engine=run_engine()
        logger.info(f"You have successfully created an engine object for your Postgres DB at {host} for user {user}!")
    except Exception as ex:
        logger.info("Sorry your engine has not been created. Some exception has occurred. Please check and try it again!\n", ex)

# Function to instantiate a database connection
def connect():
    """ create a database connection to the Postgres database and a cursor object to control queries
    :return: Connection and a cursor object or None
    """
    try:
        conn = psycopg2.connect(
            database=database,
            user=user,
            password=password,
            host=host,
            port=port)
        cur = conn.cursor()
        logger.info(f"Successfully created a connection with the Postgres Database {database} at host {host} for user {user}!")
    except (Exception, psycopg2.DatabaseError) as error:
        logger.info("Error while creating PostgreSQL database connection", error)
    
    return conn, cur

# With PostgreSQL I deleted the foreign keys in the company table of the statements, as the tables are sequentially built, which means that errors will be
# thrown if the statement tables are not yet created but we try to reference them as a foreign key in the company table!
def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS company (
                id BIGINT PRIMARY KEY,
                shortName TEXT,
                symbol VARCHAR(100),
                industry VARCHAR(100),
                sector VARCHAR(100),
                currency VARCHAR(100),
                exchangeshortname VARCHAR(100),
                bs_id INTEGER,
                is_id INTEGER,
                cs_id INTEGER
        )
        """,
        """ CREATE TABLE IF NOT EXISTS balancesheet (
                id BIGINT,
                shortName TEXT,
                date TEXT,
                item TEXT,
                value TEXT,
                company_id BIGINT,
                PRIMARY KEY (id, date),
                FOREIGN KEY (company_id) REFERENCES company (id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS incomestatement (
                id BIGINT,
                shortName TEXT,
                date TEXT,
                item TEXT,
                value TEXT,
                company_id BIGINT,
                PRIMARY KEY (id, date),
                FOREIGN KEY (company_id) REFERENCES company (id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cashflowstatement (
                id BIGINT,
                shortName TEXT,
                date TEXT,
                item TEXT,
                value TEXT,
                company_id BIGINT,
                PRIMARY KEY (id, date),
                FOREIGN KEY (company_id) REFERENCES company (id)
        )
        """
        )
    conn = None
    try:
        # connect to the PostgreSQL server
        conn, cur = connect()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
        logger.info(f"Successfully created the tables in the Postgres Database {database} at host {host} for user {user}!")
    except (Exception, psycopg2.DatabaseError) as error:
        logger.info(error)
    finally:
        if conn is not None:
            conn.close()
  
  
if __name__ == '__main__':
    create_tables()

#===================================================#
# GET COMPANY DATA & STORE IT IN DB
#===================================================#

# Ticker list
ticker_list=['GTLB', 'PLTR', 'IFX.DE', 'EOAN.DE', 'SHL.DE', 'PAH3', 'NEL.OL', 'PLUG', 'DIS', 'KO', 'MMM', 'MSFT', 'SQ', 'ABNB', 'AMZN', 'EBAY', 'ORCL', 'PYPL', 'QCOM', 'SNOW', 'TEAM', 'TSLA', 'DBK.DE', 'PM']
# Now write a dictionary with the respective company as key and the Ticker as value
import json
  
companyTicker_dict = {'E.ON SE': "EOAN.DE",
                      'Infineon Technologies AG' :"IFX.DE",
                     'GitLab Inc.': "GTLB",
                     'Palantir Technologies Inc.': "PLTR",
                     'PORSCHE AUTOM.HLDG VZO':"PAH3.DE",
                     'Siemens Healthineers AG': "SHL.DE",
                     'Plug Power Inc.': "PLUG",
                     'Nel ASA': "NEL.OL",
                     'The Walt Disney Company':'DIS',
                     'The Coca-Cola Company':'KO',
                     '3M Company':'MMM',
                     'Microsoft Corporation':'MSFT',
                     'Block Inc.': 'SQ',
                     'Airbnb, Inc.':'ABNB',
                     'Amazon.com, Inc.':'AMZN',
                     'eBay Inc.':'EBAY',
                     'Oracle Corporation':'ORCL',
                     'PayPal Holdings, Inc.':'PYPL',
                     'QUALCOMM Incorporated':'QCOM',
                     'Snowflake Inc.':'SNOW',
                     'Atlassian Corporation Plc':'TEAM',
                     'Tesla, Inc.':'TSLA',
                     'Deutsche Bank Aktiengesellschaft':'DBK.DE',
                     'Philip Morris International Inc.':'PM'}
  
with open('ticker_dict.json', 'w') as convert_file:
     convert_file.write(json.dumps(companyTicker_dict))

#=== Get Financial Data from FMP
bs_df_list, inc_df_list, cf_df_list=retrieve_data_from_api(ticker_list)
# Complete Balance Sheet Set
final_bs_df=pd.concat(bs_df_list)
# Complete Income Statement Set
final_inc_df=pd.concat(inc_df_list)
# Complete Cash Flow Statement Set
final_cs_df=pd.concat(cf_df_list)
final_bs_df['id']=range(0, len(final_bs_df['item']))
final_cs_df['id']=range(0, len(final_cs_df['item']))
final_inc_df['id']=range(0, len(final_inc_df['item']))
# Backup data as csv
final_bs_df.to_csv(f"data/final_bs_df{today}.csv")
final_cs_df.to_csv(f"data/final_cs_df{today}.csv")
final_inc_df.to_csv(f"data/final_inc_df{today}.csv")

#=== Get General Company Data
# Make a dictionary to transform it into a dataframe later
dict_list=[]

for ticker in ticker_list:
    # create a ticker object, that will deliver a dictionary of infos
    info_dict_list=get_all_company_data(ticker)
    try:
        info_dict=info_dict_list[0]
        subdict={k: info_dict.get(k, None) for k in ('companyName', 'symbol', 'industry', 'sector', 'currency', 'exchangeShortName')}
        #subdict_df=pd.DataFrame(subdict, index=[0]) # you need to pass an index '0' if you have just one entry as it is just one company
        dict_list.append(subdict)
    except IndexError:
        logger.info(f"{ticker} obviously has no data. Continue!")
        continue

company_data = pd.DataFrame(dict_list)
company_data=company_data.rename(columns = {'companyName':'shortname', 'exchangeShortName':'exchangeshortname'}, inplace = False)

# Create an id-column in the company data table
company_data['id']=company_data.index
# Map the company id as foreign key for the statement tables
company_id_mapper = pd.Series(company_data.id.values, index=company_data.shortname).to_dict()

# Map back and forth between the company table and the balance sheet table to link their respective foreign keys
# Map company id to balance sheet table
final_bs_df['company_id'] = final_bs_df['shortname'].map(company_id_mapper)
# Map balance sheet id to company table
bs_id_mapper = pd.Series(final_bs_df.id.values, index=final_bs_df.shortname)
company_data['bs_id'] = company_data['shortname'].map(bs_id_mapper.to_dict())

# Swap columns to fit into the database schema
final_bs_df=swap_columns(final_bs_df, 'value', 'id')
final_bs_df=swap_columns(final_bs_df, 'date', 'id')
final_bs_df=swap_columns(final_bs_df, 'item', 'id')
final_bs_df=swap_columns(final_bs_df, 'shortname', 'id')
final_bs_df=swap_columns(final_bs_df, 'item', 'date')

#=== CREATE TEMPORARY TABLES & INSERT DATA
# CREATE A FUNCTION FOR TEMPORARY TABLE CREATION
def create_temp_tables():
    """ create temporary tables in the PostgreSQL database in which 
        we can store our yahoo finance data with 'to_sql' append or replace (probably 'replace')"""
    
    commands = (
        """ CREATE TABLE IF NOT EXISTS company_temp (
                id BIGINT PRIMARY KEY,
                shortName TEXT,
                symbol VARCHAR(100),
                industry VARCHAR(100),
                sector VARCHAR(100),
                currency VARCHAR(100),
                exchangeshortname VARCHAR(100),
                bs_id INTEGER,
                is_id INTEGER,
                cs_id INTEGER
        )
        """,
        """ CREATE TABLE IF NOT EXISTS balancesheet_temp (
                id BIGINT,
                shortName TEXT,
                date TEXT,
                item TEXT,
                value TEXT,
                company_id BIGINT,
                PRIMARY KEY (id, date),
                FOREIGN KEY (company_id) REFERENCES company (id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS incomestatement_temp (
                id BIGINT,
                shortName TEXT,
                date TEXT,
                item TEXT,
                value TEXT,
                company_id BIGINT,
                PRIMARY KEY (id, date),
                FOREIGN KEY (company_id) REFERENCES company (id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cashflowstatement_temp (
                id BIGINT,
                shortName TEXT,
                date TEXT,
                item TEXT,
                value TEXT,
                company_id BIGINT,
                PRIMARY KEY (id, date),
                FOREIGN KEY (company_id) REFERENCES company (id)
        )
        """
        )
    conn = None
    try:
        # connect to the PostgreSQL server
        conn, cur = connect()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logger.info(error)
    finally:
        if conn is not None:
            conn.close()
  
  
if __name__ == '__main__':
    create_temp_tables()

#---WRITING DATA TO TABLE-----------------------------------#
# USE 'engine' as connection entrypoint from sqlalchemy engine defined above! Otherwise sqlalchemy will use the default sqlite3 schema, which does not
# match with our 'Postgres' schema here and which will throw an error! : https://stackoverflow.com/questions/45326026/to-sql-pandas-data-frame-into-sql-server-error-databaseerror
#engine=create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
final_bs_df.to_sql('balancesheet_temp', engine, if_exists='replace', index=False)

#=== Income Statement
# Map back and forth between the company table and the balance sheet table to link their respective foreign keys
# Map company id to balance sheet table
final_inc_df['company_id'] = final_inc_df['shortname'].map(company_id_mapper)
# Map balance sheet id to company table
is_id_mapper1 = pd.Series(final_inc_df.id.values, index=final_inc_df.shortname)
company_data['is_id'] = company_data['shortname'].map(is_id_mapper1.to_dict())
# Swap columns to fit into the database schema
final_inc_df=swap_columns(final_inc_df, 'value', 'id')
final_inc_df=swap_columns(final_inc_df, 'date', 'id')
final_inc_df=swap_columns(final_inc_df, 'item', 'id')
final_inc_df=swap_columns(final_inc_df, 'shortname', 'id')
final_inc_df=swap_columns(final_inc_df, 'item', 'date')
# Store income statement data in the database
#engine=create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
final_inc_df.to_sql("incomestatement_temp", engine, if_exists="replace", index=False)

#=== Cashflow Statement
# Map back and forth between the company table and the income statement table to link their respective foreign keys
# Map company id to income statement table
final_cs_df['company_id'] = final_cs_df['shortname'].map(company_id_mapper)
# Map income statement id to company table
cs_id_mapper = pd.Series(final_cs_df.id.values, index=final_cs_df.shortname).to_dict()
company_data['cs_id'] = company_data['shortname'].map(cs_id_mapper)
# Swap columns to fit into the database schema
final_cs_df=swap_columns(final_cs_df, 'value', 'id')
final_cs_df=swap_columns(final_cs_df, 'date', 'id')
final_cs_df=swap_columns(final_cs_df, 'item', 'id')
final_cs_df=swap_columns(final_cs_df, 'shortname', 'id')
final_cs_df=swap_columns(final_cs_df, 'item', 'date')
# Store income statement data in the database
#engine=create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
final_cs_df.to_sql("cashflowstatement_temp", engine, if_exists="replace", index=False)

# Add data content to the databases
#engine=create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
company_data.to_sql("company_temp", engine, if_exists="replace", index=False)

#=== Append only the new Data to the Statement Tables
def insert_new_data():
    """ After the data has been stored into the temporary tables, check for each table
        whether there are new id-date value pairs with an SQL WHERE clause. If yes, then insert only the new data into the 
        original standard tables.
        
        Specifically, the SQL query does the following:
        It checks in the original standard table which id-date value pairs are identical in the temporary table. Then, it selects
        only those rows from the temporary table in which the id-date value pairs are NOT yet in the original standard table and inserts these
        rows in the respective original standard table."""
    
    commands = (
        """ INSERT INTO company (id, shortname, symbol, industry, sector, currency, exchangeshortname, bs_id, is_id, cs_id)
            SELECT id, shortname, symbol, industry, sector, currency, exchangeshortname, bs_id, is_id, cs_id FROM company_temp
            WHERE NOT EXISTS (
                SELECT * FROM company
                WHERE company.id=company_temp.id
                )
        """,
        """ INSERT INTO balancesheet (id, shortname, date, item, value, company_id)
            SELECT id, shortname, date, item, value, company_id FROM balancesheet_temp
            WHERE NOT EXISTS (
                SELECT * FROM balancesheet bs
                WHERE bs.id = balancesheet_temp.id AND bs.date = balancesheet_temp.date
                )
        """,
        """ INSERT INTO incomestatement (id, shortname, date, item, value, company_id)
            SELECT id, shortname, date, item, value, company_id FROM incomestatement_temp
            WHERE NOT EXISTS (
                SELECT * FROM incomestatement
                WHERE incomestatement.id = incomestatement_temp.id AND incomestatement.date = incomestatement_temp.date
                )
        """,
        """ INSERT INTO cashflowstatement (id, shortname, date, item, value, company_id)
            SELECT id, shortname, date, item, value, company_id FROM cashflowstatement_temp
            WHERE NOT EXISTS (
                SELECT * FROM cashflowstatement cs
                WHERE cs.id = cashflowstatement_temp.id AND cs.date = cashflowstatement_temp.date
                )
        """
        )
    conn = None
    try:
        # connect to the PostgreSQL server
        conn, cur = connect()
        # insert data into each table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logger.info(error)
    finally:
        if conn is not None:
            conn.close()
  
  
if __name__ == '__main__':
    insert_new_data()

conn, cur = connect()
# insert data into each table one by one
cur.execute("DROP TABLE balancesheet_temp, incomestatement_temp, cashflowstatement_temp, company_temp")
# close communication with the PostgreSQL database server
cur.close()
# commit the changes
conn.commit()