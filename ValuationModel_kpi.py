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

import warnings
warnings.filterwarnings('ignore')

