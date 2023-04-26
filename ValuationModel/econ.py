import datetime
from datetime import date
from datetime import datetime 
import pandas as pd
import requests

def getGdpData(country='US', period='A', key='NGDP_R_XDC'):
    # possible countries to check via metadata settings under advanced and exporting it into excel via this link, checking sheet "tooltip": https://data.imf.org/?sk=4c514d48-b6ba-49ed-8ab9-52b0c1a0179b&sId=1390030341854
    # U2 = Euro Area
    # US = USA
    # FR = Frankreich
    # DE = Deutschland
    url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/'
    period = period
    country = country
    today_year = date.today().strftime('%Y')
    time = f'?startPeriod=2001&endPeriod={today_year}'
    key = key # see IFS excel file in data/; Gross Domestic Product, Expenditure Approach, Real, Percent Change, Corresponding Period Previous Year, Seasonally adj, percent

    # Navigate to series in API-returned JSON data
    link = f"{url}{period}.{country}.{key}.{time}"
    data = (requests.get(link).json()['CompactData']['DataSet']['Series'])

    # Now clean the data and get it into dataframe
    baseyr = data['@BASE_YEAR']  # Save the base year

    # Create pandas dataframe from the observations
    data_list = [[obs.get('@TIME_PERIOD'), obs.get('@OBS_VALUE')]
                for obs in data['Obs']]

    df = pd.DataFrame(data_list, columns=['date', 'value'])
    df['value'] = df['value'].astype(float)
        
    #df = df.set_index(pd.to_datetime(df['date']))['value'].astype('float')
    df['real_growth'] = df[['value']].pct_change()
    mean_gdp_growth = df['real_growth'].mean()
    return mean_gdp_growth
#mean_gdp_growth = getGdpData(country='U2', period='A', key='NGDP_R_XDC')
#print(mean_gdp_growth)