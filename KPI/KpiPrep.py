import pandas as pd
import numpy
from ValuationModel.fmp import get_latest_available_fy, ratio
from ValuationModel.assist_functions import prepare_timeseries


class MakeKpiDf:

    def __init__(self, stock=''):
        self.stock = stock
    
    
    @staticmethod
    def year_list(stock):
        latest_year=get_latest_available_fy(ticker=stock, json_entry="date")
        years=[]
        for y in range(0,5):
            y=latest_year-y
            years.append(y)
        years.reverse()
        return years

    def select_kpi(self, statement, item='', spec='standard'):
        if spec == 'standard':
            df = prepare_timeseries(statement, item=item)
            kpi_list = df['value'].tolist()[0:5]
        else:
            df = ratio(self.stock, json_entry=item)
            kpi_list = [df[item] for df in df][0:5]
        
        kpi_list.reverse()
        return kpi_list

    
    def make_kpi_df(self, statement, item_list, name='', spec='standard'):
        
        years_list = self.year_list(self.stock)
        d_list=[]
        for i in item_list:
            if spec == 'standard':
                kpi_5y = self.select_kpi(statement, item=i, spec='standard')
            else:
                kpi_5y = self.select_kpi(statement, item=i, spec='specific')
            d = {}
            for year, value in zip(years_list, kpi_5y):
                if year not in d.keys():
                    d[year] = value
                else:
                    d[year].append(value)
            d_list.append(d)

        kpi_df = pd.DataFrame(d_list)
        kpi_df[name]=item_list
        kpi_df=kpi_df.set_index([name])
        return kpi_df