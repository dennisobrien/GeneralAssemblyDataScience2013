"""Perform k-means clustering on a dataset containing various health metrics
for countries as gathered by The World Bank.

http://data.worldbank.org/topic/health
http://api.worldbank.org/datafiles/8_Topic_MetaData_en_EXCEL.xls

Requirements:
pandas  Version 0.12+
xlrd    For Excel file reading.  (pip install xlrd)
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import requests


def get_datafile():
    datafile_url = 'http://api.worldbank.org/datafiles/8_Topic_MetaData_en_EXCEL.xls'
    filepath = os.path.basename(datafile_url)
    if not os.path.exists(filepath):
        print("downloading {}".format(datafile_url))
        r = requests.get(datafile_url, stream=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print("downloaded to {}".format(filepath))
    return filepath

def get_indicator_code_to_name_map(df):
    """Return a dictionary which maps the indicator code to the indicator name."""
    return {key: value for (key, value), _group in df[['Indicator Code', 'Indicator Name']].groupby(['Indicator Code', 'Indicator Name'])}

def get_country_code_to_name_map(df):
    """Return a dictionary which maps the country code to the country name."""
    return {key: value for (key, value), _group in df[['Country Code', 'Country Name']].groupby(['Country Code', 'Country Name'])}

def get_aggregated_country_code_list():
    """Some country codes apply to aggregated regions. This is useful for droping some rows."""
    return ('ARB', 'CSS', 'EAS', 'EAP', 'EMU', 'ECS', 'ECA', 'EUU', 'HPC', 
            'HIC', 'NOC', 'OEC', 'LCN', 'LAC', 'LDC', 'LMY', 'LIC', 'LMC',
            'MEA', 'MNA', 'MIC', 'NAC', 'INX', 'OED', 'OSS', 'PSS', 'SST',
            'SAS', 'SSF', 'SSA', 'UMC', 'WLD')

def main():
    data_filepath = get_datafile()
    df = pd.read_excel(data_filepath, 'Sheet1')
    print df

if __name__=='__main__':
    main()
