"""Perform k-means clustering on a dataset containing various health metrics
for countries as gathered by The World Bank.

http://data.worldbank.org/topic/health
http://api.worldbank.org/datafiles/8_Topic_MetaData_en_EXCEL.xls

Requirements:
pandas  Version 0.12+
xlrd    For Excel file reading.  (pip install xlrd)
"""

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

def get_code_to_name_map(df):
    return {key: value for (key, value), _group in df[['Indicator Code', 'Indicator Name']].groupby(['Indicator Code', 'Indicator Name'])}

def main():
    data_filepath = get_datafile()
    df = pd.read_excel(data_filepath, 'Sheet1')
    print df

if __name__=='__main__':
    main()
