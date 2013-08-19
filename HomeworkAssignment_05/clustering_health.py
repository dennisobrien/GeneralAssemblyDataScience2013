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


def plot_histograms_for_each_column(df,n_columns=3, indicator_map=None, 
                                    n_points_threshold=20, verbose=False):
    n_subplots = sum([df.count()[column_name] > n_points_threshold for column_name in df.columns])
    n_rows = (n_subplots/n_columns) + 1
    plot_width = 6
    plot_height = 4
    figsize = (plot_width*n_columns, plot_height*n_rows)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=figsize)
    idx = 0
    for column_name in df.columns:
        n_points = df.count()[column_name]
        if n_points >= n_points_threshold:
            i_row = idx / n_columns
            i_col = idx % n_columns
            idx += 1
            if indicator_map is not None and column_name in indicator_map:
                column_description = indicator_map[column_name]
            else:
                column_description = column_name
            title = column_description
            df[column_name].hist(ax=axes[i_row,i_col])
            axes[i_row,i_col].set_title(title)
        else:
            if verbose:
                print("Skipping {}, too few points: {}".format(column_name, n_points))
    return fig

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
