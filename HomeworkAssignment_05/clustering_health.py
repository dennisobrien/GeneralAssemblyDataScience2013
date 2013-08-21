"""Perform k-means clustering on a dataset containing various health metrics
for countries as gathered by The World Bank.

http://data.worldbank.org/topic/health
http://api.worldbank.org/datafiles/8_Topic_MetaData_en_EXCEL.xls

Requirements:
pandas  Version 0.12+
xlrd    For Excel file reading.  (pip install xlrd)
"""

from itertools import product
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans


np.random.seed(42)

def cluster_kmeans(df, n_clusters=10, scale_data=True):
    df = df.dropna()
    data = df.as_matrix()
    if scale_data:
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
    model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    labels = model.fit_predict(data)
    #print("labels: {}".format(labels))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    for idx in range(n_clusters):
        df[labels==idx].plot(x=df.columns[0], y=df.columns[1], style='.', label=idx, ax=ax)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend(loc='best')
    df_clusters = pd.DataFrame([{'Country Code': country_code, 'label': label} for country_code, label in zip(df.index, labels)])
    return fig, df_clusters

def get_optimal_clusters(df, max_clusters=30, scale_data=True):
    df = df.dropna()
    data = df.as_matrix()
    if scale_data:
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
    score_data = []
    min_clusters = 2    # silhouette_score assumes there are at least two clusters
    for n_clusters in range(min_clusters, max_clusters+1):
        model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        labels = model.fit_predict(data)
        #print("clusters: {}, labels: {}".format(n_clusters, labels))
        score = metrics.silhouette_score(data, labels)#, metric='euclidean')
        score_data.append({'n_clusters':n_clusters, 'silhouette_score':score})
    score_df = pd.DataFrame(score_data)
    ax = score_df.plot(x='n_clusters', y='silhouette_score')
    ax.set_ylabel('silhouette_score')

def plot_scatter_for_each_pair(df, x_column_names, y_column_names=None,
                               n_columns=3,
                               n_points_threshold=20,
                               verbose=False):
    if y_column_names is None:
        y_column_names = df.columns
    # figure out how many subplots we will be displaying
    n_subplots = sum([min(df.count()[column_x], df.count()[column_y]) > n_points_threshold for column_x, column_y in product(x_column_names, y_column_names)])
    n_rows = int(math.ceil(1.0 * n_subplots/n_columns))
    plot_width = 6
    plot_height = 4
    figsize = (plot_width*n_columns, plot_height*n_rows)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=figsize)
    idx = 0
    for x_column_name in x_column_names:
        for y_column_name in y_column_names:
            title = "{} vs. {}".format(x_column_name, y_column_name)
            try:
                n_points = min(df.count()[x_column_name], df.count()[y_column_name])
                if n_points >= n_points_threshold:
                    i_row = idx / n_columns
                    i_col = idx % n_columns
                    idx += 1
                    df.plot(x=x_column_name, y=y_column_name, style='.', title=title, ax=axes[i_row,i_col])
                    axes[i_row, i_col].set_ylabel(y_column_name)
                else:
                    if verbose:
                        print("too few points for {}: {}".format(y_column_name, n_points))
            except KeyError:
                if verbose:
                    print "Error creating graph: {}".format(title)
    plt.tight_layout()
    return fig

def plot_histograms_for_each_column(df, column_names=None, n_columns=3, indicator_map=None, 
                                    n_points_threshold=20, verbose=False):
    if column_names is None:
        column_names = df.columns
    # figure out how many subplots we will be displaying
    n_subplots = sum([df.count()[column_name] > n_points_threshold for column_name in column_names])
    n_rows = int(math.ceil(1.0 * n_subplots/n_columns))
    plot_width = 6
    plot_height = 4
    figsize = (plot_width*n_columns, plot_height*n_rows)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=figsize)
    idx = 0
    for column_name in column_names:
        n_points = df.count()[column_name]
        if n_points >= n_points_threshold:
            i_row = idx / n_columns
            i_col = idx % n_columns
            idx += 1
            if indicator_map is not None and column_name in indicator_map:
                column_description = indicator_map[column_name]
            else:
                column_description = column_name
            title = "\n(".join(column_description.split('('))
            df[column_name].hist(ax=axes[i_row,i_col])
            axes[i_row,i_col].set_title(title)
        else:
            if verbose:
                print("Skipping {}, too few points: {}".format(column_name, n_points))
    plt.tight_layout()
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
