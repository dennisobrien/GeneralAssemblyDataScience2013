"""Python helper functions for Assignment 1.

Download the data files if necessary.
Munge the data to create a csv with header.

Datafiles from http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
"""
import os
import urllib

csv_filename = 'communities.csv'
names_filename = 'communities.names'
data_filename = 'communities.data'
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
names_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names'

def get_data_files():
    for url, filename in ((names_url, names_filename), (data_url, data_filename)):
        print("retrieving {}".format(url))
        urllib.urlretrieve(url, filename)

def get_headers():
    headers = []
    with open(names_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@attribute'):
                tag, var_name, datatype = line.split(' ', 2)
                headers.append(var_name)
    return headers

def write_csv(headers):
    with open(csv_filename, 'w') as f_out:
        with open(data_filename, 'r') as f_in:
            f_out.write('{}\n'.format(','.join(headers)))
            f_out.write(f_in.read())

def main():
    if not os.path.exists(names_filename) or not os.path.exists(data_filename):
        get_data_files()
    headers = get_headers()
    write_csv(headers)

if __name__=='__main__':
    main()
