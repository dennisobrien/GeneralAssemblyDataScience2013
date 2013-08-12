"""
Using data from the Audobon Society Field Guide to Mushrooms, by way of the
UCI Machine Learning Repository, predict if a mushroom is edible or poisonous.

Disclaimer:  This is for research purposes only.  Do not ingest mushrooms
based on this model!

The data can be found at the UCI Machine Learning Repository:
http://archive.ics.uci.edu/ml/datasets/Mushroom

Requirements:
numpy
requests        http://docs.python-requests.org/en/latest/
scikit-learn    http://scikit-learn.org/stable/index.html
"""

import numpy as np
import os
import re
import requests
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

names_uri = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names"
data_uri = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
data_filepath = os.path.basename(data_uri)


def get_attribute_dict():
    """Return a dictionary mapping the column name to a dictionary containing
    values for the column index and the value dictionary for that field.
    This is based on the description available in the names file which serves as a code book.
    We parse this text rather than hard code it.
    """
    attribute_description = """
0. classification:           edible=e,poisonous=p
1. cap-shape:                bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
4. bruises:                  bruises=t,no=f
5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
6. gill-attachment:          attached=a,descending=d,free=f,notched=n
7. gill-spacing:             close=c,crowded=w,distant=d
8. gill-size:                broad=b,narrow=n
9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
10. stalk-shape:              enlarging=e,tapering=t
11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
16. veil-type:                partial=p,universal=u
17. veil-color:               brown=n,orange=o,white=w,yellow=y
18. ring-number:              none=n,one=o,two=t
19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
21. population:               abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
"""
    regex = re.compile("""(?P<column_index>\d+).\s+(?P<column_name>[\w-]+)\:\s+(?P<values>[\w,=-]+)""")
    dct = {}
    for line in attribute_description.split('\n'):
        match = regex.match(line)
        if match:
            column_index = int(match.group('column_index'))
            column_name = match.group('column_name').replace('-', '_')
            values = match.group('values')
            value_dict = {}
            for value, symbol in [v.split('=') for v in values.split(',')]:
                value_dict[symbol] = value
            #print(column_index, column_name, value_dict)
            dct[column_name] = {'column_index': column_index, 'value_dict': value_dict}
    return dct

def get_attributes_from_dict(dct):
    attributes = []
    for key, value in sorted(dct.iteritems(), key=lambda (k,v): (v['column_index'],k)):
        attributes.append(key)
    return attributes

def retrieve_datafiles():
    """Retrieve the data files if necessary."""
    if not os.path.exists(data_filepath):
        print("retrieving data from {}".format(data_uri))
        r = requests.get(data_uri)
        with open(data_filepath, 'w') as f:
            f.write(r.text)

def get_data(filepath, column_names):
    """Return a tuple (X, y) where X is a numpy matrix representing
    the parameters and y is a numpy array of the classes."""
    data = np.genfromtxt(filepath, delimiter=',', names=column_names,
                         dtype=[(_, np.dtype('S1')) for _ in column_names])
    X = data[column_names[1:]]
    y = data[column_names[0]]
    return X, y
    
def create_model(X_train, y_train, X_test, y_test):
    # encode the categorical features as binary features
#     v = DictVectorizer(sparse=False)
#     D = []
#     for column_name in column_names[1:]:
#         D.append({feature:value for value, feature in enumerate(set(X[column_name].tolist()))})
# 
#     for d in D:
#         print(d)
    # encode the categorical independent variables
    encoder = preprocessing.OneHotEncoder()
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(predicted)
    
def main(verbose=False):
    retrieve_datafiles()
    attribute_dict = get_attribute_dict()
    column_names = get_attributes_from_dict(attribute_dict)
    if verbose:
        print(column_names)
    X, y = get_data(data_filepath, column_names)
    print(X)
    print(y)
    create_model(X, y, X, y)

if __name__=='__main__':
    main()

