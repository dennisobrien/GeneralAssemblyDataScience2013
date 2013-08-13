"""
Using data from the Audobon Society Field Guide to Mushrooms, by way of the
UCI Machine Learning Repository, predict if a mushroom is edible or poisonous.

Disclaimer:  This is for research purposes only.  Do not ingest mushrooms
based on this model!

The data can be found at the UCI Machine Learning Repository:
http://archive.ics.uci.edu/ml/datasets/Mushroom

Requirements:
numpy
pandas
scikit-learn    http://scikit-learn.org/stable/index.html
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# set the random seed in order to reproduce results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# The data has been downloaded from the UCI Machine Learning Repository
# and edited to make parsing simpler.
data_filepath = "agaricus-lepiota.expanded.data"
column_names = ['classification', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 
                'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 
                'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 
                'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 
                'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 
                'population', 'habitat']

def get_data(filepath, column_names):
    """Return a tuple (X, y) where X is a Pandas DataFrame representing
    the parameters and y is a Pandas Series of the classes."""
    dataframe = pd.read_csv(filepath, names=column_names)
    X = dataframe[column_names[1:]]
    y = dataframe[column_names[0]]
    return X, y

def one_hot_dataframe(df, column_names, replace=False):
    """Given a DataFrame of string valued features, perform one-hot encoding.
    Return the original dataframe (optionally with the original column names removed),
    the vectorized dataframe, and the vectorizer object (useful for inverse transforming).
    Adapted from http://stackoverflow.com/questions/15021521/how-to-encode-a-categorical-variable-in-sklearn
    """
    vectorizer = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in column_names)
    vectorized_df = pd.DataFrame(vectorizer.fit_transform(df[column_names].apply(mkdict, axis=1)).toarray())
    vectorized_df.columns = vectorizer.get_feature_names()
    vectorized_df.index = df.index
    if replace is True:
        df = df.drop(column_names, axis=1)
        df = df.join(vectorized_df)
    return (df, vectorized_df, vectorizer)

def create_model(X, y, n_iter=10, test_size=0.1):
    # split the data in train and test using shuffle and split
    # create an iterator that generates boolean indices for each train/test run
    ss_iter = cross_validation.ShuffleSplit(len(X), 
                                            n_iter=n_iter, 
                                            test_size=test_size, 
                                            indices=False, 
                                            random_state=RANDOM_SEED)
    cm_combined = None
    for train_indices, test_indices in ss_iter:
        # converting these to lists is much faster than leaving in Pandas DataFrame or Series
        X_train = X[train_indices].to_records(index=False).tolist()
        y_train = y[train_indices].tolist()
        X_test = X[test_indices].to_records(index=False).tolist()
        y_test = y[test_indices].tolist()
        print(y_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        #print(model.coef_)
        #print(model.get_params())
        #print(model.transform(X_test[0:2]))
        #print(predicted.tolist())
        print(model.score(X_test, y_test))
        print("POISONOUS: {}".format(sum([val=='POISONOUS' for val in y_test])))
        print("EDIBLE:    {}".format(sum([val=='EDIBLE' for val in y_test])))
        cm = confusion_matrix(y_test, predicted)
        print(cm)
        if cm_combined is None:
            cm_combined = cm
        else:
            cm_combined += cm
    cm_df = pd.DataFrame(cm_combined, index=['edible', 'poisonous'], columns=['predicted edible', 'predicted poisonous'])
    print("combined confusion matrix:")
    print(cm_df)

def main(verbose=False):
    X, y = get_data(data_filepath, column_names)
    X, X_encoded, X_vectorizer = one_hot_dataframe(X, column_names[1:])
    create_model(X_encoded, y)

if __name__=='__main__':
    main(verbose=True)

