"""
Using data from the Audobon Society Field Guide to Mushrooms, by way of the
UCI Machine Learning Repository, predict if a mushroom is edible or poisonous.

Disclaimer:  This is for research purposes only.  Do not ingest mushrooms
based on this model!

The data can be found at the UCI Machine Learning Repository:
http://archive.ics.uci.edu/ml/datasets/Mushroom

Requirements:
numpy
pandas version 0.12+
scikit-learn    http://scikit-learn.org/stable/index.html
"""

import itertools
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Silence the deprecation warnings from Pandas 0.12
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# set the random seed in order to reproduce results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LABEL_ACTUAL_POSITIVE = 'edible'
LABEL_ACTUAL_NEGATIVE = 'poisonous'
LABEL_PREDICTED_POSITIVE = 'predicted edible'
LABEL_PREDICTED_NEGATIVE = 'predicted poisonous'

# The data has been downloaded from the UCI Machine Learning Repository
# and edited to make parsing simpler.
data_filepath = "agaricus-lepiota.expanded.data"
csv_column_names = ['classification', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 
                'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 
                'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 
                'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 
                'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 
                'population', 'habitat']

def get_precision(confusion_df):
    #tp = float(confusion_df.iloc[0, 0])
    #fp = float(confusion_df.iloc[1, 0])
    tp = float(confusion_df[LABEL_PREDICTED_POSITIVE][LABEL_ACTUAL_POSITIVE])
    fp = float(confusion_df[LABEL_PREDICTED_POSITIVE][LABEL_ACTUAL_NEGATIVE])
    precision = tp / (tp + fp)
    return precision

def get_recall(confusion_df):
    #tp = float(confusion_df.iloc[0, 0])
    #fn = float(confusion_df.iloc[0, 1])
    tp = float(confusion_df[LABEL_PREDICTED_POSITIVE][LABEL_ACTUAL_POSITIVE])
    fn = float(confusion_df[LABEL_PREDICTED_NEGATIVE][LABEL_ACTUAL_POSITIVE])
    recall = tp / (tp + fn)
    return recall

def plot_confusion_matrix_dict(dct, min_precision=0, min_recall=0, auto_scale=False):
    data = {}
    features = dct.keys()
    data['feature'] = [','.join(feature) for feature in features]
    data['precision'] = [get_precision(dct[feature]) for feature in features]
    data['recall'] = [get_recall(dct[feature]) for feature in features]
    df = pd.DataFrame(data)
    df = df[(df['precision']>=min_precision) & (df['recall']>=min_recall)]
    #print(df)
    ax = df.plot(x='precision', y='recall', figsize=(10,8), style='.')#, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    if not auto_scale:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('precision (tp / (tp + fp))')
    ax.set_ylabel('recall (tp / (tp + fn))')
    ax.set_title('Precision vs. Recall for Given Features')
    for _i, row in df.iterrows():
        ax.text(row['precision'], row['recall'], row['feature'], fontsize=8)
    return df

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

def create_and_test_model(X, y, n_iter=10, test_size=0.1, random_state=RANDOM_SEED, verbose=False):
    """Create a model and test using n-fold cross validation.
    Pass random_state=None to override the fixed random seed.
    """
    # split the data in train and test using shuffle and split
    # create an iterator that generates boolean indices for each train/test run
    ss_iter = cross_validation.ShuffleSplit(len(X), 
                                            n_iter=n_iter, 
                                            test_size=test_size, 
                                            indices=False, 
                                            random_state=random_state)
    cm_combined = None
    for n_run, (train_indices, test_indices) in enumerate(ss_iter):
        # converting these to lists is much faster than leaving in Pandas DataFrame or Series
        X_train = X[train_indices].to_records(index=False).tolist()
        y_train = y[train_indices].tolist()
        X_test = X[test_indices].to_records(index=False).tolist()
        y_test = y[test_indices].tolist()
        #print(y_test)
        model = LogisticRegression(penalty='l2')
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        cm = confusion_matrix(y_test, predicted)
        cm_df = pd.DataFrame(cm, index=[LABEL_ACTUAL_POSITIVE, LABEL_ACTUAL_NEGATIVE], columns=[LABEL_PREDICTED_POSITIVE, LABEL_PREDICTED_NEGATIVE])
        if cm_combined is None:
            cm_combined = cm
        else:
            cm_combined += cm
        if verbose:
            #print(model.coef_)
            #print(model.get_params())
            #print(model.transform(X_test[0:2]))
            #print(predicted.tolist())
            print("run {} of {}".format(n_run+1, n_iter))
            print("\t" "score: {}".format(model.score(X_test, y_test)))
            print("\t" "POISONOUS: {}".format(sum([val=='POISONOUS' for val in y_test])))
            print("\t" "EDIBLE:    {}".format(sum([val=='EDIBLE' for val in y_test])))
            print("\t" "confusion matrix:\n{}\n".format(cm_df))
    cm_df = pd.DataFrame(cm_combined, index=[LABEL_ACTUAL_POSITIVE, LABEL_ACTUAL_NEGATIVE], columns=[LABEL_PREDICTED_POSITIVE, LABEL_PREDICTED_NEGATIVE])
    if verbose:
        print("combined confusion matrix:")
        print(cm_df)
    return cm_df

def test_all_features(X, y, column_names, verbose=False):
    """Create and test a model using all available features."""
    X, X_encoded, X_vectorizer = one_hot_dataframe(X, column_names)
    cm_df = create_and_test_model(X_encoded, y, verbose=verbose)
    return cm_df

def test_n_features(X, y, column_names, n_features=1, verbose=False):
    """Choose n_features at a time and run the model based only on those features.
    Return a dictionary mapping tuples of feature names to confusion matrix."""
    confusion_matrix_dict = {}
    for feature_names in itertools.combinations(column_names, n_features):
        feature_names = list(feature_names) # make pandas happy
        X, X_encoded, X_vectorizer = one_hot_dataframe(X, feature_names)
        cm_df = create_and_test_model(X_encoded, y, verbose=verbose)
        print("features: {}".format(feature_names))
        print("confusion matrix:\n{}\n\n".format(cm_df))
        confusion_matrix_dict[tuple(feature_names)] = cm_df
    return confusion_matrix_dict

def get_n_best_features(X_train, y_train, X_test, n_features):
    # FIXME: This could be more general by just returning the trained model
    model = SelectKBest(k=n_features)
    X_train_transformed = model.fit_transform(X_train, y_train)
    X_test_transformed = model.transform(X_test)
    mask = model.get_support()
    return X_train_transformed, X_test_transformed, mask

def test_top_features(X, y, column_names, n_features=10, verbose=False):
    """Use SelectKBest to determine the most predictive features.
    Then use the reduced version of the data to train and predict.
    Return a dataframe of the confusion matrix.
    """
    X, X_encoded, X_vectorizer = one_hot_dataframe(X, column_names)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_encoded,
                                                                         y,
                                                                         test_size=0.4,
                                                                         random_state=RANDOM_SEED)
    X_train_reduced, X_test_reduced, feature_mask = get_n_best_features(X_train, 
                                                                        y_train, 
                                                                        X_test,
                                                                        n_features)
    features = tuple(X_encoded.columns[feature_mask].tolist())
    if verbose:
        print("Using features: {}".format(features))
    model = LogisticRegression(penalty='l2')
    model.fit(X_train_reduced, y_train)
    predicted = model.predict(X_test_reduced)
    cm = confusion_matrix(y_test, predicted)
    cm_df = pd.DataFrame(cm, index=[LABEL_ACTUAL_POSITIVE, LABEL_ACTUAL_NEGATIVE], columns=[LABEL_PREDICTED_POSITIVE, LABEL_PREDICTED_NEGATIVE])
    if verbose:
        print(cm_df)
    return cm_df, features

def test_top_features_in_range(X, y, column_names, min_features=1, max_features=10, verbose=False):
    """Call test_top_features repeatedly with k_features in the range [min_features, max_features].
    Return a dictionary mapping features (tuple) to confusion matrix (dataframe)
    """
    data = {}
    for n_features in range(min_features, max_features+1):
        cm_df, features = test_top_features(X, y, column_names, n_features=n_features, verbose=verbose)
        data[features] = cm_df
    return data

def plot_top_features_dict(dct):
    data = {}
    features = dct.keys()
    #data['n_features'] = [len(feature) for feature in features]
    data['precision'] = [get_precision(dct[feature]) for feature in features]
    data['recall'] = [get_recall(dct[feature]) for feature in features]
    df = pd.DataFrame(data, index=[len(feature) for feature in features])
    df.sort(inplace=True)
    print(df)
    df.plot(figsize=(10,8), 
            title='Precision and Recall as a function of the Number of Features Selected',
            ylim=(0.0, 1.0))

def main(verbose=False):
    X, y = get_data(data_filepath, csv_column_names)
    test_all_features(X, y, csv_column_names[1:])
    confusion_matrix_dict = test_n_features(X, y, csv_column_names[1:], n_features=1)
    plot_confusion_matrix_dict(confusion_matrix_dict)
    test_top_features(X, y, csv_column_names[1:])

if __name__=='__main__':
    main(verbose=True)


