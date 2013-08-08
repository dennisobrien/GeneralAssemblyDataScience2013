print 'importing modules'
from time import time
import logging
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people, make_circles
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA,PCA, KernelPCA
from sklearn.svm import SVC,LinearSVC
from sklearn.lda import LDA

DoKernelPCA = True
DoIrisScreen = True
DoIrisPCA = True
DoEigenFaces = True
SaveData = True

if SaveData:
    
def getFaceData():
    # Download the data, if not already on disk and load it as numpy arrays

    # insert code here 
    
    print "Total dataset size:"
    print "n_samples: %d" % n_samples
    print "n_features: %d" % n_features
    print "n_classes: %d" % n_classes
    return X,y,n_features,target_names,n_classes,n_samples,h,w

def pcaFaces(X,y,n_components):
    print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
    # insert code here
    print "done in %0.3fs" % (time() - t0)
    print "Projecting the input data on the eigenfaces orthonormal basis"
    # insert code here
    print "done in %0.3fs" % (time() - t0)
    return X_train_pca,X_test_pca,X_test,y_train,y_test,eigenfaces

def trainFaces(X_train_pca,y_train):
    print "Fitting the classifier to the training set"
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # insert code here
    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by grid search:"
    print clf.best_estimator_
    return clf

def predictFaces(X_test_pca,y_test,target_names,n_classes):
    print "Predicting the people names on the testing set"
    # insert code here
    print "done in %0.3fs" % (time() - t0)
    print 'Classification Report:'
    # insert code here
    print 'Confusion Matrix:'
    # insert code here
    return y_pred

###############################################################################
# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def screen_plot():
    print 'Calculating explained variance of PCA components of Iris dataset'
    # center data (important for dim reduction)

    # insert code here
    
    # get covariance matrix
    # this has the wrong dimensions

    # insert code here
    
    # eigenvalue decomp
    
    # insert code here

    # pct of variance explained by each principal component
    
    # insert code here

    pl.plot(pcts)
    pl.xlabel('principal cmpts')
    pl.ylabel('pct variance explained')
    pl.title('iris scree plot')

def pca_plot():
    
    # insert code here

    print 'Explained variance ratio (first two components):', \
        pca.explained_variance_ratio_
    pl.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        pl.scatter(X_decomp[y == i, 0], X_decomp[y == i, 1], c=c, label=target_name)
    pl.legend()
    pl.title('PCA of IRIS dataset')

def trainCircles():
    print 'Training on make_circles dataset'
    # insert code here
    return X,y,X_pca,X_kpca,X_back,kpca

def plotCircles(X,y,X_pca,X_kpca,X_back,kpca):
    pl.figure()
    pl.subplot(2, 2, 1, aspect='equal')
    pl.title("Original space")
    reds = y == 0
    blues = y == 1

    pl.plot(X[reds, 0], X[reds, 1], "ro")
    pl.plot(X[blues, 0], X[blues, 1], "bo")
    pl.xlabel("$x_1$")
    pl.ylabel("$x_2$")

    X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    # projection on the first principal component (in the phi space)
    Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
    pl.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

    pl.subplot(2, 2, 2, aspect='equal')
    pl.plot(X_pca[reds, 0], X_pca[reds, 1], "ro")
    pl.plot(X_pca[blues, 0], X_pca[blues, 1], "bo")
    pl.title("Projection by PCA")
    pl.xlabel("1st principal component")
    pl.ylabel("2nd component")

    pl.subplot(2, 2, 3, aspect='equal')
    pl.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
    pl.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
    pl.title("Projection by KPCA")
    pl.xlabel("1st principal component in space induced by $\phi$")
    pl.ylabel("2nd component")

    pl.subplot(2, 2, 4, aspect='equal')
    pl.plot(X_back[reds, 0], X_back[reds, 1], "ro")
    pl.plot(X_back[blues, 0], X_back[blues, 1], "bo")
    pl.title("Original space after inverse transform")
    pl.xlabel("$x_1$")
    pl.ylabel("$x_2$")

    pl.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

###Function Calls
if DoIrisScreen:
    print '\nRunning IRIS PCA screen'
    screen_plot()
    if SaveData:
        print 'Saving Iris Screen plot\n'
        pl.savefig('%s/Iris_screen.png' % dir_out)
    else:
        print 'Plotting'
        pl.show()

if DoIrisPCA:
    print '\nRunning Iris PCA'
    pca_plot()
    if SaveData:
        print 'Saving Iris PCA plot\n'
        pl.savefig('%s/Iris_pca.png' % dir_out)
    else:
        print 'Plotting'
        pl.show()

if DoKernelPCA:
    print '\nRunning Kernel PCA'
    X,y,X_pca,X_kpca,X_back,kpca = trainCircles()
    plotCircles(X,y,X_pca,X_kpca,X_back,kpca)
    if SaveData:
        print 'Saving Kernel PCA plot'
        pl.savefig('%s/MakeCircles_KernelPCA.png' % dir_out)
    else:
        print 'Plotting'
        pl.show()

if DoEigenFaces:
    print '\nRunning Faces PCA'
    n_components = 150
    X,y,n_features,target_names,n_classes,n_samples,h,w = getFaceData()
    X_train_pca,X_test_pca,X_test,y_train,y_test,eigenfaces = pcaFaces(X,y,n_components)
    clf = trainFaces(X_train_pca,y_train)
    y_pred = predictFaces(X_test_pca,y_test,target_names,n_classes)
    prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
    plot_gallery(X_test, prediction_titles, h, w)
    if SaveData:
        print 'Saving Faces plot'
        pl.savefig('%s/Faces.png' % dir_out)
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    if SaveData:
        print 'Saving Eigen Faces plot\n'
        pl.savefig('%s/Faces_eig.png' % dir_out)
    else:
        print 'Plotting'
        pl.show()
