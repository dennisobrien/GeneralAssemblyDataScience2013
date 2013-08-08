import numpy as np
import pylab as pl
from sklearn import svm, datasets

def loadData():

def classify(X,Y):
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors

def setUpMesh():
    # create a mesh to plot in

def plotResult(xx,yy,svc,rbf_svc,poly_svc,lin_svc):
    # title for the plots
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel',
              'LinearSVC (linear kernel)']


    for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        
##function calls
print 'loading data'
X,Y = loadData()
print 'setting up classifiers'
svc,rbf_svc,poly_svc,lin_svc = classify(X,Y)
print 'setting up mash'
xx,yy = setUpMesh()
print 'plotting'
plotResult(xx,yy,svc,rbf_svc,poly_svc,lin_svc)

