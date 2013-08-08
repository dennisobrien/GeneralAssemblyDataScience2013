print 'importing modules'
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
import glob
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter

file = '/Users/jacobbollinger/Desktop/GeneralAssembly/python_examples/yahoo_data/FB_data.csv'
threshold = float(30)
cv = 10

###Use open, high, low, volume
test_data = ['28.73', '32.13', '28.96', '57668700']

def loadData(file):
    data = open(file).read()
    return data

def organizeData(data,threshold):
    data = data.split('\n')
    X = []
    y = []
    for d in data[1:-1]:
        tmp=[]
        d = d.split(',')
        tmp.append(d[1])
        tmp.append(d[2])
        tmp.append(d[3])
        tmp.append(d[5])
        X.append(tmp)
        if float(d[6])>=threshold:
            y.append(1)
        elif float(d[6])<threshold:
            y.append(0)
    return np.array(X),np.array(y)

def classify(X,y,cv):
    #clf = DecisionTreeClassifier()
    #clf = RandomForestClassifier()
    #clf = AdaBoostClassifier()
    clf = ExtraTreesClassifier()
    score = cross_val_score(clf, X, y, cv=cv)
    print '%s-fold cross validation accuracy: %s' % (cv,sum(score)/score.shape[0])
    clf = clf.fit(X,y)
    #print 'Feature Importances'
    #print clf.feature_importances_
    #X = clf.transform(X,threshold=.3)
    
    preds = clf.predict(X)
    print 'predictions counter'
    print Counter(clf.predict(X))
    fp=0
    tp=0
    fn=0
    tn=0
    for a in range(len(y)):
        if y[a]==preds[a]:
            if preds[a]==0:
                tn+=1
            elif preds[a]==1:
                tp+=1
        elif preds[a]==1:fp+=1
        elif preds[a]==0:fn+=1
    
    print 'correct positives:', tp
    print 'correct negatives:', tn
    print 'false positives:', fp
    print 'false negatives:', fn
    print 'precision:',float(tp)/(tp+fp)
    print 'recall (tp)/(tp+fn):',float(tp)/(tp+fn)
    print 'false positive rate (fp)/(fp+tn):', float(fp)/(fp+tn)
    print 'false positive rate2 (fp)/(fp+tp):', float(fp)/(fp+tp)
    print 'prediction accuracy: %s%s\n' % (100*float(tp+tn)/(tp+tn+fp+fn),'%') 
    return clf

def testOOS(clf,test_data,threshold):
    pred_proba = clf.predict_proba(test_data)
    pred = clf.predict(test_data)
    if float(pred[0])==float(1):
        print '***Classified as likely to close higher than $%s per share' % threshold
    elif float(pred[0])==float(0):
        print '***Classified as likely to close lower than $%s per share' % threshold
    print 'Probability of closing higher than $%s per share: %s' % (threshold,pred_proba[0][1])
    print 'Probability of closing lower than $%s per share: %s' % (threshold,pred_proba[0][0])

print 'loading data'
data = loadData(file)
print 'organizing data'
X,y = organizeData(data,threshold)
print 'training'
clf=classify(X,y,cv)
print 'testing oos data (open, high, low, volume): %s' % test_data
testOOS(clf,test_data,threshold)
print '\nDONE!!!'
