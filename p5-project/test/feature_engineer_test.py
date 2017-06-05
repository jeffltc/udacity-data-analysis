#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:18:54 2017

@author: newchama
"""

#!/usr/bin/python

import sys
import pickle

sys.path.append("../final_project/")
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','to_messages','deferral_payments',
                 'total_payments','exercised_stock_options',
                 'bonus','restricted_stock','shared_receipt_with_poi',        
                 'restricted_stock_deferred','total_stock_value','expenses',
                 'director_fees','deferred_income',
                 'long_term_incentive'] # You will need to use more features
                 
test_size_parameter = 0.4
components_parameter = 2
#outlier_parameter = 0.995
selector_percentile_parameter = 20
## choose from "NB","Decision Tree","Random Forest","SVM"
algorithm = "Decision Tree"




### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


my_dataset = data_dict

## feature engineer



## delete TOTAL
del my_dataset['TOTAL']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


## direct adjust data

## feature selection
from sklearn.feature_selection import SelectPercentile, f_classif

selector = SelectPercentile(f_classif, percentile = 50)

selector.fit(features,labels)

print "feature selected {}".format(len(selector.scores_))
print "feature score {}".format(selector.scores_)

features_train = selector.transform(features)


## PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=components_parameter)
pca = pca.fit(features)

print "PCA {}".format(len(pca.components_))
features_train = pca.fit_transform(features)




## test case split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=test_size_parameter, random_state=42)

print "feature length: {}".format(len(features_train))



from sklearn.model_selection import GridSearchCV


def classifier(algorithm):
    if algorithm == 'NB':
        ## GaussianNB
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    elif algorithm == 'Decision Tree':
        ## Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        
        ##best classifier
        clf = DecisionTreeClassifier(criterion = "entropy",max_depth = 2,min_samples_leaf = 9)
        
        '''
        parameters = {'max_depth':(1,5,1),'min_samples_leaf':(1,10,1)}
        print clf.best_params_
        clf = GridSearchCV(clf,parameters)
        '''
        
    elif algorithm == 'Random Forest':
        ## Random Forest
        from  sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 3)
        
        '''
        parameters = {'n_estimators':[1,2]}
        clf = GridSearchCV(clf,parameters)
        '''
    elif algorithm == 'SVM':
        from sklearn.svm import SVC
        clf = SVC()
    return clf

print "feature length before fit: {}".format(len(features_train))


## train clf
clf = classifier(algorithm)
clf = clf.fit(features_train,labels_train)


## import precision score evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

'''
print precision_score(clf.predict(pca.fit_transform(selector.transform(features_test))),labels_test)
print recall_score(clf.predict(pca.fit_transform(selector.transform(features_test))),labels_test)
'''

print precision_score(clf.predict(features_test),labels_test)
print recall_score(clf.predict(features_test),labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)