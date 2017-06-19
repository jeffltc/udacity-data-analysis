#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
my_dataset = data_dict

del_list = ['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']
for key in del_list:
    del my_dataset[key]
    
    
### add new feature to dataset

for element in my_dataset:
    record = my_dataset[element]
    to_messages = record['to_messages']
    from_messages = record['from_messages']
    from_this_person_to_poi = record['from_this_person_to_poi']
    from_poi_to_this_person = record['from_poi_to_this_person']
    if  to_messages == 0 or \
    from_messages == 'NaN' or \
    from_this_person_to_poi == 'NaN' or \
    from_poi_to_this_person == 'NaN':
        record['poi_email_rate'] = 'NaN'
    else:
        record['poi_email_rate'] = float(from_this_person_to_poi+from_poi_to_this_person)/(to_messages+from_messages)

### feature_selection

def feature_selection(percent,print_score = False):
    features_list = ['poi',
                 'to_messages',
                 'from_messages',
                 'from_this_person_to_poi',
                 'from_poi_to_this_person',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'other',
                 'poi_email_rate']
    # split data
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    # cross validation
    score_list = []
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        selector = SelectPercentile()
        selector = selector.fit(features_train,labels_train)
        score_list.append(selector.scores_)
    
    score_average = sum(score_list)/len(score_list)
    feature_score = zip(features_list[1:],score_average)
    feature_score.sort(key = lambda tup:tup[1],reverse=True)
    
    # create new feature list
    new_feature_list = ['poi']
    feature_score = feature_score[:int(len(feature_score)*percent/100)]
    for ele,_ in feature_score:
        new_feature_list.append(ele)
    if print_score == True:
        print feature_score
    return new_feature_list


## delete TOTAL


## Prepare for the Algorithm
def classifier(algorithm, GridSearch_test = False):
    if algorithm == 'NB':
        ## GaussianNB
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    elif algorithm == 'Decision Tree':
        ## Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion = "entropy",max_depth = 2,min_samples_leaf = 9)  
        if GridSearch_test:
            parameters = {'criterion':["entropy","gini"],'max_depth':(1,10,1),'min_samples_leaf':(1,200,10)}
            clf = GridSearchCV(clf,parameters)
    elif algorithm == 'Random Forest':
        ## Random Forest
        from  sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 3)
        if GridSearch_test:
            parameters = {'n_estimators':[1,10]}
            clf = GridSearchCV(clf,parameters)
    elif algorithm == 'SVM':
        from sklearn.svm import SVC
        clf = SVC(kernel = 'rbf',C=1,gamma = 1)
        if GridSearch_test:
            parameters = {'C':[0.001, 0.01, 0.1, 1, 10],
            "gamma":[0.001, 0.01, 0.1, 1]}
            clf = GridSearchCV(clf, parameters)
    return clf

clf = classifier(algorithm,False)

### feature scale

def feature_scale(features):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features


### PCA

def feature_PCA(features,labels,components_parameter):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=components_parameter)
    features = pca.fit_transform(features,labels)
    return features

### data transform

def features_transform(features,labels,selector_percentile_parameter = selector_percentile_parameter,components_parameter = components_parameter,add_feature = True):
    print add_feature
    features = feature_scale(features)
    features = feature_PCA(features,labels,components_parameter)
    return features


## Test Clf Parameter
if GridSearch_test:
    
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.4, random_state=42)
    features_train = features_transform(features_train,labels_train,add_feature = True)
    clf = clf.fit(features_train,labels_train)
    print clf.best_estimator_


## Cross Validation

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

accuracy = []
precision = []
recall = []

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0


# Parameters
components_parameter = 1 # PCA components
percent = 5 # SelectPercentile parameter
GridSearch_test = False

algorithm = "SVM" # choose from "NB","Decision Tree","Random Forest","SVM"

# Select Feature and Train Clf

features_list = feature_selection(percent)
test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)
