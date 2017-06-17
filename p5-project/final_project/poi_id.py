#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
import numpy as np

# Feature List

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
                 'other']

# Parameters

components_parameter = 1 # PCA components
selector_percentile_parameter = 15 # SelectPercentile parameter
GridSearch_test = False
add_feature = True

algorithm = "SVM" # choose from "NB","Decision Tree","Random Forest","SVM"

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.

my_dataset = data_dict

## delete TOTAL
del_list = ['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']
for key in del_list:
    del my_dataset[key]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


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

## Feature Engineer

def add_feature(features):
    new_features = []
    print len(features[0])
    for ele in features:
        new_features.append(np.append(ele,((ele[3]+ele[2])/(ele[1]+ele[0]))))
        # Calculate the new feature with from 'to_messages', 'from_messages',
        # and 'from_this_person_to_poi','from_poi_to_this_person'
    print len(features[0])
    return new_features
        
def feature_scale(features):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

### feature selection

def feature_selection(features,labels,selector_percentile_parameter):
    from sklearn.feature_selection import SelectPercentile, f_classif
    selector = SelectPercentile(f_classif, percentile = \
                                selector_percentile_parameter)
    features = selector.fit_transform(features,labels)
    return features

### PCA

def feature_PCA(features,labels,components_parameter):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=components_parameter)
    features = pca.fit_transform(features,labels)
    return features

### data transform

def features_transform(features,labels,selector_percentile_parameter = selector_percentile_parameter,components_parameter = components_parameter,add_feature = False):
    print add_feature
    if add_feature:
        features = add_feature(features)
    features = feature_scale(features)
    features = feature_selection(features,labels,selector_percentile_parameter)
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




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


##validation

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

test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
