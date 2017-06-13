#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',
                 'salary',
                 'bonus',
                 'to_messages',
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
                 'from_poi_to_this_person',
                 'from_this_person_to_poi'] # You will need to use more features
                 
#test_size_parameter = 0.4

components_parameter = 2
selector_percentile_parameter = 30

GridSearch_test = False

## choose from 'KFold cross validation','StratifiedShuffleSplit'
cv_parameter = 'StratifiedShuffleSplit'
folds = 1000

## choose from "NB","Decision Tree","Random Forest","SVM"
algorithm = "Decision Tree"


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers


### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.

my_dataset = data_dict

## feature engineer

## delete TOTAL
del my_dataset['TOTAL']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## direct adjust data

## feature scaling

## use panda to remove the outlier

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


## import grid search cv to adjust kenel

from sklearn.model_selection import GridSearchCV


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
        clf = SVC(C=1,gamma = 1)
        if GridSearch_test:
            parameters = {'C':[0.001, 0.01, 0.1, 1, 10],
            "gamma":[0.001, 0.01, 0.1, 1]}
            clf = GridSearchCV(clf, parameters)
    return clf

clf = classifier(algorithm,True)
print clf.cv_results_


## test clf parameter
if GridSearch_test:
    
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=test_size_parameter, random_state=42)

    features_train = features_transform(features_train,labels_train)

    clf = clf.fit(features_train,labels_train)
    print clf.best_params_


##Feature engineer


## feature scale

def feature_scale(features):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

### feature selection

def feature_selection(features,labels,selector_percentile_parameter):
    from sklearn.feature_selection import SelectPercentile, f_classif
    selector = SelectPercentile(f_classif, percentile = selector_percentile_parameter)
    features = selector.fit_transform(features,labels)
    return features

### PCA

def feature_PCA(features,labels,components_parameter):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=components_parameter)
    features = pca.fit_transform(features,labels)
    return features

def features_transform(features,labels,selector_percentile_parameter = selector_percentile_parameter,components_parameter = components_parameter):
    features = feature_scale(features)
    features = feature_selection(features,labels,selector_percentile_parameter)
    features = feature_PCA(features,labels,components_parameter)
    return features



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

'''
# Example starting point. Try investigating other evaluation techniques!

'''
'''
def calculate_score(clf,features,labels,print_score = False):
    ## import precision score evaluation
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    
    accuracy = accuracy_score(clf.predict(features),labels)
    precision = precision_score(clf.predict(features),labels)
    recall = recall_score(clf.predict(features),labels)
    
    if print_score:
        print clf
        print "accuracy score: {}".format(accuracy)
        print "precision score: {}".format(precision)
        print "recall score: {}".format(recall)
    
    scores = {"accuracy":accuracy,
           "precision":precision,
           "recall":recall
            }
    return scores
'''

def cross_validation(cv_parameter,features,folds):
    if cv_parameter == 'KFold cross validation':
        from sklearn.model_selection import KFold
        kf = KFold(folds)
        cv = kf.split(features)
    elif cv_parameter == 'StratifiedShuffleSplit':
        from sklearn.cross_validation import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    return cv

cv = cross_validation(cv_parameter,features,folds)



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


test_classifier(clf, my_dataset, features_list, transform = True)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
