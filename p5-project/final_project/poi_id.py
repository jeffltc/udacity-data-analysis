#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus'] # You will need to use more features


'''
## complete feature list
complete_feature_list = []
for element in data_dict['YEAP SOON']:
    if element != 'email_address' or 
    complete_feature_list.append(element)
features_list = complete_feature_list
'''


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers


### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.

my_dataset = data_dict

## feature engineer


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## use panda to remove the outlier

def outlier_cleaner(data,percentage):
    import pandas as pd
    df = pd.DataFrame(data)
    filt_df = df.loc[:, df.columns != 0]
    quant_df = filt_df.quantile([percentage])
    filt_df = filt_df.apply(lambda x: x[(x<quant_df.loc[percentage,x.name])], axis=0)
    filt_df = filt_df = pd.concat([df.loc[:,0], filt_df], axis=1)
    filt_df.dropna(inplace = True)
    data = filt_df.as_matrix()
    return data

data = outlier_cleaner(data,0.95)


'''
## show data plot 
import numpy as np
import matplotlib.pyplot as plt

x = data[:,1]
y = data[:,2]

plt.scatter(x,y)
plt.show()
'''


## PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca = pca.fit(data)
data = pca.fit_transform(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


## test case split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.5, random_state=42)

'''
## remove outlier with regression

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression = regression.fit(features_train,labels_train)
'''

## import grid search cv to adjust kenel

from sklearn.model_selection import GridSearchCV

## GaussianNB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


'''
## Decision Tree

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
parameters = {'max_depth':[1,10000]}
clf = GridSearchCV(clf,parameters)
'''

'''
## Random Forest
from  sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

'''

## train clf
clf = clf.fit(features_train,labels_train)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

'''
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''

'''
## import precision score evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "precision score:" + str(precision_score(clf.predict(features_test),labels_test))
print "recall score:" + str(recall_score(clf.predict(features_test),labels_test))
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)