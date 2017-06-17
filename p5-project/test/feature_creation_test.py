#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 05:53:49 2017

@author: jeffzhang
"""

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../final_project/")
from sklearn.svm import SVC
import numpy as np

with open("/Users/jeffzhang/GitHub/udacity-data-analysis/p5-project/final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
my_dataset = data_dict

from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi',
                'to_messages',
                'from_messages',
                 'from_this_person_to_poi',
                 'from_poi_to_this_person']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


new_feature = []
for ele in features:
    new_feature.append((ele[3]+ele[2])/(ele[1]+ele[0]))

#new_feature = np.array(new_feature)
#new_feature = np.reshape(new_feature,1,86)

clf = SVC(kernel = 'rbf',C=1,gamma = 1)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(new_feature, labels, test_size=0.4, random_state=42)

#clf = clf.fit(features_train,labels_train)


from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile = 40)

selector = selector.fit(features_train,labels_train)

print selector.scores_
