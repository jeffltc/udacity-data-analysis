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

with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
my_dataset = data_dict

from feature_format import featureFormat, targetFeatureSplit

#features_list = ['poi',
#                'to_messages',
#                'from_messages',
#                 'from_this_person_to_poi',
#                 'from_poi_to_this_person']

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



data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print len(features[0])

new_features = []
for ele in features:
    if ele[1]!= 0 and ele[0] !=0:
        new_feature = ((ele[3]+ele[2])/(ele[1]+ele[0]))
    else:
        new_feature = 0
    new_features.append(np.append(ele,new_feature))

#    print np.append(ele,((ele[3]+ele[2])/(ele[1]+ele[0])))
#    print len(np.append(ele,((ele[3]+ele[2])/(ele[1]+ele[0]))))

clf = SVC(kernel = 'rbf',C=1,gamma = 1)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(new_features, labels, test_size=0.4, random_state=42)

#clf = clf.fit(features_train,labels_train)


from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile = 15)

selector = selector.fit(features_train,labels_train)

score_list = selector.scores_

for score in score_list:
    print score