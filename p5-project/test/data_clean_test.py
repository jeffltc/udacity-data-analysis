#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:23:16 2017

@author: jeffzhang
"""

import sys
import pickle
sys.path.append("../final_project/")
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.

my_dataset = data_dict

import pandas as pd

df = pd.DataFrame(my_dataset)