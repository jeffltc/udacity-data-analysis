#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 11:35:19 2017

@author: jeffzhang
"""

## data outlier cleaner test

import sys
import pickle
sys.path.append("/Users/jeffzhang/GitHub/ud120-projects/tools")
sys.path.append("/Users/jeffzhang/GitHub/udacity-data-analysis/p5-project/final_project")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


features_list = ['poi','salary','bonus']

with open("/Users/jeffzhang/GitHub/udacity-data-analysis/p5-project/final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



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