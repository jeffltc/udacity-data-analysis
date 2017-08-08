#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:53:41 2017

@author: newchama
"""

import pandas as pd
import os 
import matplotlib.pyplot as plt

df = pd.read_csv('../input/titanic_data.csv')

#不同性别的生存率

#print df.groupby('Sex')['Survived'].mean()



data = df['Age'].dropna()
plt.hist(data)
plt.title("Age")
plt.xlabel("Value")
plt.ylabel("Frequency")

fig = plt.gcf()