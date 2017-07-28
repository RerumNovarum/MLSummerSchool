#!/usr/bin/python

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh.plotting
import catboost
import sklearn.preprocessing, sklearn.feature_selection, sklearn.model_selection
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.base
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import dateutil.parser
import collections
import sklearn.utils
import itertools
import re
from sklearn.model_selection import train_test_split

from water_transformers import *

SEED = 42
np.random.seed = SEED

y_enc = sklearn.preprocessing.LabelEncoder()
y_enc.fit(pd.Series.from_csv('out_classes.csv'))

X_tr, X_cv, X_va, y_tr, y_cv, y_va, X_te = [
    pd.read_csv(fname) for fname in ['X_tr.csv', 'X_cv.csv', 'X_va.csv',
                                     'y_tr.csv', 'y_cv.csv', 'y_va.csv',
                                     'X_te.csv',]
]

datasets = [X_tr, X_cv, X_va, X_te]
for ds in datasets:
    ds.drop(['wpt_name', 'num_private', 'funder', 'date_recorded_year', 'date_recorded_month', 'date_recorded_day', 'date_recorded_weekday', ], errors='ignore', axis=1, inplace=True)

prepr_and_catboost(X_tr, X_cv, X_va, y_tr, y_cv, y_va, X_te,
                   outfname='ans1.csv',
                   prepr=None)
