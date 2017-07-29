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

SEED = 42
np.random.seed = SEED

def show_intersections(A, B, add_cat_feats=['date_recorded', 'construction_year',
                                            'date_recorded_year', 'date_recorded_month']):
    cat_features = np.where(np.array([(X_train[c].dtype == object)
                                  or (c.endswith('code'))
                                  or (c in add_cat_feats) for c in X_train.columns.values]))[0]
    cat_feats_observed = []
    for c in cat_features:
        n_uni = len(set(A.iloc[:,c]).union(B.iloc[:,c]))
        n_int = len(set(A.iloc[:,c]).intersection(B.iloc[:,c]))
        n_te = len(set(A.iloc[:,c]))
        cat_feats_observed.append((A.columns[c], n_int/n_uni, n_int/n_te))
    cat_feats_observed = (
        pd.DataFrame(cat_feats_observed, columns=['var', 'intersection/union', 'intersection/B'])
        .sort_values('intersection/B')
    )
    return cat_feats_observed
class HideMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if np.issubdtype(X[c].dtype, np.number):
                X.loc[:,c].fillna(X.loc[:,c].median(), inplace=True)
            else:
                X.loc[:,c].fillna('Unknown', inplace=True)
                X.loc[:,c] = X[c].map(lambda x: str(x))
        return X
    
    
class DropGarbage(BaseEstimator, TransformerMixin):
    def __init__(self, cols=['id', 'funder', 'recorded_by', 'date_recorded']):
        self.cols = cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(self.cols, axis=1, errors='ignore')

class SplitDate(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def t(x):
            if isinstance(x, str):
                x = dateutil.parser.parse(x)
            return (x.year, x.month, x.day, x.weekday())
        X = X.copy()
        (X.loc[:, self.col + '_year'],
         X.loc[:, self.col + '_month'],
         X.loc[:, self.col + '_day'],
         X.loc[:, self.col + '_weekday']) = zip(
        *X.loc[:, self.col].map(t))
        return X

    
class HandyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.loc[:,'amount_per_man'] = X.loc[:,'amount_tsh']/X.loc[:,'population']
        return X

class OutcomeFrequences(BaseException, TransformerMixin):
    def __init__(self, groupby, drop=False):
        self.groupby = groupby
        self.drop = drop
        self.cnts_ = None
        self.fq_cols_ = None
        self.unknowns_ = 0
        
    def fit(self, X, y=None):
        possible_outcomes = [_.split()[-1] if isinstance(_, str) else _ for _ in np.unique(y)]
        n_outs = len(possible_outcomes)
        self.cnts_ = collections.defaultdict(lambda: np.zeros(n_outs))
        # print(self.groupby)
        igroupby = np.array([X.columns.get_loc(_) for _ in self.groupby])
        for i in range(X.shape[0]):
            # assuming $y$ is label-encoded
            keys, out = tuple(X.iloc[i, igroupby].values), y[i]
            self.cnts_[keys][out] += 1 # no of `(keys, out)` occurences
        for k in self.cnts_:
            self.cnts_[k] //= self.cnts_[k].sum()
            
        self.possible_outcomes_ = possible_outcomes
        self.fq_cols_ = ['_'.join(self.groupby + [str(out), 'fq'],) for out in possible_outcomes]
        return self
    
    def transform(self, X):
        self.unknowns_ = 0
        igroupby = np.array([X.columns.get_loc(_) for _ in self.groupby])
        new_cols = [tuple([None for out in self.possible_outcomes_]) for _ in range(X.shape[0])]
        for i in range(X.shape[0]):
            keys = tuple(X.iloc[i, igroupby].values)
            if keys in self.cnts_:
                new_cols[i] = tuple([self.cnts_[keys][out] for out in self.possible_outcomes_])
            else:
                self.unknowns_ += 1
        self.unknowns_ /= X.size
        # print(str(X)[:100])
        # print(str(new_cols)[:100])
        # print(str(self.fq_cols_[:100]))
        new_cols = pd.DataFrame(new_cols, columns=self.fq_cols_)
        for c in self.fq_cols_:
            new_cols.loc[:,c].fillna(0, inplace=True) # new_cols.loc[:,c].median(), inplace=True)
        X = pd.concat((X, new_cols,), axis=1)
        if self.drop:
            X.drop(self.groupby, axis=1, inplace=True, errors='ignore')
        return X

class ExplicitAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.loc[:, 'age'] = X.loc[:,'date_recorded_year'] - X.loc[:,'construction_year']
        return X
def split_train_cv_test(X, y, proportions=(.75, .25/2, .25/2)):
    # expecting $y$ to be numpy array
    outs = np.unique(y)
    proportions = np.array(proportions)
    classes = [np.where(y == i)[0] for i in outs]
    xparts = [[] for _ in proportions]
    yparts = [[] for _ in proportions]
    for cidx in classes:
        cidx = sklearn.utils.shuffle(cidx)
        cprops = cidx.size * proportions
        cprops = cprops.astype(int)
        cprops[-1] = cidx.size - cprops[:-1].sum()
        cx = X.iloc[cidx,:]
        cy = y[cidx]
        for xpart, ypart, sz in zip(xparts, yparts, cprops):
            xpart.append(cx.iloc[:sz,:])
            ypart.append(cy[:sz])
            cx, cy = cx.iloc[sz:,:], cy[sz:]
    xparts = [pd.concat(xpart) for xpart in xparts]
    yparts = [np.concatenate(ypart) for ypart in yparts]
    # parts = []
    # print(len(xparts), 'parts')
    # for xpart, ypart in zip(xparts, yparts):
    #     parts.append(xpart)
    #     parts.append(ypart)
    parts = xparts + yparts
    return parts

def prepr_and_catboost(X_tr, X_cv, X_te,
                       y_tr, y_cv, y_te,
                       X_test,
                       prepr=None, outfname='ans.csv',
                       catboost_args = [],
                       catboost_kvargs = {
                           'random_seed': SEED,
                           'loss_function': 'MultiClass',
                           'verbose': True,
                           'calc_feature_importance': True,
                       }):
    if prepr is not None:
        X_tr = prepr.transform(X_tr)
        X_cv = prepr.transform(X_cv)
        X_te = prepr.transform(X_te)
        X_test = prepr.transform(X_test)
    print('preprocessing done')
    clf = catboost.CatBoostClassifier(*catboost_args, **catboost_kvargs)
    print('create cb instance: %s' % clf)
    clf.fit(X_tr, y_tr,
         cat_features=np.where(np.array([(X_tr[c].dtype == object)
                                  or (c.endswith('code')) 
                                         for c in X_tr.columns.values]))[0])
    print('fit!')
    display('train score: %s' % clf.score(X_tr, y_tr))
    display('cv score: %s' % clf.score(X_cv, y_cv))
    display('test score: %s' % clf.score(X_te, y_te))
    y_test = clf.predict(X_test)
    ans=pd.DataFrame({'status_group': y_enc.inverse_transform(
        y_test.astype(int).ravel())},
                         index= X_test.iloc[:,0].values)
    ans.index.name = 'id'
    ans.to_csv(name)
    print('saved!')
    return clf, X_tr, X_cv, X_te, y_tr, y_cv, y_te
