import numpy as np
from itertools import product
from sklearn.base import BaseEstimator
import pandas as pd
import copy

class SegmentedModel(BaseEstimator):

    def __init__(self, df, groups, model):
        self.df = df
        self.model = model
        self.groups = groups
        inds_list = self.get_group_inds_list(df, groups)
        self.inds = self.get_combined_inds(inds_list)

    def fit(self, X, y):
        sub_sets = self.get_sub_sets(self.inds, X, y)
        self.models = []
        
        for (X_train, y_train) in sub_sets:
            model = copy.deepcopy(self.model)
            model.fit(X_train,y_train)
            self.models.append(model)

    def score(self, X, y):
        sub_sets = self.get_sub_sets(self.inds, X, y)
        scores = []
        weights = []
        
        for (model,(X_test, y_test)) in zip(self.models, sub_sets):
            scores.append(model.score(X_test, y_test))
            weights.append(len(y_test))
            
        return np.average(scores,weights=weights)
        
    def get_group_ind(self, df, column, value):
        return df[column].isin(value)
    
    def get_group_inds(self, df, group):
        (column,values) = group
        return [self.get_group_ind(df, column, value) for value in values]
    
    def get_group_inds_list(self, df, groups):
        return [self.get_group_inds(df, group) for group in groups]
    
    def get_combined_inds(self, group_inds_list):
        return [reduce((lambda x, y: x & y),inds) for inds in product(*group_inds_list)]
        
    def get_sub_sets(self, inds, X, y):
        return [(X.loc[ind], y.loc[ind]) for ind in inds]        
    
    def predict_proba(self,X):       
        return pd.concat([pd.DataFrame(index = X[inds].index, 
                      data = model.predict_proba(X.loc[inds])[:,1])
        for (model,inds) in zip(self.models, self.inds)]).reindex(X.index)

             
     


