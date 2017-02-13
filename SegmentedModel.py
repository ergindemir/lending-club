import numpy as np
from itertools import product

class SegmentedModel(object):

    def __init__(self, df, groups, model):
        self.df = df
        self.model = model
        self.groups = groups

    def fit(self, (X_train, X_test, y_train, y_test)):
        inds_list = self.get_group_inds_list(self.df, self.groups)
        inds = self.get_combined_inds(inds_list)
        train_test_sets = self.get_train_test_sets(inds , (X_train, X_test, y_train, y_test))
        trainscores = []
        testscores = []
        trainweights = []
        testweights = []
        model = self.model
        
        for (X_train, X_test, y_train, y_test) in train_test_sets:
            model.fit(X_train,y_train)
            trainscore = model.score(X_train,y_train)
            testscore = model.score(X_test,y_test)
            trainscores.append(trainscore)
            testscores.append(testscore)
            trainweights.append(len(y_train))
            testweights.append(len(y_test))
            
        self.trainscore = np.average(trainscores,weights=trainweights)
        self.testscore = np.average(testscores,weights=testweights)
        
    def get_group_ind(self, df, column, value):
        return df[column].isin(value)
    
    def get_group_inds(self, df, group):
        (column,values) = group
        return [self.get_group_ind(df, column, value) for value in values]
    
    def get_group_inds_list(self, df, groups):
        return [self.get_group_inds(df, group) for group in groups]
    
    def get_combined_inds(self, group_inds_list):
        if len(group_inds_list) == 1:
            return group_inds_list[0]
        else:
            return [(inds[0]&inds[1]) for inds in product(*group_inds_list)]
        
    def get_train_test_sets(self, inds, (X_train, X_test, y_train, y_test)):
        return [(X_train.loc[ind], X_test.loc[ind], y_train.loc[ind], y_test.loc[ind]) for ind in inds]        
    

