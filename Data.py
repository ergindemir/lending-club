import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Data(object):
    
    def __init__(self , dfloan=None , loan_file='data/loan.csv' , 
                 sp500_file='data/sp500.csv', exclude_columns = None):
        self.dict = {
        'float_columns' : ['int_rate', 'loan_amnt', 'funded_amnt', 'dti', 'total_rec_late_fee',
                           'funded_amnt_inv', 'collection_recovery_fee', 'recoveries',
                           'installment', 'revol_bal', 'last_pymnt_amnt', 'out_prncp',
                           'total_rec_prncp', 'out_prncp_inv', 'total_rec_int',
                           'total_pymnt_inv', 'total_pymnt', 'annual_inc', 'acc_now_delinq',
                           'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'open_acc', 'total_acc',
                           'collections_12_mths_ex_med', 'revol_util'],
                         
        'bad_status' : ['Charged Off','Default','Late (31-120 days)',
                        'In Grace Period', 'Late (16-30 days)',
                        'Does not meet the credit policy. Status:Charged Off'],
        
        'cat_columns'   : ['pymnt_plan', 'initial_list_status', 'application_type',
                           'verification_status', 'home_ownership', 'grade',
                           'emp_length', 'purpose', 'sub_grade'],
        
        'current_status' : ['Current','Issued'],
                            
        'similar_columns' : ['total_pymnt_inv','out_prncp_inv', 'out_prncp',
                             'funded_amnt_inv', 'installment', 'loan_amnt'],
                             
        'recovery_columns' : ['recoveries', 'collection_recovery_fee'],
                             
        'select_columns' : ['Volatility', 'termLength']                     
        }
    
        
        if dfloan is not None:
            self.dfloan=dfloan
        else:
            self.dfloan = pd.read_csv(loan_file,low_memory=False)
        self.sp500_file = sp500_file
        self.exclude_columns = exclude_columns    
        self.reset()

    def reset(self):            
        self.df = self.filter_rows()
        self.dffloat = self.get_float_columns()
        self.filter_columns()
        self.dfcat = self.get_cat_columns()
        self.transform_columns()
        self.add_volatility(self.sp500_file , 42) 
        
    def filter_rows(self):
        return self.dfloan[~self.dfloan.loan_status.isin(self.dict['current_status'])]
        
    def filter_columns(self):
        self.df.drop(self.dict['recovery_columns'],axis=1,inplace=True)
        self.df.drop(self.dict['similar_columns'],axis=1,inplace=True)
        
    def get_float_columns(self):
        dffloat = self.df[self.dict['float_columns']].drop(self.exclude_columns, axis=1, errors='ignore')
        dffloat.apply(lambda x: x.fillna(np.mean(x),inplace=True))
        return dffloat

    def get_cat_columns(self):
        return self.df[self.dict['cat_columns']].drop(self.exclude_columns, axis=1, errors='ignore')
    
    def transform_columns(self):
        self.df.issue_d = pd.to_datetime(self.df.issue_d,format='%b-%Y')
        self.df.last_credit_pull_d = pd.to_datetime(self.df.last_credit_pull_d,format='%b-%Y')
        self.df.last_pymnt_d = pd.to_datetime(self.df.last_pymnt_d,format='%b-%Y')
        self.df.last_pymnt_d[self.df.last_pymnt_d.isnull()] = self.df.issue_d[self.df.last_pymnt_d.isnull()]
        self.df['termLength'] = self.df.term.apply(lambda x: int(x.strip().split()[0]))
        self.df['Date'] = self.df.last_pymnt_d.values.astype('datetime64[M]') 
        self.df['default'] = self.df.loan_status.apply(lambda x: 1 if x in self.dict['bad_status'] else 0)

    def add_volatility(self, filename, ndays = 21):    
        df = pd.read_csv(filename)
        df.Date = pd.to_datetime(df.Date)
        df = df.sort_values(by='Date').reset_index(drop=True)
        ret = np.diff(df.Close)/df.Close[:-1]
        vol = [np.std(ret[n-ndays:n]) * 15.9  for n in range(ndays,len(ret)+1)]
        dfvol = df[['Date']]
        dfvol['Volatility'] = vol[0]
        dfvol.Volatility[ndays:] = vol
        dfvol.Date = dfvol.Date.values.astype('datetime64[M]') 
        dfvol = dfvol.groupby('Date').mean()
        self.df = self.df.join(dfvol, on='Date')
        self.dfvol = dfvol
        
    def encode_categorical(self, type):
        if type == 'label':
            d = defaultdict(LabelEncoder)
            dfenc = self.dfcat.apply(lambda x: d[x.name].fit_transform(x))
        else:
            dfenc = pd.concat([pd.get_dummies(self.dfcat[colname],prefix=colname) 
            for colname in self.dfcat.columns],axis=1)
        
        return dfenc

    def merge_dataset(self, type='label'):
        dfenc = self.encode_categorical(type)
        dfselect = self.df[self.dict['select_columns']]
        self.dfmerge =  pd.concat([self.dffloat,dfenc,dfselect],axis=1)
        
    def get_train_test_set(self, test_size=0.25):
        X = StandardScaler().fit_transform(self.dfmerge.values)
        y = self.df.default.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test
        
        
        
        
        
            
    
    
        





