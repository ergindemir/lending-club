import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/loan.csv',low_memory=False)

'''select continuos variables prefiltered by eda, fill missing variables
'''

float_columns = ['int_rate', 'loan_amnt', 'funded_amnt', 'dti', 'total_rec_late_fee',
       'funded_amnt_inv', 'collection_recovery_fee', 'recoveries',
       'installment', 'revol_bal', 'last_pymnt_amnt', 'out_prncp',
       'total_rec_prncp', 'out_prncp_inv', 'total_rec_int',
       'total_pymnt_inv', 'total_pymnt', 'annual_inc', 'acc_now_delinq',
       'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'open_acc', 'total_acc',
       'collections_12_mths_ex_med', 'revol_util']


df_float = df[float_columns]
df_float.apply(lambda x: x.fillna(np.mean(x),inplace=True))


'''transform variables
'''

bad_status = ['Charged Off',
 'Default',
 'Late (31-120 days)',
 'In Grace Period',
 'Late (16-30 days)',
 'Does not meet the credit policy. Status:Charged Off']

df['default'] = df.loan_status.apply(lambda x: 1 if x in bad_status else 0)

'''select categorical variables and encode
'''

cat_columns = ['pymnt_plan', 'initial_list_status', 'application_type',
       'verification_status', 'home_ownership', 'grade',
       'emp_length', 'purpose', 'sub_grade']

df_cat = df[cat_columns]
d = defaultdict(LabelEncoder)
df_cat_enc = df_cat.apply(lambda x: d[x.name].fit_transform(x))

'''
merge dataframes for input variables
'''

dfx = pd.concat([df_float,df_cat_enc],axis=1)

'''
engineered features
'''


dfx['termLength'] = df.term.apply(lambda x: int(x.strip().split()[0]))
dfx.annual_inc[dfx.annual_inc==0]=np.mean(dfx.annual_inc)
dfx['loanIncomeRatio'] = dfx.funded_amnt/dfx.annual_inc
dfx['loanIncomeRatioYear'] = dfx.loanIncomeRatio * 12.0 /dfx.termLength

'''
run first test model Random Forest
'''

X = dfx.values
y = df.default.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = RandomForestClassifier(n_estimators=50, n_jobs = 8)
model.fit(X_train,y_train)
model.score(X_test,y_test)
dfx.columns[np.argsort(model.feature_importances_)[::-1]]

'''
We have leakage, model scores 97.5% in our first trial.
let's remove columns with recovery info
'''

leakge_columns = filter(lambda x: 'rec' in x, dfx.columns)
dfx.drop(leakge_columns,inplace=True,axis=1)
X = dfx.values
y = df.default.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = RandomForestClassifier(n_estimators=50, n_jobs = 8)
model.fit(X_train,y_train)
model.score(X_test,y_test)
dfx.columns[np.argsort(model.feature_importances_)[::-1]]

'''
score = 0.9736347449796029
feature importances:
[u'total_pymnt', u'last_pymnt_amnt', u'total_pymnt_inv',
       u'out_prncp_inv', u'out_prncp', u'funded_amnt', u'funded_amnt_inv',
       u'installment', u'loan_amnt', u'int_rate', u'loanIncomeRatio',
       u'loanIncomeRatioYear', u'revol_bal', u'dti', u'revol_util',
       u'annual_inc', u'total_acc', u'sub_grade', u'open_acc', u'emp_length',
       u'grade', u'termLength', u'purpose', u'inq_last_6mths',
       u'verification_status', u'delinq_2yrs', u'initial_list_status',
       u'home_ownership', u'collections_12_mths_ex_med', u'acc_now_delinq',
       u'pymnt_plan', u'application_type']

lets drop similar columns:
'''

dfx.corr()

similar_columns = ['total_pymnt_inv','out_prncp_inv', u'out_prncp',
                   u'funded_amnt_inv', u'installment', u'loan_amnt',
                   'loanIncomeRatioYear','grade']

dfx.drop(similar_columns,inplace=True, axis=1)
X = dfx.values
y = df.default.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = RandomForestClassifier(n_estimators=50, n_jobs = 8)
model.fit(X_train,y_train)
model.score(X_test,y_test)
dfx.columns[np.argsort(model.feature_importances_)[::-1]]

'''
everything cleaned up
still got a test score of 93% without optimizing anything
'''





