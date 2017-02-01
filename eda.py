import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


#df = pd.read_csv('data/loan.csv',low_memory=False)
#df.to_pickle('data/loan.pkl')

df=pd.read_pickle('data/loan.pkl')

dfinfo=[]
for col in df.columns:
    column=df[col]
    dfinfo.append([col,column.dtype,np.sum(column.isnull()),
                   len(column.unique()),np.min(column),np.max(column)])
dfinfo = pd.DataFrame(dfinfo)    
dfinfo.columns = ['Name','Type','Nulls','Unique','Min','Max']  
    

df['issueDate'] = pd.to_datetime(df.issue_d,format='%b-%Y')
df['lastPaymentDate'] = pd.to_datetime(df.last_pymnt_d,format='%b-%Y')
df['lastCreditPullDate'] = pd.to_datetime(df.last_credit_pull_d,format='%b-%Y')

bad_status = ['Charged Off',
 'Default',
 'Late (31-120 days)',
 'In Grace Period',
 'Late (16-30 days)',
 'Does not meet the credit policy. Status:Charged Off']

df['default'] = df.loan_status.apply(lambda x: 1 if x in bad_status else 0)
df['termLength'] = df.term.apply(lambda x: int(x.strip().split()[0]))
df['Date'] = df.lastPaymentDate.values.astype('datetime64[M]') 
#with open(r"data/df.pkl", "wb") as output_file:
#     cPickle.dump(df, output_file)
#
#now=datetime.datetime.now()
#with open(r"data/loan.pkl", "rb") as output_file: 
#    df3 = cPickle.load(output_file)
#print datetime.datetime.now()-now
#
#now=datetime.datetime.now()
#df3=pd.read_pickle('data/loan.pkl')
#print datetime.datetime.now()-now

emp_length_values = df.groupby(df.emp_length).default.mean().sort_values().index                           
emp_length_dict=dict(zip(emp_length_values,range(len(emp_length_values))))
df['empLength'] = df.emp_length.apply(lambda x: emp_length_dict[x])

home_ownership_values = df.groupby(df.home_ownership).default.mean().sort_values().index
home_ownership_dict = dict(zip(home_ownership_values,range(len(home_ownership_values))))
df['homeOwnership'] = df.home_ownership.apply(lambda x: home_ownership_dict[x])

df['verificationStatus'] = df.verification_status.apply(lambda x:1 if x in ['Verified', 'Source Verified'] else 0)

#now=datetime.datetime.now()
#df.to_pickle('data/df.pkl')
#print datetime.datetime.now()-now

def gini(X, y):
    '''
    INPUT:
        - y: 1d numpy array
    OUTPUT:
        - float
    Return the entropy of the array y.
    '''
    total = 0
    for x in np.unique(X):
        ind = x==X
        p = np.mean(y[ind])
        g = p**2+(1-p)**2
        total += sum(ind) * g
    return 1-total/len(y)

def iv(X, y):
    total = 0
    N0 = np.sum(y==0)*1.0
    N1 = np.sum(y==1)*1.0
    if len(np.unique(X))>100: return 0
    for x in np.unique(X):
        ind = x==X
        n0 = np.sum(y[ind]==0)/N0
        n1 = np.sum(y[ind]==1)/N1
        if n1*n0 != 0:
            total += (n1-n0)*(np.log(n1/n0))
    return total

def iv_cont(X, y, mask):
    total = 0
    N0 = np.sum(y==0)*1.0
    N1 = np.sum(y==1)*1.0
    for ind in mask:
        n0 = np.sum(y[ind]==0)/N0
        n1 = np.sum(y[ind]==1)/N1
        if n1*n0 != 0:
            total += (n1-n0)*(np.log(n1/n0))
    return total

def discrete(x, N=10):
    mask=[]
    for i in range(N):
        point1 = np.percentile(x, i * 100.0/N)
        point2 = np.percentile(x, (i + 1) * 100.0/N)
        mask.append((x >= point1) & (x <= point2))

    return mask

for col in df_float.columns:
    x = df_float[col].values
    print col, iv_cont(x,y,discrete(x))

for col in df_str.columns:
    print col, iv(df_str[col].values,y)


