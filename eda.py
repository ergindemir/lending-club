import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df = pd.read_csv('data/loan.csv',low_memory=False)
#df.to_pickle('data/loan.pkl')

df=pd.read_pickle('data/loan.pkl')

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
