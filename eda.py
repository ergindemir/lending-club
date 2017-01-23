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
df['termLength'] = df.term.apply(lambda x: int(x.strip().split()[0]))

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







