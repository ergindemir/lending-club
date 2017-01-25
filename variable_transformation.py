import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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

#X = dfx.values
#y = df.default.values
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
#model = RandomForestClassifier(n_estimators=50, n_jobs = 8)
#model.fit(X_train,y_train)
#model.score(X_test,y_test)
#dfx.columns[np.argsort(model.feature_importances_)[::-1]]

'''
We have leakage, model scores 97.5% in our first trial.
let's remove columns with recovery info
'''

leakge_columns = filter(lambda x: 'rec' in x, dfx.columns)
dfx.drop(leakge_columns,inplace=True,axis=1)


#X = dfx.values
#y = df.default.values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#model = RandomForestClassifier(n_estimators=50, n_jobs = 8)
#model.fit(X_train,y_train)
#model.score(X_test,y_test)
#dfx.columns[np.argsort(model.feature_importances_)[::-1]]

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
model = RandomForestClassifier(n_estimators=50, n_jobs = -1)
model.fit(X_train,y_train)
model.score(X_test,y_test)
dfx.columns[np.argsort(model.feature_importances_)[::-1]]

'''
everything cleaned up
still got a test score of 93% without optimizing anything
'''

dfdummy = pd.concat([pd.get_dummies(df_cat[colname],
    prefix=colname) for colname in df_cat.columns],axis=1)

leakge_columns = filter(lambda x: 'rec' in x, df_lr.columns)

df_lr = pd.concat([df_float,dfdummy],axis=1)
df_lr.drop(leakge_columns,inplace=True,axis=1)
df_lr.drop(np.intersect1d(similar_columns,df_lr.columns),inplace=True,axis=1)

X = df_lr.values
X = StandardScaler().fit_transform(X)
y = df.default.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LogisticRegression(n_jobs=-1)
model.fit(X_train,y_train)
model.score(X_test,y_test)

ind = np.argsort(np.abs(model.coef_))[0]

[(df_lr.columns[i],model.coef_[0][i])for i in ind[::-1]]
'''
feature importance from LR:
[('int_rate', 3.5814977219697099),
 ('last_pymnt_amnt', -3.4298523152035707),
 ('grade_A', 0.84117926187450665),
 ('grade_E', -0.66941324458055484),
 ('grade_F', -0.60681370063382323),
 ('total_pymnt', -0.58654439702191474),
 ('grade_D', -0.52244564183429165),
 ('grade_B', 0.49833640825792536),
 ('funded_amnt', 0.47120870674462445),
 ('sub_grade_B1', 0.45181693614758267),
 ('sub_grade_A1', 0.39307571614383718),
 ('sub_grade_A2', 0.37886387393668175),
 ('sub_grade_A4', 0.37596689706611874),
 ('sub_grade_E5', -0.36714105891239579),
 ('grade_G', -0.36426336589458613),
 ('sub_grade_E4', -0.3346203452900825),
 ('sub_grade_D5', -0.33203308792882996),
 ('sub_grade_B2', 0.33172121601599885),
 ('sub_grade_A3', 0.32964917413671213),
 ('sub_grade_A5', 0.30265369476633353),
 ('sub_grade_E3', -0.29911698814257676),
 ('sub_grade_D4', -0.29153437260811788),
 ('sub_grade_F3', -0.27942718106018744),
 ('sub_grade_F1', -0.27564507580691178),
 ('sub_grade_F2', -0.2754256653869448),
 ('sub_grade_F4', -0.27321258355252209),
 ('sub_grade_E2', -0.2627289441258796),
 ('annual_inc', -0.25522757367699495),
 ('sub_grade_F5', -0.25464488596379947),
 ('sub_grade_D3', -0.2350403719485841),
 ('sub_grade_C5', -0.23246847252476802),
 ('sub_grade_E1', -0.22302431777647844),
 ('inq_last_6mths', 0.21914150941608784),
 ('sub_grade_B3', 0.2021180884384895),
 ('sub_grade_G1', -0.19409320854396528),
 ('sub_grade_G2', -0.18862519330795696),
 ('sub_grade_D2', -0.17525053186048034),
 ('sub_grade_C1', 0.16231657530447813),
 ('sub_grade_G3', -0.15987544572303261),
 ('initial_list_status_w', -0.15269539088029987),
 ('initial_list_status_f', 0.15269539088026521),
 ('sub_grade_C4', -0.14824156191818258),
 ('sub_grade_G4', -0.13650590806971666),
 ('dti', -0.13435493605311538),
 ('sub_grade_G5', -0.12303488523465124),
 ('sub_grade_D1', -0.095438296624895461),
 ('grade_C', -0.095189127446325669),
 ('revol_util', 0.090596928754678493),
 ('purpose_small_business', 0.079802048918564725),
 ('total_acc', 0.07175584182653362),
 ('open_acc', -0.064312554914798939),
 ('purpose_credit_card', -0.0639831699561524),
 ('sub_grade_B4', 0.056062112256364163),
 ('sub_grade_C2', 0.054843519796289195),
 ('home_ownership_RENT', 0.053940630647455967),
 ('sub_grade_B5', -0.053443985989292363),
 ('sub_grade_C3', -0.049232290926077733),
 ('emp_length_10+ years', -0.04515275031054268),
 ('home_ownership_MORTGAGE', -0.04114529530562356),
 ('verification_status_Source Verified', -0.037159704300606118),
 ('application_type_JOINT', -0.03462202701566601),
 ('application_type_INDIVIDUAL', 0.034622027015626167),
 ('purpose_educational', 0.032978340398744282),
 ('verification_status_Verified', 0.031051576466457134),
 ('collections_12_mths_ex_med', -0.0308736332658965),
 ('emp_length_6 years', 0.027622585137041654),
 ('emp_length_< 1 year', 0.022805285478647564),
 ('purpose_debt_consolidation', 0.02280242314570477),
 ('home_ownership_OWN', -0.020686935080658865),
 ('home_ownership_OTHER', 0.020081664138000176),
 ('purpose_wedding', 0.019106231220019809),
 ('emp_length_7 years', 0.017150388919615484),
 ('purpose_house', 0.012802541729828577),
 ('home_ownership_ANY', -0.012289765411230377),
 ('purpose_other', 0.011267927862611606),
 ('purpose_medical', 0.0099681922560761575),
 ('purpose_renewable_energy', 0.0098134772838511831),
 ('acc_now_delinq', -0.0096003269084417864),
 ('emp_length_5 years', 0.0093150120302942663),
 ('revol_bal', -0.0090179724298383099),
 ('emp_length_n/a', -0.0085015938292129604),
 ('emp_length_4 years', 0.007883276625076506),
 ('verification_status_Not Verified', 0.0073633490661713557),
 ('emp_length_1 year', 0.0072516502179542043),
 ('purpose_moving', 0.0067907615991156271),
 ('pymnt_plan_n', -0.0065375304036341822),
 ('pymnt_plan_y', 0.0065375304036316504),
 ('emp_length_8 years', -0.0064124773038829706),
 ('emp_length_3 years', 0.0056706525324684183),
 ('purpose_vacation', -0.0051409069271894725),
 ('home_ownership_NONE', 0.0045810498613152381),
 ('purpose_home_improvement', 0.0036644924933391951),
 ('emp_length_2 years', 0.0025890579091142197),
 ('emp_length_9 years', 0.0023233347094614911),
 ('delinq_2yrs', 0.0006818137849406694),
 ('purpose_major_purchase', 0.00026621834707112009),
 ('purpose_car', 0.00014269045701486175)]
'''

'''
lets drop grade and sub_grade one at a time and compare performance
'''

subgrade_columns = filter(lambda x: 'sub_grade_' in x, df_lr.columns)
df_lr_subgrade = df_lr.drop(subgrade_columns,axis=1)


X = df_lr_subgrade.values
X = StandardScaler().fit_transform(X)
y = df.default.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LogisticRegression(n_jobs=-1)
model.fit(X_train,y_train)
model.score(X_test,y_test)

ind = np.argsort(np.abs(model.coef_))[0]

[(df_lr_subgrade.columns[i],model.coef_[0][i])for i in ind[::-1]]
   
'''
[('last_pymnt_amnt', -3.1932502123950868),
 ('int_rate', 1.7074944976930293),
 ('grade_A', 0.58353904961713987),
 ('grade_E', -0.50485202993901301),
 ('grade_F', -0.47559161301660124),
 ('grade_B', 0.36110992016046684),
 ('grade_D', -0.35384192592519487),
 ('funded_amnt', 0.29026702752874173),
 ('grade_G', -0.28010994417012419),
 ('total_pymnt', -0.26069176258123977),
 ('annual_inc', -0.21275793734192935),
 ('inq_last_6mths', 0.19508015213145061),
 ('initial_list_status_f', 0.17081253871734156),
 ('initial_list_status_w', -0.17081253871733434),
 ('dti', -0.15725721485840105),
 ('revol_util', 0.086038107463410979),
 ('total_acc', 0.080264235153855473),
 ('purpose_small_business', 0.069078677180304818),
 ('open_acc', -0.069017875780450849),
 ('verification_status_Source Verified', -0.054654022643418669),
 ('home_ownership_RENT', 0.048249534098427763),
 ('purpose_credit_card', -0.045674635277344333),
 ('emp_length_10+ years', -0.045590909060186445),
 ('collections_12_mths_ex_med', -0.043316844216944046),
 ('grade_C', -0.039882244723710922),
 ('verification_status_Verified', 0.036441984475543442),
 ('emp_length_6 years', 0.035623862888116531),
 ('home_ownership_MORTGAGE', -0.035588675113381894),
 ('application_type_JOINT', -0.031636668935529959),
 ('application_type_INDIVIDUAL', 0.031636668935502543),
 ('purpose_educational', 0.028842045794028923),
 ('purpose_wedding', 0.024872882983461405),
 ('home_ownership_OWN', -0.020504038811385898),
 ('verification_status_Not Verified', 0.020278953337557051),
 ('emp_length_7 years', 0.020229148087233535),
 ('home_ownership_OTHER', 0.019352068644399988),
 ('purpose_debt_consolidation', 0.019090960662440558),
 ('emp_length_5 years', 0.018684603746916512),
 ('emp_length_< 1 year', 0.017874528131535044),
 ('home_ownership_ANY', -0.012983243631218387),
 ('purpose_house', 0.012624066153301999),
 ('emp_length_n/a', -0.012279553793013265),
 ('purpose_vacation', -0.011796442312498722),
 ('delinq_2yrs', -0.0096725800280010581),
 ('purpose_renewable_energy', 0.0093068085802711312),
 ('acc_now_delinq', -0.0071899954340728248),
 ('emp_length_8 years', -0.0069590076696161435),
 ('emp_length_1 year', 0.0066432211727723636),
 ('purpose_medical', 0.006628693738763943),
 ('revol_bal', -0.0058338257846197459),
 ('emp_length_9 years', 0.0042849368847328066),
 ('emp_length_4 years', 0.0041310033201715545),
 ('pymnt_plan_y', 0.0032874075449436397),
 ('pymnt_plan_n', -0.0032874075449368123),
 ('emp_length_3 years', 0.0021701360156854903),
 ('purpose_home_improvement', -0.0016161256413481054),
 ('purpose_moving', 0.00096181273226899288),
 ('emp_length_2 years', 0.00055872256124722827),
 ('home_ownership_NONE', 0.00037994451555043893),
 ('purpose_major_purchase', 0.00036494653784666597),
 ('purpose_other', -4.9859325191414796e-05),
 ('purpose_car', -4.8452336820822633e-05)]
 '''
 
