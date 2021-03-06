## Business Overview
With cyclical financial instabilities in credit risk markets, the area of credit risk modelling has become
ever more important, leading to the need for more accurate and robust models. With the introduction
of the Basel II capital accord in 2004, financial institutions now prefer developing their own models
without relying on regulators fixed estimates.
Using risk models financial institutions derive 3 key risk parameters:
* PD: Probability of Default, the likelihood that a loan won’t be paid on time and fall into default.
* LGD: Loss Given Default, the estimated percentage loss of the exposure in case of default.
* EAD: Exposure at Default, the monetary exposure in case of default.

Financial institutions use these parameters to:

1. Decide whether to approve an applicant’s loan request based on the PD
2. Calculate the Expected Loss, which can be defined as the expected mean loss within 12 months:

  EL = PD X LGD X EAD

Within the scope of this project, the focus will be the estimation of the PD

## EDA I
https://github.com/ergindemir/lending-club/blob/master/EDA.ipynb

In this initial stage of EDA we basically determine:
* Columns with too many missing values
* Columns with totally unique values
* Categorical variables
* Continuous variables
* The count of missing values and how to handle them

## EDA II
https://github.com/ergindemir/lending-club/blob/master/EDA%20float.ipynb

This is the second EDA where we examine the continuous variables and develop strategies to impute missing values

## Data Class
https://github.com/ergindemir/lending-club/blob/master/Data.py

All data operations are performed by the Data class.
The Data object loads all the datasets in memory, processes them and provides clean data in desired format.
The basic operations of Data class are:
* Filter rows, drop rows with ongoing credit status, only include rows with default or paid off status.
* Filter columns identified as useless by the EDA
* Select continuous variables and impute them if necessary
* Select categorical variables and encode them either with LabelEncoder or dummifier, depending on model type.
* Transform variables
* Generate volatility curve, calculate monthly averages and match it to the dataset.
* Exclude features manually
* Create Train/Test data sets.

## SegmentedModel Class
https://github.com/ergindemir/lending-club/blob/master/SegmentedModel.py

To increase the model performance of the logistic regression model, 
the dataset is split into subsets using grouped column values.
Then, a separate model is built for each subset and corresponding score is evaluated.
The overall score is calculated by combining each score by it's weight (subset size).
                                                                       
For example the data set can be split into separate subsets for each credit grade.
The assumption is that customers in each grade group will have different behavior and a separate model for each group will perform better. 

The downside of this approach is that there is less data rows for each subset. Therefore we construct subsets with comparable row counts by grouping.

For this data set we presume that segmentation based on the features ‘grade’, ‘termLength’ and ‘purpose’ might give better performance relative to the baseline model without any segmentation. Furthermore, the SegmentedModel class supports segmentation based on multiple columns. Therefore the model will be tested not only with these 3 groups but also with all possible combinations of them.

## Final Analysis
https://github.com/ergindemir/lending-club/blob/master/Segmentation.ipynb

The baseline performances are scored for logistic regression and random forest models.
Then segmented models have been scored for 3 main features and their combinations, with 7 models in total.
The segmented model performances are plotted in decreasing scores with the performances of the baseline models.

## Results

* All segmentation options gave better accuracy than the baseline logistic regression model. 
* For segmentation strategy 'grade' is a significant column. If one has to use logistic regression model, one should use segmentation based on the 'grade' column to achieve a significant performance gain.
* Columns 'termLength' and 'purpose' provide minor gains when segmentation is applied solely on these columns.
* Segmentation based on 3 columns at the same time, surprisingly, gave the best performance gain. The gain compared to 'grade' column, however, is miniscule, and can be ignored if somebody opts for a simpler model.

![Alt](https://github.com/ergindemir/lending-club/blob/master/segmented_model_performance.png "Model Performance")

![Alt](https://github.com/ergindemir/lending-club/blob/master/roc_curves.png "ROC Curves")

