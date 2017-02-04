# Lending Club Data Analysis
In this project the main focus is to built a model that predicts the probability of default accurately.
The stages of the modelling process can be found at:
## 1. EDA I
https://github.com/ergindemir/lending-club/blob/master/EDA.ipynb

In this initial stage of EDA we basically determine:
* Columns with too many missing values
* Columns with totally unique values
* Categorical variables
* Continous variables
* The count of missing values and how to handle them

## 2.EDA II
https://github.com/ergindemir/lending-club/blob/master/EDA%20float.ipynb

This is the second EDA where we examine the continous variables and develop strategies to impute missing values

## 3. Data Class
https://github.com/ergindemir/lending-club/blob/master/Data.py

All data operations are performed by the Data class.
The Data objects loads all the datasets in memory, processes them and provides cleane data in desired format.
The basic functions of Data class are:
* Filter rows, drop rows with ongoing credit status, only include rows with default or paid off status.
* Filter columns identified as useless by the EDA
* Select continous variables and impute them if necessary
* Select categorical variables and ecode them either with LabelEncoder or dummifier, depending on model type.
* Transform variables
* Generate volatility curve, calculate monthly averages and match it to the dataset.
* Exclude features manually
* Create Train/Test data sets.

## 4. Final Analysis
In this part we briefly demonstrate the importance of the market volatility concept in predicting default probability.

## 5. Results
It comes out volatility is a key component determining the default rate of a credit portfolio.
As a single predictor variable, it imporves overall model performance by at least 1% for a non volatile market.
For volatile periods, the model performance improves more than 30%.

![Alt](https://github.com/ergindemir/lending-club/blob/master/model%20performance.png "Model Performance")
