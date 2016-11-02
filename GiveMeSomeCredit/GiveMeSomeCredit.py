# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:06:18 2016
. 
@author: Vinod
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cstrain=pd.read_csv("C:/Users/Vinod/Documents/Python Scripts/GiveMeSomeCredit/cs-training.csv")
cstrain = cstrain.drop('Unnamed: 0', 1)
cstrain.head()

feat_cols= ['age','NumberOfTime30-59DaysPastDueNotWorse','RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']

train=cstrain.loc[:,feat_cols]
train.shape
y=cstrain.loc[:,['SeriousDlqin2yrs']].SeriousDlqin2yrs
y.shape


y.value_counts()/len(y)
plt.hist(y)
#==============================================================================
# unbalanced dataset with 93% as 0 and 6% as 1.
# A random coin toss has 50% accuracy. Without any prediction, the data can produce an accuracy of 93%.
# Hence Accuracy is not a very good metric to refer to.
# We need to look at the ROC and AUC
# 
#==============================================================================


# NULL value assessment
def NullTreatMent_Median(data):
    data['MonthlyIncome'].fillna(data['MonthlyIncome'].median(), inplace=True)
    data['NumberOfDependents'].fillna(0.0,inplace=True)
    return data

## only two columns (Monthly income and #dependents have null elements
train.apply(lambda x: sum(x.isnull()),axis=0)  ## for entire dataset
train.NumberOfDependents.value_counts()
sum(train.NumberOfDependents.isnull())/len(train.NumberOfDependents)

train.MonthlyIncome.describe()
sum(train.MonthlyIncome.isnull())/len(train.MonthlyIncome)
# 20% of the Monthly income is null







## Data Transformation - Outlier 
data=train
data['SeriousDlqin2yrs']=y
data.head()


data.RevolvingUtilizationOfUnsecuredLines.describe()
plt.hist(data.RevolvingUtilizationOfUnsecuredLines) ## heavily skewed factor
np.percentile(data.RevolvingUtilizationOfUnsecuredLines,99.5)
#99.5 % percentage of data points, it is better to cap to avoid any data skewness

def Treat_RevolvingUtilizationOfUnsecuredLines(data):
    New = []
    for x in data.RevolvingUtilizationOfUnsecuredLines:
        if x > 1.5 :
            New.append(1.5)
        else:
            New.append(x)
    data.RevolvingUtilizationOfUnsecuredLines=New
    return data

data=Treat_RevolvingUtilizationOfUnsecuredLines(data)

## Feature2: age
data.age.hist()
len(data.age[data.age.values < 20]) #one entry in <20
len(data.age[data.age.values > 95])## <0.1% data at 95+ years age
np.percentile(data.age,99.9)

def Treat_age(data):
    New=[]
    for x in data.age:
        if x < 20:
            New.append(20)
        elif x > 95:
            New.append(95)
        else:
            New.append(x)
    data.age=New
    return data

data=Treat_age(data)

## FEature 3: NumberOfTime30-59DaysPastDueNotWorse
data['NumberOfTime30-59DaysPastDueNotWorse'].hist()
data['NumberOfTime30-59DaysPastDueNotWorse'].describe()
data['NumberOfTime30-59DaysPastDueNotWorse'].value_counts() ## few values > 8
pd.crosstab(cstrain['NumberOfTime30-59DaysPastDueNotWorse'], cstrain.SeriousDlqin2yrs) 
### no real pattern with the target variable, therefore we can correct outlier values alone

def Treat_NumberOfTime3059DaysPastDueNotWorse(data):
    New=[]
    for x in data['NumberOfTime30-59DaysPastDueNotWorse']:
        if x in (98,96):
            New.append(0)
        else:
            New.append(x)
    data['NumberOfTime30-59DaysPastDueNotWorse']=New
    return data

data=Treat_NumberOfTime3059DaysPastDueNotWorse(data)

## feature 4 : debtratio
data.DebtRatio.describe()
len(data.DebtRatio[data.DebtRatio.values > 1])/len(data) ## 23% data have ratio > 1
data.DebtRatio.plot.box()
#for x in data.DebtRatio:
#    if x > 1.0:
#        data['DebtRatioIndex']= ">1"
#    else:
#        data['DebtRatioIndex']="<1"
#len(data.DebtRatio[data.DebtRatio.values > 1])
np.percentile(data.DebtRatio,5), np.percentile(data.DebtRatio,50),np.percentile(data.DebtRatio,80),np.percentile(data.DebtRatio,90)
data.DebtRatio[data.DebtRatio.values > 4]=4 ## All beyond entries are grouped together
data.DebtRatio.hist()

## feature 5: MonthlyIncome
data['MonthlyIncome'].plot.box()
data=NullTreatMent_Median(data)

## feature 6 -NumberOfOpenCreditLinesAndLoans - This could offer some imp info about problem. Not transforming it
data.NumberOfOpenCreditLinesAndLoans.hist()
data.NumberOfOpenCreditLinesAndLoans.plot.box()

## feature 7 - NumberOfTimes90DaysLate
data.NumberOfTimes90DaysLate.hist()
data.NumberOfTimes90DaysLate.value_counts()
## Same problem as 'NumberOfTime30-59DaysPastDueNotWorse'
def Treat_NumberOfTimes90DaysLate(data):
    New=[]
    for x in data['NumberOfTimes90DaysLate']:
        if x in (98,96):
            New.append(0)
        else:
            New.append(x)
    data['NumberOfTimes90DaysLate']=New
    return data
data=Treat_NumberOfTimes90DaysLate(data)

## feat 8 -NumberRealEstateLoansOrLines
data.NumberRealEstateLoansOrLines.hist()
data.NumberRealEstateLoansOrLines.describe()
data.NumberRealEstateLoansOrLines.value_counts()

def Treat_NumberRealEstateLoansOrLines(data):
    New = []
    for x in data.NumberRealEstateLoansOrLines:
        if x >= 10 :
            New.append(10)
        else:
            New.append(x)
    data.NumberRealEstateLoansOrLines=New
    return data
data=Treat_NumberRealEstateLoansOrLines(data)

## feat 9 - NumberOfTime60-89DaysPastDueNotWorse
data['NumberOfTime60-89DaysPastDueNotWorse'].hist()
data['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()
## Same problem as 'NumberOfTime30-59DaysPastDueNotWorse'
def Treat_NumberOfTime6089DaysPastDueNotWorse(data):
    New=[]
    for x in data['NumberOfTime60-89DaysPastDueNotWorse']:
        if x in (98,96):
            New.append(0)
        else:
            New.append(x)
    data['NumberOfTime60-89DaysPastDueNotWorse']=New
    return data
data=Treat_NumberOfTime6089DaysPastDueNotWorse(data)

## feat 10 - NumberOfDependents
data.NumberOfDependents.hist()
data.NumberOfDependents.value_counts()
    ## Since long tail dist. we can group last few entries
    def Treat_NumberOfDependents(data):
        New=[]
        for x in data['NumberOfDependents']:
            if x > 10.0:
                New.append(10.0)
            else:
                New.append(x)
        data['NumberOfDependents']=New
        return data
    data=Treat_NumberOfDependents(data)



## TWO WAY ANALYSIS
data.boxplot(column="RevolvingUtilizationOfUnsecuredLines",by="SeriousDlqin2yrs")
data.boxplot(column="age",by="SeriousDlqin2yrs")
data.boxplot(column="NumberOfTime30-59DaysPastDueNotWorse",by="SeriousDlqin2yrs")
#data.boxplot(column="DebtRatio",by="SeriousDlqin2yrs") ## not useful
data.boxplot(column="MonthlyIncome",by="SeriousDlqin2yrs")
#data.boxplot(column="NumberOfTimes90DaysLate",by="SeriousDlqin2yrs")
data.boxplot(column="NumberOfDependents",by="SeriousDlqin2yrs") ## higher 

## AS #dependents increases, monthly income decrases, leading to credit crises
data.pivot_table(index='SeriousDlqin2yrs', values="MonthlyIncome", columns="NumberOfDependents",aggfunc=np.median)




## NEW FEATURE CREATION
data['TotalDebt']=data.DebtRatio * data.MonthlyIncome
if data.TotalDebt <=1:
    data['LogTotalDebt']= 0.0
else:
    data['LogTotalDebt']= np.log(data.TotalDebt)

#temp1 = data.loc[1:,['MonthlyIncome','NumberOfDependents']]
for i in range(0,1):
    if data.loc[i,'NumberOfDependents']> 0:
        data.loc[i,'IncomePerDependent']=data.loc[i,'MonthlyIncome']/data.loc[i,'NumberOfDependents']
    else:
        data.loc[i,'IncomePerDependent']=data.loc[i,'MonthlyIncome']

data.head()
data.shape

####################### MODELING - Predictions ###############
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score


xtrain=data.loc[1:125000,feat_cols]
xtrain.shape
ytrain=data.loc[1:125000,['SeriousDlqin2yrs']].SeriousDlqin2yrs
ytrain.shape

xtest=data.loc[125000:150000,feat_cols]
ytest=data.loc[125000:150000,['SeriousDlqin2yrs']].SeriousDlqin2yrs


lr_unbalanced=LogisticRegression()
lr_balanced=LogisticRegression(class_weight = "balanced")
rf_unbalanced=RandomForestClassifier()
rf_balanced=RandomForestClassifier(class_weight='balanced',oob_score=True)
gbm=GradientBoostingClassifier()
#lr_unbalanced.fit(xtrain,ytrain)
#yhat= lr_unbalanced.predict(xtrain)
#metrics.accuracy_score(yhat,ytrain) ## 93.5% Training error

#preds=lr_unbalanced.predict_proba(xtrain)[:,1]
#fpr, tpr, _ = metrics.roc_curve(y, preds)
#roc_auc=metrics.auc(fpr,tpr)
#df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

#plt.plot(fpr,tpr, label='Roc is %.2f' % (roc_auc) , lw=2)

acc_roc_plot(lr_unbalanced,"lr_unbalanced",xtrain,ytrain,xtest,ytest)
acc_roc_plot(lr_balanced,"lr_balanced",xtrain,ytrain,xtest,ytest)
acc_roc_plot(rf_unbalanced,"rf_unbalanced",xtrain,ytrain,xtest,ytest)
acc_roc_plot(rf_balanced,"rf_balanced",xtrain,ytrain,xtest,ytest)
acc_roc_plot(gbm,"gbm",xtrain,ytrain,xtest,ytest)
plt.legend()
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")


#acc_roc_plot(lr_unbalanced,"lr_unbalanced",xtrain,ytrain,xtrain,ytrain)
#acc_roc_plot(lr_balanced,"lr_balanced",xtrain,ytrain,xtrain,ytrain)
#acc_roc_plot(rf_unbalanced,"rf_unbalanced",xtrain,ytrain,xtrain,ytrain)
#acc_roc_plot(rf_balanced,"rf_balanced",xtrain,ytrain,xtrain,ytrain)
#acc_roc_plot(gbm,"gbm",xtrain,ytrain,xtrain,ytrain)
#plt.legend()
#plt.xlabel("False positive rate (FPR)")
#plt.ylabel("True positive rate (TPR)")

Accuracy_list =[]
def acc_roc_plot(model,model_detail,xtrain,ytrain, xtest,ytest):
    model.fit(xtrain,ytrain)
    yhat=model.predict(xtest)
    acc = metrics.accuracy_score(yhat,ytest) ## Training error
    Accuracy_list.append({model_detail:acc}) ## Training error
    preds=model.predict_proba(xtest)[:,1]
    fpr, tpr, _ = metrics.roc_curve(ytest, preds)
    roc_auc=metrics.auc(fpr,tpr)
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    plt.plot(fpr,tpr, label='%s %.3f' % (str(model_detail), roc_auc) , lw=2)

#Since RF gives a low ROC than LR, it is required to tune the random forest model and also perform cross validation

# Trying to test the stability of the models and data, using Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(rf_balanced,xtrain,ytrain,cv=5)
cross_val_score(gbm,xtrain,ytrain,cv=5)
## GBM performs marginally better than RF on accuracy


## Model & Algorithm Tuning - Hyper parameterized tuning
rf_tuning = RandomForestClassifier(class_weight='balanced',oob_score=True)
rf_params = {
       'n_estimators': [100,500, 700,1000,2000,5000],
       'max_features': ['auto', 'sqrt', 'log2']
       }
rf_cv = GridSearchCV(rf_tuning,param_grid=rf_params, cv=5)
rf_cv.fit(xtrain,ytrain)

## Best model
RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='sqrt',
            max_leaf_nodes=None, min_impurity_split=1e-07,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=1,
            oob_score=True, random_state=None, verbose=0, warm_start=False)

acc_roc_plot(rf_cv.best_estimator_,"rf_tuned",xtrain,ytrain,xtest,ytest)
plt.legend()
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")

yhat=rf_cv.best_estimator_.predict(xtest)
acc = metrics.accuracy_score(yhat,ytest) ## Testing error

import os,sys
import Image
rf_tuned_roc = Image.open("rf_tuned_roc.jpeg")
print(rf_tuned_roc)


gbm_params = {
       'n_estimators': [100,500, 700,1000,2000,5000],
       'learning_rate': [0.1,0.5,0.01,0.05]
       }
gbm_cv = GridSearchCV(gbm,param_grid=gbm_params, cv=5)
gbm_cv.fit(xtrain,ytrain)


######################################################
## KAGGLE SUBMISSION
#####################################################
cstest=pd.read_csv("C:/Users/Vinod/Documents/Python Scripts/GiveMeSomeCredit/cs-test.csv")
cstest = cstest.drop('Unnamed: 0', 1)
cstest = cstest.drop('SeriousDlqin2yrs', 1)
cstest.head()
feat_cols= ['age','NumberOfTime30-59DaysPastDueNotWorse','RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']

oxtest=cstest.loc[:,feat_cols]
oxtest=Treat_RevolvingUtilizationOfUnsecuredLines(oxtest)
oxtest=Treat_age(oxtest)
oxtest=Treat_NumberOfTime3059DaysPastDueNotWorse(oxtest)
oxtest.DebtRatio[oxtest.DebtRatio.values > 4]=4
oxtest=NullTreatMent_Median(oxtest)
oxtest=Treat_NumberOfTimes90DaysLate(oxtest)
oxtest=Treat_NumberRealEstateLoansOrLines(oxtest)
oxtest=Treat_NumberOfTime6089DaysPastDueNotWorse(oxtest)
oxtest=Treat_NumberOfDependents(oxtest)
oxtest['TotalDebt']=oxtest.DebtRatio * oxtest.MonthlyIncome
for i in range(0,len(oxtest)):
    if oxtest.loc[i,'NumberOfDependents']> 0:
        oxtest.loc[i,'IncomePerDependent']=oxtest.loc[i,'MonthlyIncome']/oxtest.loc[i,'NumberOfDependents']
    else:
        oxtest.loc[i,'IncomePerDependent']=oxtest.loc[i,'MonthlyIncome']

oxtest.head()

yhat= rf_cv.best_estimator_.predict_proba(oxtest)[:,1]
sum(yhat) 
out=pd.DataFrame({'Probability':yhat}).to_csv("C:/Users/Vinod/Documents/Python Scripts/GiveMeSomeCredit/rf_tuned.csv")