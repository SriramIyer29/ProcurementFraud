
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

amldt = pd.read_csv(r"Money_Laundering_Dataset.csv")

amldt.head()

amldt.describe

## amldt Visualisations
plt.hist(amldt.type)

## here we can see maximum most transaction is with payment method

## To find skewness of amldt 
amldt.skew()
## Here we can see for most columns like amount,oldbalance,newbalance for both sender and receiever are all positively skewed
## Which is to be expected since there will be less higher values as most regular transactions will be below a certain threshold

## Box plots
sns.boxplot(x=amldt['oldbalanceOrg'])
sns.boxplot(x=amldt['newbalanceOrig'])
sns.boxplot(x=amldt['oldbalanceDest'])  
sns.boxplot(x=amldt['newbalanceDest'])   

## now to check target variable if the amldt is balanced or not

amldt['isFraud'].value_counts()

# typecasting as the amldt is binary in nature fraud and not fraud and not continuous
amldt['isFraud'] = amldt['isFraud'].astype('object')

amldt.dtypes

## now to check Missing values
amldt.isna().sum()

## Now dealing with missing values
##Since here data is positively skewed and mean imputation will give us higher mean values for the these hence we will use median imputuation

amldt.oldbalanceOrg= amldt.oldbalanceOrg.fillna(amldt.oldbalanceOrg.median())
amldt.newbalanceOrig= amldt.newbalanceOrig.fillna(amldt.newbalanceOrig.median())
amldt.oldbalanceDest= amldt.oldbalanceDest.fillna(amldt.oldbalanceDest.median())
amldt.isFraud= amldt.isFraud.fillna(amldt.isFraud.median())
amldt.newbalanceDest= amldt.newbalanceDest.fillna(amldt.newbalanceDest.median())
amldt.step= amldt.step.fillna(amldt.step.mean())

## Now lets remove the unwanted columns which are not needed in the amldt set
amldt = amldt.drop("Unnamed: 0",axis=1) 
amldt = amldt.drop(['nameOrig','nameDest','isFlaggedFraud'], axis = 1)


## Now since there are some categorical columns we use dummy variables

amldt = pd.get_dummies(amldt, columns = ['type'])

## To use label encoder to column isFraud
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


label_encoder = preprocessing.LabelEncoder()
amldt['isFraud'] = label_encoder.fit_transform(amldt['isFraud'])

# preproccessing numerical data to standard values

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
amldt_scaled = pd.amldtFrame(std_scaler.fit_transform(amldt.loc[:,~amldt.columns.isin(['isFraud'])]))
amldt_scaled.columns = amldt.columns[:-1]
amldt_scaled['isFraud'] = amldt['isFraud']

## Model Building part impoorting libraries

#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
#Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score

x = amldt.drop('isFraud',axis = 1)
y = amldt['isFraud']

## using random forest to train and test the model

# split the amldt into training and testing
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


smote = SMOTE(random_state = 42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# train a Random Forest Classifier on the resampled amldt
Random_forest = RandomForestClassifier(n_estimators = 90,random_state = 42)
Random_forest.fit(x_train_resampled,y_train_resampled)

# Evaluating the training error
y_train_pred = Random_forest.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train accuracy:", train_accuracy)

# evaluate the testing error
y_test_pred = Random_forest.predict(x_test)
test_accuracy = accuracy_score(y_test,y_test_pred)

print("Test accuracy:", test_accuracy)

# Make  predictions on the testing set 
y_pred = Random_forest.predict(x_test)

# evaluate the model's performance using accuaracy, F1 score, precision, and sensitivity
print("Accuaracy:", accuracy_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Sensitivity:", recall_score(y_test,y_pred))

conf_mat = confusion_matrix(y_test,y_pred)
conf_mat

## Hyper parameter tuning
from sklearn.model_selection import GridSearchCV


param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
Random_Forest= RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=Random_Forest, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)


Random_Forest= RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], 
                             max_depth=grid_search.best_params_['max_depth'], 
                             min_samples_split=grid_search.best_params_['min_samples_split'], 
                             random_state=42)
Random_Forest.fit(x_train, y_train)

# evaluate the model on the testing set
y_pred = Random_Forest.predict(x_test)

# evaluate the model's performance using accuaracy, F1 score, precision, and sensitivity
print("Accuaracy:", accuracy_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Sensitivity:", recall_score(y_test,y_pred))

conf_mat = confusion_matrix(y_test,y_pred)
conf_mat

## now lets try other models and compare accuracy

logreg_cv = LogisticRegression(solver='liblinear',random_state=123)
dt_cv=DecisionTreeClassifier(random_state=123)
knn_cv=KNeighborsClassifier()
nb_cv=GaussianNB()
cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',2:'KNN',3:'Naive Bayes'}
cv_models=[logreg_cv,dt_cv,knn_cv,nb_cv]

for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, x_train, y_train, cv=10, scoring ='accuracy').mean()))

## lets hyperparameter tuning Naive bayes
param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
nbModel_grid.fit(x_train, y_train)
print(nbModel_grid.best_estimator_)

#Predict with the selected best parameter
y_pred=nbModel_grid.predict(x_test)

#Plotting confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
cm

print(classification_report(y_test, y_pred, target_names=['Not Fraud','Fraud']))

import pickle
import streamlit as st
# save the model in pickle file
import os
os.getcwd()
pickle.dump(Random_Forest, open('model_dep.pkl', 'wb'))



!streamlit run app.py & npx localtunnel --port 8501