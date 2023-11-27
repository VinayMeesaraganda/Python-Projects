#!/usr/bin/env python
# coding: utf-8

# In[23]:


#import Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Read csv file

with open("C:/vinay/Python/Credit card Fraud Detection/creditcard_2023.csv",'r') as creditcard:
    df=pd.read_csv(creditcard, index_col=None)
    


# In[3]:


# Set pandas display options
pd.set_option('display.max_columns', None)

# Display sample of csv file
df.head()


# ##### Data cleaning

# In[4]:


# check for null values

Null_values=df.isnull().any().any()
print(f'Null values in the data frame : {Null_values}')


# In[5]:


# check for duplicate values

duplicate_values=df[df.duplicated()]
print(f'Duplicate values in the data frame : {len(duplicate_values)}')


# In[6]:


#check for data types

data_types=df.info()
print(data_types)


# In[7]:


#Extract columns for outliers
df_columns=df.iloc[:,1:30].columns
print(df_columns)


# In[8]:


num_rows = 6
num_cols = 5

# Create a grid of subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through each column and create a box plot in the corresponding subplot
for i, column in enumerate(df_columns):
    sns.boxplot(data=df[column], ax=axes[i])
    axes[i].set_xlabel(column,fontweight='bold')
    axes[i].set_ylabel('Range',fontweight='bold')
    axes[i].set_title(f"Box Plot of {column} with Outliers",fontweight='bold')

# remove excess subplots
fig.delaxes(axes[-1])
    
# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plots
plt.show()


# In[10]:


for column in df_columns:
    
    #calculate quartiles and IQR
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)

    IQR=Q3-Q1

    #calculate upper and lower limits
    upper_limit=Q3+1.5*IQR
    lower_limit=Q1-1.5*IQR
    
    #resetting the outliers to upper and lower limit
    df.loc[df[column]>=upper_limit,column]=upper_limit
    df.loc[df[column]<=lower_limit,column]=lower_limit


# In[11]:


num_rows = 6
num_cols = 5

# Create a grid of subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through each column and create a box plot in the corresponding subplot
for i, column in enumerate(df_columns):
    sns.boxplot(data=df[column], ax=axes[i])
    axes[i].set_xlabel(column,fontweight='bold')
    axes[i].set_ylabel('Range',fontweight='bold')
    axes[i].set_title(f"Box Plot of {column} with Outliers",fontweight='bold')
    
# remove excess subplots
fig.delaxes(axes[-1])
    
# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plots
plt.show()


# #### Exploratory Data  Analysis

# In[19]:


df.describe()


# In[12]:


#create a correlation Matrix
correlation_matrix=df.iloc[:,1:].corr()

# set plot size
plt.figure(figsize=(25,15))

#plot heat map using seaborn
sns.heatmap(data=correlation_matrix,annot=True,cmap='coolwarm',fmt=".2f",linewidths=0.5)

#set title
plt.title("Correlation Matrix")

#show plot 
plt.show()


# In[14]:


num_rows = 6
num_cols = 5

# Create a grid of subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15)) 

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Iterate through each column and create a box plot in the corresponding subplot
for i, column in enumerate(df.iloc[:,1:]):
    sns.histplot(data=df[column],bins=25, kde=True , ax=axes[i])
    axes[i].set_xlabel(column,fontweight='bold')
    axes[i].set_ylabel('Frequency',fontweight='bold')
    axes[i].set_title(f"{column} Distribution",fontweight='bold')
    
    
# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plots
plt.show()


# In[15]:


# Observing the Amount Disribution 
sns.kdeplot(data= df['Amount'],color = 'Green', fill=True)
plt.title('Amount Distribution',size=14)
plt.show()


# ##### Data Pre-Processing

# In[16]:


x=df.iloc[:,1:30]
y=df['Class']

#import train_test split library
from sklearn.model_selection import train_test_split

# Split into train-validation
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[17]:


#checking if this is a blanaced dataset
y.value_counts()


# In[18]:


from sklearn.preprocessing import StandardScaler

#intilize standard scalar
std_scalar=StandardScaler()

#Transfrom X_train dataset into scalar
x_train=std_scalar.fit_transform(x_train)

#Transform x_test dataset into scalar
x_test=std_scalar.transform(x_test)


# #### Machine Learning

# In[32]:


#import Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score #To perform cross validation
from sklearn.metrics import accuracy_score #To calculate accuracy of the model

#intitlize logistic regression
LR=LogisticRegression(max_iter=1000)

#fit the data
LR.fit(x_train,y_train)

#predict the data
y_pred=LR.predict(x_test)

#calculate accuracy score
accuracy=accuracy_score(y_test,y_pred)

print(f"The accuracy of logistic regression is : {accuracy}")

# calculate cross validation score
cv_score=cross_val_score(LR,x,y,cv=5)
print(f"The cross validation score of Logistic Regerssion is: {cv_score}")

#calculate mean and standard deviation of cross validation scores
cv_score_mean=np.mean(cv_score)
cv_score_std=np.std(cv_score)

print(f'Mean of cross validation scores: {cv_score_mean}')
print(f'standard deviation of cross validation scores: {cv_score_std}')

#store the results
DF_ML=[]
results=[accuracy,cv_score_mean,cv_score_std]
DF_ML.append(results)


# In[33]:


#import Random forest Classiffier
from sklearn.ensemble import RandomForestClassifier

#intitlize logistic regression
RF=RandomForestClassifier()

#fit the data
RF.fit(x_train,y_train)

#predict the data
y_pred=RF.predict(x_test)

#calculate accuracy score
accuracy=accuracy_score(y_test,y_pred)

print(f"The accuracy of Random Forest Classifier is : {accuracy}")

# calculate cross validation score
cv_score=cross_val_score(RF,x,y,cv=5)
print(f"The cross validation score of Random Forest Classifier is: {cv_score}")

#calculate mean and standard deviation of cross validation scores
cv_score_mean=np.mean(cv_score)
cv_score_std=np.std(cv_score)

print(f'Mean of cross validation scores: {cv_score_mean}')
print(f'standard deviation of cross validation scores: {cv_score_std}')

#store the results
results=[accuracy,cv_score_mean,cv_score_std]
DF_ML.append(results)


# In[35]:


#import Decision Tree
from sklearn.tree import DecisionTreeClassifier

#intitlize DecisionTreeClassifier
DT=DecisionTreeClassifier()

#fit the data
DT.fit(x_train,y_train)

#predict the data
y_pred=DT.predict(x_test)

#calculate accuracy score
accuracy=accuracy_score(y_test,y_pred)

print(f"The accuracy of Decision Tree Classifier is : {accuracy}")

# calculate cross validation score
cv_score=cross_val_score(DT,x,y,cv=5)
print(f"The cross validation score of Decision Tree Classifier is: {cv_score}")

#calculate mean and standard deviation of cross validation scores
cv_score_mean=np.mean(cv_score)
cv_score_std=np.std(cv_score)

print(f'Mean of cross validation scores: {cv_score_mean}')
print(f'standard deviation of cross validation scores: {cv_score_std}')

#store the reuslts
results=[accuracy,cv_score_mean,cv_score_std]
DF_ML.append(results)


# In[46]:


#prepare a dataset for model comparision
Algo_names=["LogisticRegression","RandomForestClassifier","DecisionTreeClassifier"]
names=['Accuracy','cv_score_mean','cv_score_std']
model_comparision=pd.DataFrame(data=DF_ML,columns=names)
model_comparision.insert(0,'Model_Name',Algo_names)


# In[47]:


model_comparision


# In[62]:


fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(20,5))
axes=axes.flatten()
sns.barplot(x='Model_Name',y='Accuracy',data=model_comparision,ax=axes[0])
sns.barplot(x='Model_Name',y='cv_score_mean',data=model_comparision,ax=axes[1])
sns.barplot(x='Model_Name',y='cv_score_std',data=model_comparision,ax=axes[2])
fig.suptitle('Metrics : Model Comparision',fontsize=20,fontweight='bold')
plt.tight_layout()
plt.show()

