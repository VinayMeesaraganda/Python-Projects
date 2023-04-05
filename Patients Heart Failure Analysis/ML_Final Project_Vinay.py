#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("heart_failure_clinical_records_dataset.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# ### Explanatory Data Analysis (EDA)

# In[6]:


sns.countplot(data=df,x='DEATH_EVENT')


# In[7]:


df1=df[['anaemia','diabetes','high_blood_pressure','sex','smoking','DEATH_EVENT']]


# In[8]:


fig, axes = plt.subplots(2,3, figsize=(18,14))
for i, ax in zip(df1, axes.flat):
    sns.countplot(data=df1,x=df1[i],hue='DEATH_EVENT', ax=ax)
plt.show()

The patients having 'Anaemia' and whose sex is 'Male' have higher chance of death rate.
The above graph shows smoking shows a minimal affect on the functioning of heart failure.
As the ratio to the deaths of smokers to non-smokers is [1:2]
The number of heart failure patients survived is 2 times of number patients deceased. [2:1] ratio
# In[9]:


sns.histplot(data=df, x="age",multiple="stack", hue="DEATH_EVENT")

The number of patients admitted are high in the age range between 55 and 65.
Most of the deceased also falls under the range of 55 to 75
The chances of recoverey are very low for old age people i.e 10-20%
The chances of recoverey are very high for yonger age people i.e 70-80%
# In[10]:


df2=df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time','DEATH_EVENT']]


# In[11]:


df2.head()


# In[12]:


fig, axes = plt.subplots(2,4,figsize=(18,14))
for i, ax in zip(df2, axes.flat):
    sns.histplot(data=df1,x=df2[i],multiple="stack", hue="DEATH_EVENT", ax=ax)
plt.show()

Most of the people who admitted has 'creatinine_phosphokinase' in the range 0 to 1000
The people who are having less'ejection_fraction' has more chances of death
The people who are having platelets count close to 200000 and more have higher chances of recoverey.
The patients having serum sodium in the range of 137 to 142 have more chances of recovery. 
The number of follow up period increases the chances of recovery increases.
# In[13]:


plt.figure(figsize=(10,8))
corelation_matrix=df.corr()
sns.heatmap(corelation_matrix,annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[14]:


sorted_values=corelation_matrix['DEATH_EVENT'].sort_values()
print(sorted_values)


# In[15]:


# sns.pairplot(data=df[['creatinine_phosphokinase','anaemia','high_blood_pressure','age','serum_creatinine','DEATH_EVENT']])


# ## Machine Learning Modeling

# In[16]:


x=df.iloc[:,:-1]
y=df.DEATH_EVENT
x.head()


# In[17]:


y.head()


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[19]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


# #### Logistic Regression (LR)

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)

#predict the data
y_pred = logreg.predict(x_test)

#print Accuracy for Logistic Regression
print("Accuracy:",round(logreg.score(x_test, y_test),2))

#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# #### Hyper Parameter Tuning by calling the GridSearchCV method.

# In[21]:


parameters = {'penalty' : ['l1','l2'], 'C': np.logspace(-3,3,7),'solver': ['newton-cg', 'lbfgs', 'liblinear']}


# In[22]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(logreg,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='roc_auc',        # metric for scoring
                   cv=10)                     # number of folds
grid.fit(x_train,y_train)
print("Tuned Hyperparameters :", grid.best_params_)


# In[23]:


logreg = LogisticRegression(C = 0.1, penalty = 'l2',solver = 'liblinear')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
print("Accuracy:",logreg.score(x_test, y_test))
#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# In[24]:


from sklearn.metrics import roc_curve
fpr,tpr,thresh= roc_curve(y_test,logreg.predict_proba(x_test)[:,1],pos_label=1)
#Plot ROC Curve
plt.plot(fpr, tpr, linestyle='--',color='orange', label='Logistic Regression')
plt.plot([0, 1], [0, 1], "k--", color='blue',label="chance level (AUC = 0.5)")
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
auc_score1 = roc_auc_score(y_test,y_pred)
print("Auc_score:",round(auc_score1,2))


# In[25]:


Accuracy_score=[]
AUC_score=[]
name=[]
Accuracy_score.append(logreg.score(x_test, y_test))
AUC_score.append(auc_score1)
name.append("Logistic Regression")


# #### Decision Tree (DT)

# In[26]:


from sklearn.tree import DecisionTreeClassifier

# instantiate the model (using the default parameters)
Desc_Tree=DecisionTreeClassifier()

# fit the model with data
Desc_Tree.fit(x_train,y_train)

#predict the data
y_pred = Desc_Tree.predict(x_test)

#print Accuracy for Logistic Regression
print("Accuracy:",round(Desc_Tree.score(x_test, y_test),2))

#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# #### Hyper Parameter Tuning by calling the GridSearchCV method.

# In[27]:


DTparameters={'criterion' : ['gini', 'entropy'],
            'max_features': ['log2', 'sqrt','auto'],
            'max_depth': [1,2, 3, 5, 10,None],
            'random_state' : [0,1,2,3,4,5]
           }


# In[28]:


grid= GridSearchCV(Desc_Tree,                 # model
                   param_grid = DTparameters,   # hyperparameters
                   scoring='roc_auc',        # metric for scoring
                   cv=10)                     # number of folds
grid.fit(x_train,y_train)
print("Tuned Hyperparameters :", grid.best_params_)


# In[29]:


Desc_Tree = DecisionTreeClassifier(criterion='gini',max_depth=3,max_features='log2',random_state=3)
Desc_Tree.fit(x_train,y_train)
y_pred = Desc_Tree.predict(x_test)
print("Accuracy:",round(Desc_Tree.score(x_test, y_test),2))
#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# In[30]:


from sklearn.metrics import roc_curve
fpr,tpr,thresh= roc_curve(y_test,Desc_Tree.predict_proba(x_test)[:,1],pos_label=1)
#Plot ROC Curve
plt.plot(fpr, tpr, linestyle='--',color='orange', label='Decision Tree')
plt.plot([0, 1], [0, 1], "k--",color='blue', label="chance level (AUC = 0.5)")
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
auc_score2 = roc_auc_score(y_test,y_pred)
print("Auc_score:",round(auc_score2,3))


# In[31]:


Accuracy_score.append(Desc_Tree.score(x_test, y_test))
AUC_score.append(auc_score2)
name.append("Decision Tree")


# #### Random Forest (RF)

# In[32]:


from sklearn.ensemble import RandomForestClassifier

# instantiate the model (using the default parameters)
RF=RandomForestClassifier()


# fit the model with data
RF.fit(x_train,y_train)

#predict the data
y_pred = RF.predict(x_test)

#print Accuracy for Logistic Regression
print("Accuracy:",round(RF.score(x_test, y_test),2))

#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# #### Hyper Parameter Tuning by calling the GridSearchCV method.

# In[33]:


RFparameters={'bootstrap': [True], 
            'max_depth': [5, 10, None], 
            'max_features': ['auto', 'sqrt', 'log2'],
            'n_estimators': [200,700]}


# In[34]:


grid= GridSearchCV(RF,                        # model
                   param_grid = RFparameters, # hyperparameters
                   scoring='roc_auc',        # metric for scoring
                   cv=10)                     # number of folds
grid.fit(x_train,y_train)
print("Tuned Hyperparameters :", grid.best_params_)


# In[35]:


RF= RandomForestClassifier(bootstrap=True, max_depth=5,max_features='sqrt',n_estimators=700)
RF.fit(x_train,y_train)
y_pred = RF.predict(x_test)
print("Accuracy:",round(RF.score(x_test, y_test),2))
#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# In[36]:


from sklearn.metrics import roc_curve
fpr,tpr,thresh= roc_curve(y_test,RF.predict_proba(x_test)[:,1],pos_label=1)
#Plot ROC Curve
plt.plot(fpr, tpr, linestyle='--',color='orange', label='Random Forest')
plt.plot([0, 1], [0, 1], "k--",color='blue', label="chance level (AUC = 0.5)")
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
auc_score3 = roc_auc_score(y_test,y_pred)
print("Auc_score:",round(auc_score3,3))


# In[37]:


Accuracy_score.append(RF.score(x_test, y_test))
AUC_score.append(auc_score3)
name.append("Random Forest")


# #### XGBoost (XGB)

# In[38]:


from xgboost import XGBClassifier
# instantiate the model (using the default parameters)
XGB=XGBClassifier(eval_metric='logloss')


# fit the model with data
XGB.fit(x_train,y_train)

#predict the data
y_pred = XGB.predict(x_test)

#print Accuracy for Logistic Regression
print("Accuracy:",round(XGB.score(x_test, y_test),2))

#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# In[39]:


XGBparameters={"colsample_bytree": [ 0.3, 0.5 , 0.8 ],
            "reg_alpha": [0, 0.5, 1, 5],
            "reg_lambda": [0, 0.5, 1, 5]
           }


# In[40]:


grid= GridSearchCV(XGB,                        
                   param_grid = XGBparameters, 
                   scoring='roc_auc',
                   refit='recall', 
                   n_jobs=-1,
                   cv=10)                     
grid.fit(x_train,y_train)
print("Tuned Hyperparameters :", grid.best_params_)


# In[41]:


XGB= XGBClassifier(colsample_bytree=0.5, reg_alpha=1,reg_lambda=0.5,eval_metric='logloss')
XGB.fit(x_train,y_train)
y_pred = XGB.predict(x_test)
print("Accuracy:",round(XGB.score(x_test, y_test),2))
#Display Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm,annot=True,fmt='g')


# In[42]:


from sklearn.metrics import roc_curve
fpr,tpr,thresh= roc_curve(y_test,XGB.predict_proba(x_test)[:,1],pos_label=1)
#Plot ROC Curve
plt.plot(fpr, tpr, linestyle='--',color='orange', label='Random Forest')
plt.plot([0, 1], [0, 1], "k--",color='blue', label="chance level (AUC = 0.5)")
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
auc_score4 = roc_auc_score(y_test,y_pred)
print("Auc_score:",round(auc_score4,3))


# In[43]:


Accuracy_score.append(XGB.score(x_test, y_test))
AUC_score.append(auc_score4)
name.append("XGBOOST")


# ## Summary For Machine Learning Modeling

# In[44]:


scores = pd.DataFrame({'name': name,'Accuracy_score': Accuracy_score,'AUC_score': AUC_score})
scores


# ## Machine Learning Interpretability/Explanability

# In[60]:


import eli5
from eli5.sklearn import PermutationImportance
import lime
from lime import lime_tabular
import shap


# ##### 3_A) Logistic Regression

# In[46]:


eli5.show_weights(logreg.fit(x_test,y_test), feature_names=x.columns.tolist())


# In[47]:


negative_row=x_test[np.where(y_test==0)[0][0]]
print('Negative Row:',negative_row)
positive_row=x_test[np.where(y_test==1)[0][0]]
print('Positive Row:',positive_row)


# In[48]:


eli5.show_prediction(logreg,positive_row,feature_names=x.columns.tolist())


# In[49]:


eli5.show_prediction(logreg,negative_row,feature_names=x.columns.tolist())


# ##### 3_B) Decision Tree

# In[50]:


eli5.show_weights(Desc_Tree.fit(x_test,y_test), feature_names=x.columns.tolist())


# In[51]:


eli5.show_prediction(Desc_Tree,positive_row,feature_names=x.columns.tolist())


# In[52]:


eli5.show_prediction(Desc_Tree,negative_row,feature_names=x.columns.tolist())


# #### 3c) Random Forest

# In[53]:


from lime.lime_tabular import LimeTabularExplainer


# In[54]:


classes=[0,1]
all_feat=x.columns
explainer = lime.lime_tabular.LimeTabularExplainer(x_train,mode='classification',feature_selection= 'auto',
                                                  class_names=classes,feature_names = all_feat, 
                                                   kernel_width=None,discretize_continuous=True)


# In[55]:


exp=explainer.explain_instance(positive_row, RF.predict_proba,num_features=len(x_train))
exp.show_in_notebook(show_table=True,show_all=True)
print("R2 score:",exp.score)


# In[56]:


exp=explainer.explain_instance(negative_row, RF.predict_proba,num_features=len(x_train))
exp.show_in_notebook(show_table=True,show_all=True)
print("R2 score:",exp.score)


# #### 3c) XG Boost

# In[57]:


exp=explainer.explain_instance(positive_row, XGB.predict_proba,num_features=len(x_train))
exp.show_in_notebook(show_table=True,show_all=True)
print("R2 score:",exp.score)


# In[58]:


exp=explainer.explain_instance(negative_row,XGB.predict_proba,num_features=len(x_train))
exp.show_in_notebook(show_table=True,show_all=True)
print("R2 score:",exp.score)


# #### 3D) SHAP XGBOOST

# In[59]:


explainer = shap.TreeExplainer(XGB)
shap_values = explainer.shap_values(x_train)


# In[ ]:


shap_values1=explainer.shap_values(negative_row.reshape(1,-1))
shap.initjs()
shap.force_plot(explainer.expected_value,shap_values1,positive_row,feature_names=x.columns)


# In[ ]:


shap_values2=explainer.shap_values(positive_row.reshape(1,-1))
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values2,positive_row,feature_names=x.columns)


# In[ ]:


shap.summary_plot(shap_values, features=x_train, feature_names=x.columns,plot_type='bar')


# In[ ]:


shap.force_plot(explainer.expected_value, shap_values[0,:], features=x.columns)


# In[ ]:


shap_values


# In[ ]:


shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([x.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['feature', 'importance']
importance_df = importance_df.sort_values('importance', ascending=False)
print(importance_df)


# In[61]:


df=pd.DataFrame([(np.abs(shap_values).mean(0)).tolist(),x.columns.tolist()], columns=["feature", "importance" ])


# In[62]:


x_train


# ### 4) Predict observations for Postive row

# In[63]:


xpositive_row=x_test[np.where(y_test==1)[0][0]]
xnegative_row=x_test[np.where(y_test==0)[0][0]]


# In[64]:


print ("Logistic Regression :")
predict1=logreg.predict_proba(xpositive_row.reshape(1,-1))[0]
print("\tpredicted value :",predict1)
print ("Decision Tree:")
predict2=Desc_Tree.predict_proba(xpositive_row.reshape(1,-1))[0]
print("\tpredicted value :",predict2)
print ("Random Forest:")
predict3=RF.predict_proba(xpositive_row.reshape(1,-1))[0]
print("\tpredicted value :",predict3)
print ("XGB Classifier :")
predict4=XGB.predict_proba(xpositive_row.reshape(1,-1))[0]
print("\tpredicted value :",predict4)


# ### 4) Predict observations for negative row

# In[65]:


print ("Logistic Regression :")
predict1=logreg.predict_proba(xpositive_row.reshape(1,-1))[0]
print("\tpredicted value :",predict1)
print ("Decision Tree:")
predict2=Desc_Tree.predict_proba(xnegative_row.reshape(1,-1))[0]
print("\tpredicted value :",predict2)
print ("Random Forest:")
predict3=RF.predict_proba(xnegative_row.reshape(1,-1))[0]
print("\tpredicted value :",predict3)
print ("XGB Classifier :")
predict4=XGB.predict_proba(xnegative_row.reshape(1,-1))[0]
print("\tpredicted value :",predict4)


# According to my observations XGBoost out performed Every other Algorithm and for the Positive row its value is 0.95 and for the negative row it is 0.93
