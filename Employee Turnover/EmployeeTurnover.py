#!/usr/bin/env python
# coding: utf-8

# # Understanding and Predicting Employee Turnover
# 
# ## HR Analytics
# ***

# ## Table of Contents
# ### The Problem
# - [Why is Employee Turnover a Problem?](#problem)
# 
# ### Data Quality Check
# - [Simple Inspection of Data](#datacleaning)
# 
# ### Descriptive Statistics
# - [Summary Statistics, Turnover Rate, Size of Data](#descriptive_statistics)
# - [Correlation Matrix](#correlation_matrix)
# 
# ### Exploratory Data Analysis
# - [Employee Satisfaction, Evaluation, and Project Count Distribution](#eda1)
# - [Employee Satisfaction VS Evaluation](#eda2)
# - [K Means Clustering of Employee](#clustering)
# - [Employee Satisfaction Distribution](#satisfaction)
# - [Employee Project Count Distribution](#project_count)
# - [Employee Average Monthly Hours Distribution](#avghours)
# 
# ### Simple Pre-Processing
# - [Pre-Processing: Categorical and Numerical Variable](#pre_processing)
# 
# ### Class Imbalance
# - [How to Treat Class Imbalance Problems](#class_imbalance)
# 
# ### Split Train/Test Set
# - [Splitting the Data into Train/Test Sets](#train_test_split)
# 
# ### Resample Techniques to Treat Imbalance Data
# - [Evaluate Original, Upsampled, and Downsampled Data Metrics](#resample)
# - [Choose Which Sampling Technique to Use For Model - Upsampling](#upsampling)
# 
# ### Conclusion
# - [Retention Plan](#retention_plan)
# 
# 
# 

# ***
# ### Objective: 
# - To understand what factors contributed most to employee turnover.
# 
# - To perform clustering to find any meaningful patterns of employee traits.
# 
# - To create a model that predicts the likelihood if a certain employee will leave the company or not. 
# 
# - To create or improve different retention strategies on targeted employees. 
# 
# The implementation of this model will allow management to create better decision-making actions.
# 
# ### We'll be covering:
# 1. Descriptive Analytics - What happened?
# 2. Predictive Analytics - What might happen?
# 3. Prescriptive Analytics - What should we do?
# 

# <a id='problem'></a>
# ### The Problem:
# 
# One of the most common problems at work is **turnover.** 
# 
# Replacing a worker earning about **50,000 dollars** cost the company about **10,000 dollars** or 20% of that worker’s yearly income according to the Center of American Progress.
# 
# Replacing a high-level employee can cost multiple of that...
# 
# **Cost include:**
# - Cost of off-boarding 
# - Cost of hiring (advertising, interviewing, hiring)
# - Cost of onboarding a new person (training, management time)
# - Lost productivity (a new person may take 1-2 years to reach the productivity of an existing person)
# 
# **Annual Cost of Turnover** = (Hiring + Onboarding + Development + Unfilled Time) * (# Employees x Annual Turnover Percentage)
# 
# **Annual Cost of Turnover** = (1,000 + 500) x (15,000 * 24%)
# 
# **Annual Cost of Turnover)** = 1500 x 3600
# 
# **Annual Cost of Turnover)** = 5400000
# 

# ## Example
# 
# 1. Jobs (earning under 30k a year): the cost to replace a 10/hour retail employee would be **3,328 dollars**.
# 2. Jobs (earning 30k-50k a year) - the cost to replace a 40k manager would be **8,000 dollars**.
# 3. Jobs of executives (earning 100k+ a year) - the cost to replace a 100k CEO is **213,000 dollars**.

# # Import Packages
# ***

# In[1]:


# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the Data
# ***

# In[83]:


df = pd.read_csv('HR-data.csv')


# In[49]:


# Examine the dataset
df.head()


# <a id='datacleaning'></a>
# # Data Quality Check
# ***

# In[50]:


# Can you check to see if there are any missing values in our data set
df.isnull().sum()


# In[123]:


# Rename Columns
# Renaming certain columns for better readability
df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

df.head(5)


# In[52]:


df.info()


# In[53]:


# Check the type of our features. Are there any data inconsistencies?
df.dtypes


# <a id='descriptive_statistics'></a>
# # Exploratory Data Analysis
# ***

# In[54]:


# How many employees are in the dataset?
df.shape


# In[55]:


# Calculate the turnover rate of our company's dataset. What's the rate of turnover?
turnover_rate = df.turnover.value_counts() / len(df)*100
turnover_rate


# In[56]:


# Display the statistical overview of the employees
df.describe()


# In[57]:


# Display the mean summary of Employees (Turnover V.S. Non-turnover). What do you notice between the groups?
turnover_Summary = df.groupby('turnover')
turnover_Summary


# In[58]:


df.groupby('turnover').mean()*100


# In[59]:


turnover_Summary.std()


# In[60]:


df.workAccident.value_counts()


# <a id='correlation_matrix'></a>
# ### Correlation Matrix

# In[118]:


# Create a correlation matrix. What features correlate the most with turnover? What other correlations did you find?
corr = df.corr()
size = plt.figure(figsize = (12,6))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True)
plt.title('Heatmap of Correlation Matrix')


# <a id='eda1'></a>
# # EDA 1. 
# ## Distribution of Satisfaction, Evaluation, and Project Count

# In[62]:


# Plot the distribution of Employee Satisfaction, Evaluation, and Project Count. What story can you tell?

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(df.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count')

# Graph Employee Evaluation
sns.distplot(df.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(df.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count')


# <a id='eda2'></a>
# # EDA 2.
# ## Satisfaction VS Evaluation
# 
# - There are **3** distinct clusters for employees who left the company
#  
# **Cluster 1 (Hard-working and Sad Employee):** Satisfaction was below 0.2 and evaluations were greater than 0.75. Which could be a good indication that employees who left the company were good workers but felt horrible at their job. 
#  - **Question:** What could be the reason for feeling so horrible when you are highly evaluated? Could it be working too hard? Could this cluster mean employees who are "overworked"?
# 
# **Cluster 2 (Bad and Sad Employee):** Satisfaction between about 0.35~0.45 and evaluations below ~0.58. This could be seen as employees who were badly evaluated and felt bad at work.
#  - **Question:** Could this cluster mean employees who "under-performed"?
# 
# **Cluster 3 (Hard-working and Happy Employee):** Satisfaction between 0.7~1.0 and evaluations were greater than 0.8. Which could mean that employees in this cluster were "ideal". They loved their work and were evaluated highly for their performance. 
#  - **Question:** Could this cluser mean that employees left because they found another job opportunity?

# In[63]:


sns.lmplot(x='satisfaction', y='evaluation', data=df,
           fit_reg=False, # No regression line
           hue='turnover')   # Color by evolution stage


# <a id='clustering'></a>
# ##  K-Means Clustering of Employee Turnover
# ***
# **Cluster 1 (Blue):** Hard-working and Sad Employees
# 
# **Cluster 2 (Red):** Bad and Sad Employee 
# 
# **Cluster 3 (Green):** Hard-working and Happy Employee 

# In[64]:


# Import KMeans Model
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(df[df.turnover==1][["satisfaction","evaluation"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="satisfaction",y="evaluation", data=df[df.turnover==1],
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Satisfaction")
plt.ylabel("Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Turnover")
plt.show()


# <a id='satisfaction'></a>
# # EDA 3. Employee Satisfaction
# 
# There is a **tri-modal** distribution for employees that turnovered
# - Employees who had really low satisfaction levels **(0.2 or less)** left the company more
# - Employees who had low satisfaction levels **(0.3~0.5)** left the company more
# - Employees who had really high satisfaction levels **(0.7 or more)** left the company more

# In[65]:


#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'satisfaction'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'satisfaction'] , color='r',shade=True, label='turnover')
plt.title('Employee Satisfaction Distribution - Turnover V.S. No Turnover')


# <a id='project_count'></a>
# # EDA 4. Employee Project Count 
# 
# Summary: 
# - More than half of the employees with **2,6, and 7** projects left the company
# - Majority of the employees who did not leave the company had **3,4, and 5** projects
# - All of the employees with **7** projects left the company
# - There is an increase in employee turnover rate as project count increases

# In[66]:


ax = sns.barplot(x="projectCount", y="projectCount", hue="turnover", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent")


# <a id='department'></a>
# # EDA 5. Employee Department Distribution

# In[67]:


hrleft = df[df['turnover']==1]

hrleft = pd.DataFrame(hrleft.department.value_counts()).reset_index()
hrstay = pd.DataFrame(df.department.value_counts()).reset_index()

hr_merge = pd.merge(hrleft, hrstay, how='inner', on='index')

hr_merge = hr_merge.rename(columns={"department_x":'left', "department_y":'stay', "index":'department' })
hr_merge


# In[121]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the total schools per city
sns.set_color_codes("pastel")
sns.barplot(x="stay", y='department', data=hr_merge,
            label="Total", color="b")

# Plot the total community schools per city
sns.set_color_codes("muted")
sns.barplot(x="left", y="department", data=hr_merge,
            label="Left", color="r")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set( ylabel="Department", title='Employees Per Department',
       xlabel="# of Employees")
sns.despine(left=True, bottom=True)


# <a id='avghours'></a>
# # EDA 5. Average Monthly Hours
# 
# **Summary:** 
#  - A bi-modal distribution for employees that turnovered 
#  - Employees who had less hours of work **(~150hours or less)** left the company more
#  - Employees who had too many hours of work **(~250 or more)** left the company 
#  - Employees who left generally were **underworked** or **overworked**.
# 

# In[69]:


#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')
ax.set(xlabel='Employee Average Monthly Hours', ylabel='Frequency')
plt.title('Employee AverageMonthly Hours Distribution - Turnover V.S. No Turnover')


# <a id='pre_processing'></a>
# # Pre-processing 
# ***
# 
# - Apply **get_dummies()** to the categorical variables.
# - Seperate categorical variables and numeric variables, then combine them.

# In[28]:


cat_var = ['department','salary','turnover','promotion']
num_var = ['satisfaction','evaluation','projectCount','averageMonthlyHours','yearsAtCompany', 'workAccident']
categorical_df = pd.get_dummies(df[cat_var], drop_first=True)
numerical_df = df[num_var]

new_df = pd.concat([categorical_df,numerical_df], axis=1)
new_df.head()


# In[85]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['department'] = le.fit_transform(df['department'])
df['salary'] = le.fit_transform(df['salary'])


# In[86]:


df.head()


# <a id='class_imbalance'></a>
# # Class Imbalance
# 
# ### Employee Turnover Rate: 24%

# In[73]:


df.turnover.value_counts(1)


# In[74]:


plt.figure(figsize=(12,8))
turnover = df.turnover.value_counts()
sns.barplot(y=turnover.values, x=turnover.index, alpha=0.6)
plt.title('Distribution of Employee Turnover')
plt.xlabel('Employee Turnover', fontsize=16)
plt.ylabel('Count', fontsize=16)


# # How to Treat Imbalanced Datasets
# 
# There are many ways of dealing with imbalanced data. We will focus in the following approaches:
# 
# 1. Oversampling — SMOTE

# <a id='train_test_split'></a>
# # Split Train/Test Set
# ***
# 
# Let's split our data into a train and test set. We'll fit our model with the train set and leave our test set for our last evaluation.

# In[91]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

# Create the X and y set
X=df[["evaluation",'projectCount','averageMonthlyHours','workAccident','promotion','department','salary']]
y=df['turnover']
# Define train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123, stratify=y)


# In[92]:


X_train


# In[93]:


y_train


# <a id='resample'></a>
# # Oversample Methods
# 
# Let's train a base logistic regression model on the three types of samples to see which yields the best result:
# 1. **Orginal Sample**

# In[94]:


pip install imblearn


# In[109]:


from imblearn.over_sampling import SMOTE
smote =SMOTE()


# In[110]:


#applying the smote technique
X_train_smote,Y_train_smote= smote.fit_resample(X_train.astype('float'),y_train)
X_test_smote,Y_test_smote= smote.fit_resample(X_test.astype('float'),y_test)


# In[111]:


print(y_train.value_counts())
print(Y_train_smote.value_counts())


# In[112]:


print(y_test.value_counts())
print(Y_test_smote.value_counts())


# # Train Three Models
# ***
# 
# 1. Logistic Regression
# 2. Random Forest
# 3. Support Vector Machine

# <a id='lr'></a>
# # Test Logistic Regression Performance
# ### Logistic Regression F1 Score (0.56)

# In[113]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

# Fit the model to the Upsampling data
lr = lr.fit(X_train_smote,Y_train_smote)

print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, lr.predict(X_test))

print ("Logistic Regression AUC = %2.2f" % lr_auc)

lr2 = lr.fit(X_train_smote,Y_train_smote)
print(classification_report(y_test, lr.predict(X_test)))


# <a id='rf'></a>
# # Random Forest Classifier 
# ***

# Notice how the random forest classifier takes a while to run on the dataset. That is one downside to the algorithm, it takes a lot of computation. But it has a better performance than the sipler models like Logistic Regression

# ### Random Forest F1 Score (0.97)

# In[122]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
rf = rf.fit(X_train_smote,Y_train_smote)

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))


# <a id='feature_importance'></a>
# # Random Forest Feature Importances

# In[116]:


# Get Feature Importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances


# In[117]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the Feature Importance
sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances,
            label="Total", color="b")


# # What to Optimize
# 
# We want our machine learn model to capture as much of the minority class as possible (turnover group). Our objective is to catch ALL of the highly probable turnover employee at the risk of flagging some low-risk non-turnover employee. 

# ## Conclusion
# **Binary Classification**: Turnover V.S. Non Turnover
# 
# **Instance Scoring**: Likelihood of employee responding to an offer/incentive to save them from leaving.
# 
# **Need for Application**: Save employees from leaving
# 
# In our employee retention problem, rather than simply predicting whether an employee will leave the company within a certain time frame, we would much rather have an estimate of the probability that he/she will leave the company. 
# We would rank employees by their probability of leaving, then allocate a limited incentive budget to the highest probability instances. 
# 
# Consider employee turnover domain where an employee is given treatment by Human  Resources because they think the employee will leave the company within a month, but the employee actually does not. This is a false positive. This mistake could be expensive, inconvenient, and time consuming for both the Human Resources and employee, but is a good investment for relational growth. 
# 
# Compare this with the opposite error, where Human Resources does not give treatment/incentives to the employees and they do leave. This is a false negative. This type of error is more detrimental because the company lost an employee, which could lead to great setbacks and more money to rehire. 
# Depending on these errors, different costs are weighed based on the type of employee being treated. For example, if it’s a high-salary employee then would we need a costlier form of treatment? What if it’s a low-salary employee? The cost for each error is different and should be weighed accordingly. 
#  
#  **Solution 1:** 
#  - We can rank employees by their probability of leaving, then allocate a limited incentive budget to the highest probability instances.
#  - OR, we can allocate our incentive budget to the instances with the highest expected loss, for which we'll need the probability of turnover.
# 
# **Solution 2:** 
# Develop learning programs for managers. Then use analytics to gauge their performance and measure progress. Some advice:
#  - Be a good coach
#  - Empower the team and do not micromanage
#  - Express interest for team member success
#  - Have clear vision / strategy for team
#  - Help team with career development    

# # Selection Bias
# ***
# 
# - One thing to note about this dataset is the turnover feature. We don't know if the employees that left are interns, contractors, full-time, or part-time. These are important variables to take into consideration when performing a machine learning algorithm to it. 
# 
# - Another thing to note down is the type of bias of the evaluation feature. Evaluation is heavily subjective, and can vary tremendously depending on who is the evaluator. If the employee knows the evaluator, then he/she will probably have a higher score. 

# In[ ]:




