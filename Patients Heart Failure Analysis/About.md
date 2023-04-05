# Patients Heart Failure Analysis

In this project, we utilized the "heart_failure_clinical_records_dataset" from Kaggle and performed various data analysis techniques, 
such as data cleaning, data visualization, and data analysis using pandas, seaborn, matplotlib, and shap. 
We then developed machine learning models with hyperparameter tuning using grid search cv after performing the SMOTE technique. 
Finally, we compared the accuracy of the models to identify the best one for the given dataset.

# Dataset

Link : https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

# Insights

- The patients whose sex is 'Male' have higher chance of death rate.
- The above graph shows smoking shows a minimal affect on the functioning of heart failure.As the ratio to the deaths of smokers to non-smokers is [1:2]
- The number of heart failure patients survived is 2 times of number patients deceased. [2:1] ratio
- The number of patients admitted are high in the age range between 55 and 65.
- The chances of recoverey are very high for yonger age people i.e 70-80%
- Most of the people who admitted has 'creatinine_phosphokinase' in the range 0 to 1000
- The people who are having less'ejection_fraction' has more chances of death
- The number of follow up period increases the chances of recovery increases.
