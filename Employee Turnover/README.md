# Understanding and Predicting Employee Turnover

This project analyzes employee data to understand factors contributing to turnover and builds a model to predict likelihood of employee turnover.

## Dataset

Link : [HR-data.csv](https://github.com/VinayMeesaraganda/Python-Projects/files/11165153/HR-data.csv)

## Data Overview

The data contains 14,999 employee records with the following features:

- Satisfaction Level 
- Last Evaluation Score
- Number of Projects  
- Average Monthly Hours
- Years at the Company
- Work Accident (1 - Yes, 0 - No)  
- Left Job (Target Variable)
- Promotion in Last 5 Years
- Department 
- Salary

## Analysis

- Performed exploratory analysis to understand distributions and correlations
- Average satisfaction is lower and evaluation scores are higher for employees who left
- Clustering reveals 3 categories of leavers: unsatisfied, underperforming, and high achievers
- Logistic regression model achieves 56% F1 score after SMOTE oversampling
- Random forest model achieves 97% F1 score after SMOTE oversampling 

## Conclusion

The random forest model accurately predicts employee turnover. It can be used to identify employees likely to leave so retention incentives can be offered. Important features are number of projects, monthly hours and evaluation scores.
