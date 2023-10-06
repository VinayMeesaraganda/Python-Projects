# Stock Price Analysis and Prediction

This project analyzes and models stock price data for Cognizant using Python. 

## Data Overview

The historical daily stock price data contains the following features:

- Open Price
- High Price  
- Low Price
- Close Price
- Adjusted Close Price 
- Volume

## Analysis Steps

- Load the data and inspect it
- Create a target variable for whether next day's price will increase or decrease
- Split data into train and test sets
- Build classification models like KNN, Random Forest, SVM etc. 
- Evaluate model performance using classification metrics
- Calculate moving averages and create trading signals based on crossover
- Build classification models on moving average features

## Key Findings

- Random forest model gives highest accuracy for price direction prediction
- Using moving averages improves accuracy to 98%
- Short term averages crossing long term averages provide effective trading signals  

The analysis provides a good framework for stock data modeling and strategy building that can be extended to other stocks or financial datasets.
