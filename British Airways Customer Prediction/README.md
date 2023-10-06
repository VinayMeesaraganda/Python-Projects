# British Airways Customer Booking Prediction

This project builds a model to predict if a customer will complete their flight booking on the British Airways website.

## Data Overview

The data contains 50000 rows with the following features:

- Number of passengers
- Sales channel (Online/Offline)  
- Trip type (One-way/Round trip)
- Days between booking and travel   
- Length of stay
- Departure time
- Day of travel  
- Route
- Country of booking origin
- Wants extra baggage
- Wants preferred seats
- Wants in-flight meals
- Flight duration
- Booking completed (Target variable)

## Data Preparation

- Categorical features are label encoded to numeric
- Data is checked for null and duplicate values
- Standard scaling is applied
- Train-test split of 80:20

## Model Building 

- A Random Forest Classifier model is built
- Hyperparameter tuning can be performed to improve accuracy
  
## Evaluation

- Model performance is evaluated using accuracy, precision, recall, F1 score
- Feature importance indicates most predictive features

## Conclusion

The model achieves 85% accuracy in predicting if a customer will complete the booking. The techniques used can be extended to other classification problems as well. Important features like flight duration and booking origin are identified.
