- [Santander Customer Predictions (Kaggle)](#sec-1)
  - [Competition/Task Description](#sec-1-1)
  - [Brief Summary of My Exploration](#sec-1-2)

# Santander Customer Predictions (Kaggle)<a id="sec-1"></a>

## Competition/Task Description<a id="sec-1-1"></a>

The competition main page is [Kaggle: Santander Customer Transaction Predictions](https://www.kaggle.com/c/santander-customer-transaction-prediction/overview)

We are told:

"In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem."

We are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID<sub>code</sub> column. The task is to predict the value of target column in the test set, evaluated on auc.

## Brief Summary of My Exploration<a id="sec-1-2"></a>

I was initially intrigued by this competition as I was looking to do something quick with little data preprocessing and short training times. The data for this competition was also
