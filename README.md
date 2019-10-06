# kaggle-titanic
Simple machine learning project for predicting survival of passenger on Titanic based on person information (Gender, Age, etc.)

Reference:  Dataset and problem from https://www.kaggle.com/c/titanic

We have done in two approaches:

1. Artificial Neural Network approach

Code in `main.py`. Using 2 layers neural network with either raw feature (remove some column) or custom feature from `preprocess.py`

2. Multiple machine learning approaches by https://www.kaggle.com/startupsci/titanic-data-science-solutions

Code in `Titanic_prediction.ipynb` which consist of all feature engineering above and many basic ML techniques like Decision Tree, SVM, etc.

We also add XGBoost in addition to those mentioned in the article and found that XGBoost outperformed all other approaches by a considerable margin.
