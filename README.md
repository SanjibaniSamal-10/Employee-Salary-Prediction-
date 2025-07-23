# Employee-Salary-Prediction-
# Project Overview
The goal of this project is to build a predictive model that can accurately classify whether an individual's income is greater than $50,000 (>50K) or less than or equal to $50,000 (<=50K). The dataset includes various attributes that describe an individual's demographics and other relevant features, such as age, education level, occupation, and more.

# Dataset
Source: UCI Machine Learning Repository

Reference Dataset URL: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

Original Dataset URL: https://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

Number of Instances: 48,842 (38,581 for training, 9646 for testing)

Number of Attributes: 14 features + 1 target variable (Income)
# Attributes:
age: Continuous

workclass: Categorical (e.g., Private, Self-emp, Government)

fnlwgt: Continuous

education: Categorical (e.g., Bachelors, Masters, Doctorate)

education-num: Continuous

marital-status: Categorical (e.g., Married, Never-married, Divorced)

occupation: Categorical (e.g., Tech-support, Sales, Exec-managerial)

relationship: Categorical (e.g., Wife, Own-child, Husband)

race: Categorical (e.g., White, Black, Asian-Pac-Islander)

sex: Categorical (e.g., Female, Male)

capital-gain: Continuous

capital-loss: Continuous

hours-per-week: Continuous

native-country: Categorical (e.g., United-States, Canada, Mexico)

income: Categorical (Target variable: <=50K, >50K)
# Objective
The primary objective of this project is to explore and analyze the dataset to uncover the relationships between different features and the target variable (Income).
Based on the analysis, various machine learning algorithms will be applied to build a model that can predict the income class of an individual.

# Steps Involved
Data Preprocessing: Handling missing values, encoding categorical features, and normalizing numerical features.

Exploratory Data Analysis (EDA): Understanding the distribution of features and their relationships with the target variable.

Model Building: Applying different machine learning algorithms like Logistic Regression, Decision Trees, Random Forest, SVM, Gradient Boosting,Ada Boost,Naive Bayes,and KNN .

Model Evaluation: Assessing the performance of the models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
# Results
Obtained The Best Modal as Gradient Boosting Classifier With Accuracy of 86.40%  and F1 Score of 0.6622

Deployment Link : https://solid-corners-hope.loca.lt  : 34.82.201.225

# Acknowledgements
This dataset was originally extracted from the 1994 U.S. Census by Barry Becker and has been a popular choice for benchmarking machine learning algorithms. The dataset is publicly available on the UCI Machine Learning Repository.

# References
Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996.
