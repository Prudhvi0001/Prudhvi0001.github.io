---
title:  "Basic Machine Learning Pipeline"
date:   2020-07-07
layout: single
author_profile: true
comments: true
classes: wide
header:
    image: "/assets/images/Posts/pipeline.png"
---



For Code Check out <a href="https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline" target="_blank">Link</a>

# Get the Status of Data
### Libraries: Pandas, NumPy Etc.

- Load data `pd.read_{filetype}('filename')`
- Get Shape of data `data.shape`
- Get the statistical information of the data `data.describe()`
- Know the Data types and info `data.info()`
- Know the count of missing values `data.isnull().sum()`



# Visualize The Data
### Libraries: Matplotlib, Seaborn, Plotly Etc.

- Plot the histograms and know the scaling of your data `data.hist(bins=10,figsize = (_,_),grid= False, Etc......)`
- Visualize the importance of each feature on the target variable `sns.FacetGrid(data, hue= "Column_name", col = "column_name", row, etc...)` and plot either `hist or scatter..`
- You can use `sns.factorplot()` for some variables 
- `sns.boxplot(x, y, data)`
- Plot the distributions if varibale has different classes by defining `kind='kde'`
- Plot the Heat Map of all the columns to get a better view `corr = data.corr()`  and  `sns.heatmap(corr, .....)`
- Plot some Vilions on the way



# Take care of missing values and outliers

- You can either replace with mean or most repeating value of in the column or know the columns that are impacting the missing columns and replace them with their respective similar column values



# Feature Engineering

- Taking care of missing categorical classes by assigning a new class to it.
- Combining similar columns to a single column.
- Binning a column into groups
- converting categorical into numerical variables such as count of the letters in a word, on hot encoding, LabelEncoder  etc..
- Triming down numercial columns for better understading.
- Fill Null values by modeling the data on a ml algorithm with null values as test data and the remaining as train data.
- Scaling all the numerical fetures into a single scale



# Modeling the Data
### Libraries: From Sklearn import Models, Metrics, Model_selection, cross_validation, etc..

- Train a small portion of the data and claculate the scores on different algorithms to find a significant increase or decrease in the scores 
- Use Kflod validation of determine a good model
- Get the important features that are helping to preidct the data better
- Filter out the important features and use tecniques like Gradient Boosting, AdaBoost Etc.
- Train the best models on whole data.
- For Better Predictions Ensemble the best models using Maximum Voting or percentages Etc.
- Submit the Predictions OR Test the model on the Test Data.