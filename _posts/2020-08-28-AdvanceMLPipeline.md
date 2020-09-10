---
title:  "Process of Creating an Effective Machine Learning Pipeline"
date:   2020-08-28
layout: single
author_profile: true
comments: true
classes: wide
header:
    image: "/assets/images/Posts/AmazonSageMaker.png"
---



To Understand the ML pipeline in an Interactive way check out the [**Explorer**](https://shrouded-harbor-01593.herokuapp.com) App.

😱: There was a time where I thought ML pipelines are something similar to OOPS in Coding languages which we have to learn to code ml models in a particular way, But that's not True at all Rather Pipelines are just a thought process on how to think and organize your ML Models. for better debugging and understanding why the model actually works In this Blog let me explain you the Nuts and Bolts of ML Pipeline so that you don't have to worry about it anymore.

Although ML pipelines are easy to understand it takes weeks or months for a company to come up with a ML Pipeline that meets the requirments Let us see why?

*Before you start diving in, building models and predicting the future ask yourself these questions.*

*Or step-1 of a ML pipeline*

### ML Problem Framing

> **What is the Business Problem for this particular scenario ?**
>
> **What are you trying to slove/Predict?**
>
> **Do you really need a Machine Learning Approach to solve this Problem?**

Here are some tips to know if you really need if you need a machine learning approach.

- Are there any repeating pattrens that you want to understand (Fraud/UnFraudlent Transactions)
- Do you have required data to understand the pattrens. (Good data : Rows = 10X the number of [features/Columns](https://en.wikipedia.org/wiki/Feature_(machine_learning)) you have)
- Understand the type of prediction that you want to make wheather it is an Binary/Multi Classification problem or you need to predict a Continuos value(Regression) such as stock price.

**Wait.................................**..*! you are not ready yet.*

Now it's time to ask Domain expects and test your assumptions (which can again be yourself if you are doing an own project)

**The more questions you ask at this stage the better your model is going to be**

- what are the important features that effect your predictions.
- Are they any feature overlaps?
- How to test your model (This is not as easy as spliting data)?
- Questions that can help you understand the domain and problem that you are solving.



*Let's get on with the boring part and dive in into actually predicting the Future.*

### Data Collection & Integration

No matter from where you or your team collects the data there might be some noise in it. So you should be equiped with tools which can help you clean and integrate the data properly and you should be able to handle all kinds of data types thrown at you. 

Although it is not intresting to do but it is useful to produce intresting results. Becoz real data doesn't consists of numerical and categorical features it consists of garbage and others [Indolence](https://www.google.com/search?client=safari&rls=en&q=indolence&ie=UTF-8&oe=UTF-8).

###### Some Handy Non Popular Data Cleaning Tools: [PrettyPandas](https://github.com/HHammond/PrettyPandas) ,  [Tabulate](https://pypi.org/project/tabulate/) , [Scrubadub](https://scrubadub.readthedocs.io/en/stable/index.html) , [ftfy](https://github.com/LuminosoInsight/python-ftfy) , [Link for More](https://mode.com/blog/python-data-cleaning-libraries/)

Here is an elobrated article by [Robert R.F. DeFilippi](https://medium.com/@rrfd/cleaning-and-prepping-data-with-python-for-data-science-best-practices-and-helpful-packages-af1edfbe2a3) and [Real Python](https://realpython.com/python-data-cleaning-numpy-pandas/) on how to get started do check it out.

##### Get the Status of Data

- Load data `pd.read_{filetype}('filename')`
- Get Shape of data `data.shape`
- Get the statistical information of the data `data.describe()`
- Know the Data types and info `data.info()`,  `data.dtypes`
- Know the count of missing values `data.isnull().sum()`

*Whooo!, My data looks stunning and I can't wait to run my model and get predictions.*

<iframe src="https://giphy.com/embed/l2JdWFvDVUYMOXNoA" width="480" height="366" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>

### Data Preparation

This is where the true skills of a data scientist comes into picture.

Take a random sample of your training data (Any size that doesn't take too much time to run) and you really need to dig deep and should be able to answer all your questions such as.

- What features are there?
- Does it match your Expectations?
- Is their enough information to make accurate predictions?
- Should you remove some of the features?
- Are their any features that are not represented properly.
- Are the labels correctly classified in the Training data?
- What is happening for incorrect data?
- Are there any missing values or outliers Etc.

If you do your homework properly you will able to build your model without any worries of wrong prediction.

How can we answer these questions programmatically?  That takes us to our next step 👇

### Data Visualization & Analysis

I love visualization, when you draw your data on a chart it reveals unseen pattrens, outliers and things the we are not able to see until now. Basic plots such as histograms and scatter plots can tell you a lot about your data 

Some of the things that you need to know are:

- Trying to Understand Label/Target summaries.
- Histograms to detect scale and distribution: [Data.plot.hist()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.hist.html)
- Scatter plot to understand the Variance/Correlation and Trend: [Data.plot.scatter()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html)
- Box Plot to detect outliers: [Data.boxplot()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html)
- Get correlation coeffecients vy [Data.corr()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr)(Default is Pearson) and plot the Heat map of these values for better understding the realtion between features and target columns by [sns.heatmap(Data.corr())](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- Plot to categorical plot onto a FacetGrid: [sns.catplot()](https://seaborn.pydata.org/generated/seaborn.catplot.html)

You don't want hurt your model right. If you data contains missing values or outliers are incorrect data types your model will suffer. In order to run your model smootly you need to take the burden of cleaning it.

Things to consider:

- Imputing missing values with mean, mode Or using non missing values to predict missing values or [Dig Deep](http://www.stat.columbia.edu/~gelman/arm/missing.pdf) ,  [Link](https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779) ,  [Link](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/)
- Taking Care of Outliers by removing or [Dig Deep](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)  [Link](https://cxl.com/blog/outliers/) , [Link](https://www.kdnuggets.com/2017/01/3-methods-deal-outliers.html) 
- Finding Correlated features Etc..

Now That part the you have been waiting so long for Selecting the model that you want to Run.

It is not easy to choose one model as their are wide range of models and all the models have their basic assumptions. choosing the model that aligns with your assumptions of the Dataset is also and Artwork done by Data Scientists.

Models are mainly classified into.

##### Supervised:

- Linear Regression, Logistic Regression, SVM, Desicion Trees, Random Forests, XGBoost, LightGBM, KNN

##### Unsupervisied:

- Clustering, K-Means, Hierarchical clustering, Fuzzy C-Means, PCA, SVD, ICA Etc..

##### Reinforcement Learning:

- Model Free RL-(TRPO, PPO, A3C, PG), Q-learning -(DQN, QR-DQN, C51, HER), Model Based RL

##### Deep Learning:

- NN, CNN, RNN, LSTM, R-CNN, GANS, Etc..

*What's Next Can I run my Model Now..........* Yes If you don't want better results (The Risk that should not be willing to take no matter the cost). These next part is arguably the crtical and time consuming part of the machine learing pipeline. It only comes with experience but I will try to explain as much as I can to help you get started. Get Ready .!

### Feature Selection & Engineering

The main ideas for Feature engineering comes from asking questions and what you have learned from visualization process above.

Questions worth asking are:

- Do these features make sense?
- How can I engineer new features based on visualizations.
- Can I create new features by combining old features.

Tips for Feature Engineering:

- Combining similar columns to a single column.

- Converting categorical into numerical variables such as count of the letters in a word or one hot encoding, LabelEncoder  Etc.
- Converting time into days or months.
- Scaling the Numerical features so that model find local minima easily.
- Binning column data into groups.(It also useful for handling outliers)

Dig Deep - [ML Mastery](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/) ,  [Medium](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114) ,  [Kaggle](https://www.kaggle.com/search?q=feature+engineering) ,  [Elite](https://elitedatascience.com/feature-engineering) ,  [Elite2](https://elitedatascience.com/feature-engineering-best-practices)

*WOOOOH......Atlast! Finally ready for training huh huh **Sort of**:*

## Model Training

Steps for Model Training:

- Split Data into **Train, Dev, Test:**
  - Make sure to Randomize your Data to overcome Bias when you run your Model.
  - Create a test set that closely represnts the train set (Stratified Test set, Same Date ranges Etc..)
  - Using Cross Validation.

- **Bias** and **Variance:** 
  - Underfitting = Low variance and High Bias = Simple Model
  - Over Fitting = Low Bias and High variance = Complex Model
  - [Hyparmeter Tuning](https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568) helps you balance the model bias and variance.

You Did IT........

<iframe src="https://giphy.com/embed/ZcKASxMYMKA9SQnhIl" width="480" height="445" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>

It's time to collect your salary. RUN the model. 

###### Libraries : [Mxnet](https://mxnet.apache.org/versions/1.6/), [Scikit-Learn](https://scikit-learn.org/stable/), [TensorFlow](https://www.tensorflow.org), [PyTorch](https://pytorch.org), [Cloud AutoML](https://cloud.google.com/automl), [Pycaret](https://pycaret.org), [Theano](http://deeplearning.net/software/theano/), [Keras](https://keras.io), [SciPy](https://www.scipy.org), [NLTK](https://www.nltk.org), [MLlib](https://spark.apache.org/mllib/), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/),  [Many More](https://www.g2.com/products/scikit-learn/competitors/alternatives)

**Sorry! It's not done Yet.** There are some deatils to evaluate and intrepet which makes you stand out from the crowd such as:

### Model Evaluation

There are different metrics to evaluate your model, sometimes you may have to come up with your own metric to evaluate the model performance. Anyways here are some of common metrics that helps you understand what is going wrong in the model or how accurate is the model for real world or to understand why is the model even working in the first place. [Choosing the right metric](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4)

*Don't just run single model run multiple Regression/Classification models and chosse the best one based on the metrics here.* 

##### Regression Metrics:

- MSE, RMSE, $R^2$ , Adjusted $R^2$  [Dig Deep](https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914) ,  [Interveiw Prespective](https://towardsdatascience.com/metrics-to-understand-regression-models-in-plain-english-part-1-c902b2f4156f)

##### Classification Metrics:

- Confusion Matrix, Precision, Recall, F1-score, ROC, AUC  [Dig Deep](https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)

*The End:*

Face it you have framed the problem, Cleaned the data, Did Data visualization, Selected important feature and did feature engineering, You trained your model again and again by tuning the hyper parameters and testing on Dev set what's left.

### Prediction

**`Model.Predict(Test_Data)`** And Deploy (This is an Entirely Seprate Topic )



> There may be a question still bugging in your mind.

**Everything is Fine, But all of this is Just Theory how can I actually implement it.?**

*Ans: As I said earlier Pipelines are just a thought Process to solve a problem in systematic manner so that it will be easy to debug in the future: There are tools like  **[Amazon SageMaker](https://aws.amazon.com/sagemaker/)** which takes care of all these steps at high scale and provides with you extra featuresand tools to organize, mointor, track your models*

### TIPS

- When you want to test multiple models it is also advisable to test the models on small datasets rather than running on entire datasets.
- Check the important features that helped and model and continue to feature engineering again on this features.
- Drop the features which are not correlated with the target variable in any way.
- Ensmeble different models for better prediction and randomization using Maximum Voting or Percentages.
- Train the best model on both train and dev set best testing on test set.
- Modeling is not just a one time step you need the right executable code to Re-run your model to be updated with the current needs.

Check out this [Blog]() (Yet to Come) were this steps were used to solve a real world example. 

Let me known in the comments if you have learned something new today.