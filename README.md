# Twitter Sentiment Analysis

Repository contains a set of machine learning approaches to analyzing and predicting tweet (and other text) sentiment based on several approaches - including:

## 1. Model Overview

### 1.1 Feature Engineering

Features can be created using a variety of text and metadata extraction methods.

<strong>Note:</strong> *term* can mean any abstracted unit of importance in the texts / tweets. For example, the terms used could be specific words, specific sets of characters, usernames, hashtags, etc.

Methods for feature engineering:

##### 1.1.1 delta Term Frequency - Inverse Document Frequency (d-TF-IDF)

This feature is the relative frequency of a *term* between positive and negative sentiment and normalized over entire *document* frequencies. We can create a set of *d-TF-IDF* sparse feature matrix by performing <strong>Bag-of-words</strong> and / or extracting top most frequenctly used terms. _Note: each term used in this approach becomes a feature in the resulting matrix._

Furthermore, *d-TF-IDF* values are represented as vectors with directionality (+ / -) set between the response class types (e.g. 1 / 0 or positive / negative). The intensity of the relative frequency / use of the term in one of the response classes versus the other(s) corresponds to the absolute value. In other words, values close to *0* equate to little to no difference in the relative term use between positive and negative tweets (or whatever the response classes may be).

##### 1.1.2 Normalized Scoring

Another feature generation method is finding some type of term(s) that appears substantially throughout some set of tweets / texts. For example, we may find that there is little to no overlap between username occurrences in positive vs. negative tweets. We can create a feature with the a normalized, directional score for usernames that appear in each observation. This method can be used for special words / terms and hashtags etc.

### 1.2 XGBoost

<strong>5-Fold Cross-Validation (CV)</strong> on random parameter (*grid search*) generation through feature dimensional space in XGBoost package. Use of optimization on _i) mean error_, _ii) maximum AUC_, and a custom _iii) normalized Gini opitmization_ function. Model and feature selection as well as fine tuning were performed with a combination of deep forests, bagging, and *gbtree* boosting.

Each iteration of each model uses a random sample of <code>50% of the features (param: colsample)</code>. In addition, the CV model runs involve performing on <code>20% of observations (i.e. 5-Fold CV)</code> and the final training rounds involve sampling <code>50% of the training observations</code>.

### 1.3 SVM

Use of <strong>Support Vector Machines (SVM)</strong> with both linear and non-linear kernel approaches. Linear-SVM models use 5-Fold CV to extract optimal *Cost (C) function value(s)* and the kernel-SVM models also involve optimizing for the *Sigma / Gamma - regularization parameter*.

## 2. Ensemble Approach

Ensemble methods can be created in several ways between two or more of the models described above.

### 2.1 Models

We can perform 5-Folds CV to create the optimal set of parameters for:

i. linear-SVM
ii. kernel-SVM
iii. gradient tree-boosted logistic regression (XGBoost)

These three models can be generated through optimization for *AUC, Gini, or error (classification or misclassification accuracy)*. Combinations of the above three methods with the listed optimization types are used as the ensemble model.   

### 2.2 Averaging

We can perform *averaging* and *geometric averaging* on *n* models to generate a final prediction on our testing data. This can occur before or after thresholding if the predictions are probabilities (versus *hard predictions* - e.g. 0/1 for binary response).

### 2.3 Weighted Combinations

We can also perform weighted combinations on the models. The weights of each model can be derived from some relative measure of stability or accuracy. For example, the weight to each model could be its *Gini* score relative to the model with the best Gini score. Once weights are established, a linear or quadratic combination of the models can be performed to generate the final prediction.
