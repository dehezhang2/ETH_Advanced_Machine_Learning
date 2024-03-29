# Advanced Machine Learning Project

## Task 0

* Predict the mean of the dataset => method: just output the mean :)

## Task 1: Predict a person's age from the brain image data

### 1. Set up the environment (using anaconda, under the root of this repository)

```shell
conda env create -f env.yml
```

### 2. Impute Missing Values

* We use KNN imputer to impute the missing values. Although it is sensitive to outliers, the performance is better than median (we have also tried to impute by median value and record the index. After removing the outlier, use KNN imputer again to impute the nan values. The performance is worse than simply using KNN imputer). 
* An important issue is that we should normalize the data before imputation. 
* The hyper parameter `n_neighbors` is set to 75. 

### 3. Remove Outliers

* We use a combination of isolation forest and local outlier factor to remove outliers, but the number of removed outliers is small. The outlier is selected as an intersection of outlier selected by isolation forest and local outlier factor. 
* When removing the outliers, we concatenate the training data $X$ and the age $y$ (since outlier can be detected by observing the output value). 

### 4. Feature Selection (Important step)

* We use a combination of random forest regressor (select the feature on the top layers of the tree) and pearson correlation (select the feature has high correlation with output). Also we remove the highly correlated features.
* As we use model fusion, each model cannot have much correlation with each other. Thus, we use different hyperaparameters for feature selection. 

### 5. Model

* We use a combination of lightgbm (tree based algorithm) and Gaussian Process to predict the age. 
* For Gaussian process, we fuse two models with Matern and RationalQuadratic kernels. 
* We use K-fold validation to get the validation score, and average the prediction result of K folds as the final output. 

## Task 2: Predict heart disease class according to the ECG signal

### 1. Feature Extraction

* In this project, we use the library biopsspy ecg to get the feature. The used features includes
  * fast fourier transformation of the signal
  * reprojected signal using wavelet (using signal that is similar to the ecg)
  * PQRST indices and value
  * U wave
  * mean, median, min, max, standard deviation of 
    * Peak values 
    * rpeaks and its differentiation
    * templates 
    * heart rate and its differentiation
    * Heart rate ts and its differentiation
* We found the order of features is also important. 

### 2. Feature Selection

We then use extratree classifier to select 200 features of above feature according to the feature importance.

### 3. Model

* Gradient boosting as the model for prediction.
* Dataset expansion: To solve the imbalance problem, we expand the dataset.
* Hierarchy classifier: we use “one agains others” strategy, to divide the multi-class problem into a stack of binary classification tasks.
  * First classify noisy and non-noisy signals
  * Then classify healthy and unhealthy signals
  * Finally classify class one and two
  * For each classifier, we expand the dataset to make the binary classes balanced. 
* Voting:
  * Use 5 fold to predict the validation score and average the prediction of the results.
  * Randomly shuffle the dataset, and select results with higher validation scores. 

