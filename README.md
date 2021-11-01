# Advanced Machine Learning Project

## Task 0

* Predict the mean of the dataset => method: just output the mean :)

## Task1

### 1. Set up the environment (using anaconda, under the root of this repository)

```shell
conda env create -f env.yml
```

### 2. Impute Missing Values

* We use median imputation since there are many outliers in the dataset, because mean is sensitive to outliers.
* We can try
  * matrix completition
  * GAN
  * iterative

### 3. Remove Outliers

* We use isolationforest to remove outliers, but the number of removed outliers is small. 
* we can try
  * Ransac
  * 

### 4. Feature Selection (Important step)

* 