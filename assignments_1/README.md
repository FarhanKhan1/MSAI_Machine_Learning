# Problem Statement

This model predicts sale price of residential homes in Ames, Iowa, based on physical and quality related features/characteristics of property. This model helps buyers, sellers individually and can also be integrated in a real estate platform, where it will try to predict a fair market value of a house given its attributes.

### **1. Target Variable**
**SalePrice:**
The target value is a continuous single price in ($)USDs for a single house given its features.

### **2. Predictors**
To be selected

# About the Dataset:

### **1. Source and URL:** 
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv

### **2. Number of observations:**
There are **1460** observations/records in the dataset.

### **3. Number of variables:**
There are 80 features in the dataset, 79 independent and 1 dependent variable
We are going to use strongly related variables (independent with dependent) and some of the categorical variables with medium relationship just for the practice purpose to add some categorical variables.

#### **Features Pre-Selection:**
The following points will initially select features at high level

##### **i. Avoid Exact Linear Dependence:**
**GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF**, This relationship exists among these variables. So, we only keep GrLivArea (Above grade (ground) living area square feet)
**6 Columns Dropped under Exact Linear Dependecies:**
1. 1stFlrSF 
2. 2ndFlrSF
3. LowQualFinSF 
4. BsmtFinSF1
5. BsmtFinSF2
6. BsmtUnfSF

##### **ii. Avoid Near-Duplicate/high correlation Predictors:**
**GarageArea has correlation with GarageCars**,  Generally, it is considered that a larger GarageArea will accomodate more cars. It is better to drop GarageArea in favor of GarageCars as it has (0-4) discrete numbers, easy to handle. **(corr of GarageArea with GarageCars is 0.88)**
Similarly, we do this for other near duplicate predictors to find if there is any correlation among the independent variables.
**7 Columns Dropped under Near duplication/high correlation:**
1. GarageArea
2. GarageYrBlt
3. YearRemodAdd
4. LotFrontage
5. ExterQual
6. BsmtQual
7. KitchenQual

##### **iii. Avoid Near-Constant Predictors:**
**For example, Utilities has near constant distribution, one category accounts for almost 99% of the records**, this characteristic of a column makes most of the columns, encoded with OHE, zero/sparse and model does not learn variation from such a feature. Linear Regression is all about finding the variation in the dependent with respect to the variation in independent variables. The most reasonable option is dropping them.

1. Utilities
2. Street
3. Condition2
4. RoofMatl
5. PoolArea
6. PoolQC
7. 3SsnPorch
8. LandSlope
9. Alley
10. Heating

### **4. Description and units of each selected variable:**
