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

**i. Avoid Exact Linear Dependence:**
GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF This relationship exists among these variables. So, we only keep GrLivArea (Above grade (ground) living area square feet)



### **4. Description and units of each selected variable:**
