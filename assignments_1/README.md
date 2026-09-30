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
**19 Columns Dropped under Near duplication/high correlation:**
1. GarageArea
2. GarageYrBlt
3. YearRemodAdd
4. LotFrontage
5. ExterQual
6. BsmtQual
7. KitchenQual
8. TotRmsAbvGrd
9. Foundation
10. SaleType
11. HouseStyle
12. LotShape
13. Exterior1st
14. Exterior2nd
15. MasVnrType
16. GarageType
17. GarageQual
18. GarageCond
19. MSSubClass

##### **iii. Avoid Near-Constant Predictors:**
**For example, Utilities has near constant distribution, one category accounts for almost 99% of the records**, this characteristic of a column makes most of the columns, encoded with OHE, zero/sparse and model does not learn variation from such a feature. Linear Regression is all about finding the variation in the dependent with respect to the variation in independent variables. The most reasonable option is dropping them.

**21 Columns Dropped under Near-Constant Predictors:**
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
11. LandContour
12. MSZoning
13. Condition1
14. RoofStyle
15. BsmtCond
16. BsmtFinType2
17. Electrical
18. Functional
19. Fence
20. MiscFeature
21. PavedDrive

##### **iv. Drop Weakly Correlated Predictors:**
Some columns showed enough variation on their own, they were not near-constant, but their relationship with `SalePrice` was weak or near-zero. Since these columns don't help explain the change in the dependent variable, keeping them adds no real value to the model.

**14 Columns Dropped under Weak Correlation with Target:**
1. BsmtFullBath
2. BsmtHalfBath
3. HalfBath
4. BedroomAbvGr
5. KitchenAbvGr
6. WoodDeckSF
7. OpenPorchSF
8. EnclosedPorch
9. ScreenPorch
10. MiscVal
11. MoSold
12. YrSold
13. OverallCond
14. ExterCond

##### **v. Preserve Encoding Diversity:**
While selecting features, a mix of numerical, ordinal, and nominal predictors was intentionally kept rather than relying only on numerical columns. This gives the assignment's preprocessing and encoding steps (Step 4) genuine material to work with, and also lets us practice ordinal encoding (e.g. `HeatingQC`, `BsmtExposure`) and one-hot encoding (e.g. `Neighborhood`, `BldgType`) on real, meaningful columns rather than skipping categorical variables altogether.

### **4. Description and units of each selected variable:**

- **LotArea:** It is the size of the plot the house sits on.[Continuous numerical value, square feet].

- **LotConfig:** Lot relative to street(s) (Inside, Corner, CulDSac, FR2, FR3). [Nominal categorical value].

- **Neighborhood:** Physical location within Ames city limits. [Nominal categorical value].

- **BldgType:** Type of dwelling (single-family, duplex, townhouse, etc.). [Nominal categorical value].

- **OverallQual:** Overall material and finish of the house (v.poor to v.excellent). [Ordinal categorical value, scale 1–10].

- **YearBuilt:** Construction year. [Discrete numerical value, year].

- **MasVnrArea:** Masonry veneer area in house. [Continuous numerical value, square feet].

- **BsmtExposure:** Refers to walkout or garden level walls. [Ordinal categorical value].

- **BsmtFinType1:** Rating of the main finished basement area. [Ordinal categorical value].

- **TotalBsmtSF:** Total basement area. [Continuous numerical value, square feet].

- **HeatingQC:** Heating quality and condition. [Ordinal categorical value].

- **CentralAir:** Refers to the central air conditioning. [Binary categorical value, Y/N].

- **GrLivArea:** Above-grade (ground) living area. [Continuous numerical value, square feet].

- **FullBath:** Number of full bathrooms above ground. [Discrete numerical value, count].

- **Fireplaces:** Number of fireplaces. [Discrete numerical value, count].

- **FireplaceQu:** quality of Fireplace. [Ordinal categorical value].

- **GarageFinish:** Interior finish of the garage. [Ordinal categorical value].

- **GarageCars:** Cars capacity of Garage. [Discrete numerical value, count].

- **SaleCondition:** Condition of the sale. [Nominal categorical value].

- **SalePrice (target):** Sale price of the house. [Continuous numerical, USD($)].

# Initial Examination:

### **1. Data Types (Numerical/Categorical):**

`preprocessed_data.info()`

```
<class 'pandas.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 20 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   LotArea        1460 non-null   int64  
 1   LotConfig      1460 non-null   str    
 2   Neighborhood   1460 non-null   str    
 3   BldgType       1460 non-null   str    
 4   OverallQual    1460 non-null   int64  
 5   YearBuilt      1460 non-null   int64  
 6   MasVnrArea     1452 non-null   float64
 7   BsmtExposure   1422 non-null   str    
 8   BsmtFinType1   1423 non-null   str    
 9   TotalBsmtSF    1460 non-null   int64  
 10  HeatingQC      1460 non-null   str    
 11  CentralAir     1460 non-null   str    
 12  GrLivArea      1460 non-null   int64  
 13  FullBath       1460 non-null   int64  
 14  Fireplaces     1460 non-null   int64  
 15  FireplaceQu    770 non-null    str    
 16  GarageFinish   1379 non-null   str    
 17  GarageCars     1460 non-null   int64  
 18  SaleCondition  1460 non-null   str    
 19  SalePrice      1460 non-null   int64  
dtypes: float64(1), int64(9), str(10)
memory usage: 228.3 KB
```

### **2. Missing values (How many and which predictor variables) :**

`preprocessed_data.isnull().sum()`

```
LotArea            0
LotConfig          0
Neighborhood       0
BldgType           0
OverallQual        0
YearBuilt          0
MasVnrArea         8
BsmtExposure      38
BsmtFinType1      37
TotalBsmtSF        0
HeatingQC          0
CentralAir         0
GrLivArea          0
FullBath           0
Fireplaces         0
FireplaceQu      690
GarageFinish      81
GarageCars         0
SaleCondition      0
SalePrice          0
dtype: int64
```

**MasVnrArea:** This feature has 8 missing values, confirmed via MasVnrType also NaN, not "None"
**BsmtExposure:** This feature shows 38 missing values, but only 1 is a real missing value. Most of them are just equavilant to 'N/A' because the house has no basement.
**BsmtFinType1:** This has no Missing values, 37 are just for No basement, confirmed against the 'TotalBsmtSF'.
**FireplaceQu:** all of 690 means there is no Fireplace at all. This is confirmed with 'Fireplaces' columns, which shows zero against all NaN in the 'FireplaceQu'

### **3. Duplicate records (How many) :**

`preprocessed_data.duplicated().sum()`

```
np.int64(0)
```

This shows it has no duplicate values.

### **4. Potential outliers (How many):**
Following are the outliers calculated with IQR method. The reason is the skewness of the numerical features.

**Skewness values in dataset:**

```
LotArea        12.207688
YearBuilt      -0.613461
MasVnrArea      2.669084
TotalBsmtSF     1.524255
GrLivArea       1.366560
FullBath        0.036562
Fireplaces      0.649565
GarageCars     -0.342549
SalePrice       1.882876
dtype: float64
```

**Number of Outliers in the Numerical columns:**

```
LotArea: 69 outliers
YearBuilt: 7 outliers
MasVnrArea: 96 outliers
TotalBsmtSF: 61 outliers
GrLivArea: 31 outliers
FullBath: 0 outliers
Fireplaces: 5 outliers
GarageCars: 5 outliers
SalePrice: 61 outliers
```
