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

# Splitting dataset(train & test), Preprocessing pipeline [missing values, outliers and feature scaling]:

### **1. Splitting Dataset into training and testing:**

1. The dataset is split into Training(75%) and Test(25%), using a fixed random seed for reproducibility.
2.  Since, the Dependent Variable is highly right-skewed (1.88, see Step 3: Potential outlier), To avoid meaningless distribution of data in train and test sets, the split is **stratified** on `SalePrice`, binned into 5 quantile-based strata (`pd.qcut`)
3. To represent price range meaningfully, we take 5 bins to have enough strata. This way, each stratum gets (219(training) + 73(test) = 290). This generate reliable split given the skewed distribution.

`price_bins = pd.qcut(preprocessed_data["SalePrice"], q=5, labels=False)`
`train_data, test_data = train_test_split(
    preprocessed_data,
    test_size=0.25,
    random_state=42,
    stratify=price_bins
)`

`print(train_data.shape)`
`print(test_data.shape)` 

```
(1089, 20)
(364, 20)
```

### **2. Handling Missing Values:**

Most `NaN` values in the dataset are not genuinely missing rather represents "feature absent" case (no basement, no fireplace, no garage) other than the following 2 features.

- **`MasVnrArea`**: 8 genuinely missing values.

- **`BsmtExposure`**: 1 genuinely missing value.

**Approach:**

1. **`MasVnrArea`**: we impute with **median**, rather than **mean**, because the column is right-skewed (skew = 2.67).

2. **`BsmtExposure`'s single genuine missing row**: Before the train/test split, we drop the record,which is okay and does not effect the dataset, instead of imputing this single row with other related complexities like changing the representation of other NaN values(shows absent feature only).

3. **"feature absent" columns** (`BsmtExposure`'s remaining 37 "no basement" rows, `BsmtFinType1`, `FireplaceQu`, `GarageFinish`): filled with constant `"None"` category, rather than a statistical imputation. This preserves the meaning "this feature does not exist" and also introduces another category. Later, we encode it as a lowest rank value in ordinal columns.

We fit imputation statistics to training dataset only and apply identically to the testing, which helps in avoiding data leakage.

### **3. Handling Outliers:**

We use **IQR method** due to the skewness in the Continuous features (mentioned under the Potential outliers heading above). instead of mean based techniques like Z-Score.

**Approach:**

We review each flagged column individually (via scatter plot and distribution shape), that distinguishes genuine anomalies than legitimately rare but valid records. This helps in deciding to keep them rather than removing them.

- **`GrLivArea`**: 2 rows in this dataset, where unusually large living areas (> 4000 sq ft) are paired with unusually low sale prices, have been removed due to negligable number of records.

- **`LotArea`**: 4 additional rows with extreme lot sizes (> 100,000 sq ft), were far beyond the rest of the distribution can disproportionately influence our model.

In total, **6 rows were removed** as genuine outliers, out of 1,460. This removal was done **before the train/test split**, since it is a fixed, rule-based decision, decreasing the risk of data leakage between train and test sets.

### **4. Encoding Catregorical predictors:**

### **4. Encoding Categorical predictors:**

The categorical predictors fall into two types, each needs its own encoding techniques:

**Ordinal columns (6)** — `OrdinalEncoder` is used with an **explicitly defined category order** per column (not the default alphabetical order),which ensures higher-order categories map to higher numeric values. `None` shows absent feature, which sounds lowest category naturally.

- `BsmtExposure`: None < No < Mn < Av < Gd
- `BsmtFinType1`: None < Unf < LwQ < Rec < BLQ < ALQ < GLQ
- `HeatingQC`: Po < Fa < TA < Gd < Ex
- `FireplaceQu`: None < Po < Fa < TA < Gd < Ex
- `GarageFinish`: None < Unf < RFn < Fin
- `OverallQual`: This is already stored as integers (1–10) in the raw data, so there is no need to transform it.

**Nominal columns (5)** — no natural order, so `OneHotEncoder` is used. `Neighborhood` (25 categories, currently, we simply OHE them, If there is any issue with the model scores, we revisit and split it into ranges to reduce the dimension), `CentralAir` (binary), `LotConfig`, `BldgType`, `SaleCondition`.

`drop="first"` is applied to  avoid exact linear dependence between the resulting dummy columns and the model's intercept term.

This produces 38 columns from the 5 nominal predictors, so the total feature count becomes 52, previously 19.

### **5. Scaling (Standardization):**

Since this assignment implements Gradient Descent and Stochastic Gradient Descent (Q1, Step 6–7), feature scaling is a requirement, not just good practice: unscaled features with very different ranges and units (e.g. `LotArea` in square feet vs. `FullBath` as a 0–3 count) cause slow or unstable convergence for gradient-based optimization.

**Approach:**

For full consistency across the input features, we scale all of the 52 features including dummy OHE instead of just scaling originally continuous columns. This is because it helps Gradient Decent algorithm in convergence more conveniently than otherwise. We use `StandardScaler` (mean 0, standard deviation 1) which is beneficial for skewed data plus it also keep the outliers so our look alike outliers should stay same as we do not want to change these legitimate rare outliers.  

To keep the Evaluation step easy and more comparing friendly, we do not scale the `SalePrice`. This is intentional approach, which is also common among the professionals.

**Verification:** after transformation, we have mean of ≈0 (4.18e-17) and a std of ≈1 (1.0005), which confirms the scaling works fine.


# Examine multicollinearity among the predictors using:

### Context: Correlation's Role in Feature Selection

During feature pre-selection (Rule ii: "Avoid Near-Duplicate/High Correlation Predictors"), the correlation among the predictors helps in resolving the issues of similar features. 

- `GarageCars` vs. `GarageArea` (corr = 0.88) — `GarageArea` was dropped
- `OverallQual` vs. [`ExterQual`, `BsmtQual`, `KitchenQual`] the corr ranges from [0.66 to 0.73], so we drop three, and only keep `OverallQual`
- `YearBuilt` vs. `GarageYrBlt` (corr = 0.83) — `GarageYrBlt` was dropped
- `TotRmsAbvGrd` vs. `GrLivArea` (corr = 0.83) — `TotRmsAbvGrd` was dropped

In each case, when two predictors showed strong correlation (roughly ≥ 0.6–0.7) and carried largely overlapping information, only one was retained, to avoid the instability and misleading coefficients that multicollinearity causes in linear regression.
To avoid the misleading coefficients and instability, that can be caused by multicollinearity in Linear Regression, we only retain one feature among high correlated features. The threshold is roughly 0.65+ 

Among the remaining 19 predictors, **no pair crosses this threshold**, meaning the earlier selection process was effective. These details can be found in the `data_understanding.ipynb`.

# If GD or SGD is selected:

1. **Standardize predictors:** All 52 features have been standardized via `StandardScaler`.
2. **Learning rate:** eta0 = 0.01, with learning_rate='invscaling' decays as (eta0 / iteration^0.25).
3. **Stopping criterion:** When loss improves by less than **tol = 0.001** for **n_iter_no_change = 5** consective rows.
4. **Iterations or epochs:** **n_iter_ = 25** (from the fitted model)

# Evaluate each implementation on the training and test sets using:

### SVD:

1. Train R²: 0.8893185936134853
2. Test R²: 0.8519552138826101
3. Train MAE: 17987.927328373222
4. Test MAE: 20082.50240091017
5. Train RMSE: 25937.810519237744
6. Test RMSE: 32037.122660524805

### SGD:

1. Train R²: 0.8889205237492102
2. Test R²: 0.8507325439003631
3. Train MAE: 17876.772237698242
4. Test MAE: 19934.165617263185
5. Train RMSE: 25984.41181429236
6. Test RMSE: 32169.144481815976

