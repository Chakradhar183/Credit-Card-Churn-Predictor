# Credit Card Customers - Data Dictionary

## Dataset Overview

**Source**: [Kaggle - Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)  
**Rows**: 10,127 customers  
**Columns**: 23 features  
**Target Variable**: Attrition_Flag (Churn status)  
**Imbalance**: 16.1% churn rate (Attrited Customer)

---

## Column Descriptions

### 🔑 Identifier

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `CLIENTNUM` | Integer | Unique customer ID | 768805383 |

**Note**: This column is dropped during preprocessing as it has no predictive value.

---

### 👤 Demographic Features

| Column | Type | Description | Values/Range | Missing |
|--------|------|-------------|--------------|---------|
| `Customer_Age` | Integer | Age of customer in years | 26-73 | No |
| `Gender` | Categorical | Customer gender | M, F | No |
| `Dependent_count` | Integer | Number of dependents | 0-5 | No |
| `Education_Level` | Categorical | Highest education level | Uneducated, High School, College, Graduate, Post-Graduate, Doctorate, Unknown | No |
| `Marital_Status` | Categorical | Marital status | Married, Single, Divorced, Unknown | No |
| `Income_Category` | Categorical | Annual income bracket | Less than $40K, $40K - $60K, $60K - $80K, $80K - $120K, $120K +, Unknown | No |

**Churn Insights**:
- Age: Younger customers (<35) churn at higher rates
- Education: PhD holders have lowest churn (high value awareness)
- Income: Low income (<$40K) correlates with higher churn

---

### 💳 Account Features

| Column | Type | Description | Values/Range | Missing |
|--------|------|-------------|--------------|---------|
| `Card_Category` | Categorical | Type of credit card | Blue, Silver, Gold, Platinum | No |
| `Months_on_book` | Integer | Tenure (months since account opened) | 13-56 | No |
| `Total_Relationship_Count` | Integer | Number of products held with the bank | 1-6 | No |
| `Months_Inactive_12_mon` | Integer | Months inactive in last 12 months | 0-6 | No |
| `Contacts_Count_12_mon` | Integer | Customer service contacts in last 12 months | 0-6 | No |

**Churn Insights**:
- Card Category: Blue (basic) cards churn most; Platinum churn least
- Tenure: New customers (<24 months) are high risk
- Relationship Count: 1-2 products = high churn; 4+ products = low churn
- Inactivity: 3+ inactive months strongly predicts churn

---

### 💰 Credit Features

| Column | Type | Description | Values/Range | Missing |
|--------|------|-------------|--------------|---------|
| `Credit_Limit` | Float | Total credit limit ($) | $1,438 - $34,516 | No |
| `Total_Revolving_Bal` | Integer | Total revolving balance ($) | $0 - $2,517 | No |
| `Avg_Open_To_Buy` | Float | Average open to buy (Credit Limit - Balance) | $3 - $34,516 | No |
| `Avg_Utilization_Ratio` | Float | Average card utilization ratio | 0.000 - 0.999 | No |

**Churn Insights**:
- Credit Limit: Low limits (<$5K) → higher churn
- Revolving Balance: $0 balance = likely not using card → churn risk
- Utilization: Both very low (<10%) and very high (>70%) predict churn

**Formulas**:
- `Avg_Open_To_Buy` = Credit_Limit - Total_Revolving_Bal
- `Avg_Utilization_Ratio` = Total_Revolving_Bal / Credit_Limit

---

### 📊 Transaction Features

| Column | Type | Description | Values/Range | Missing |
|--------|------|-------------|--------------|---------|
| `Total_Amt_Chng_Q4_Q1` | Float | Change in transaction amount (Q4 over Q1) | 0.000 - 3.397 | No |
| `Total_Trans_Amt` | Integer | Total transaction amount (last 12 months) | $510 - $18,484 | No |
| `Total_Trans_Ct` | Integer | Total transaction count (last 12 months) | 10 - 139 | No |
| `Total_Ct_Chng_Q4_Q1` | Float | Change in transaction count (Q4 over Q1) | 0.000 - 3.714 | No |

**Churn Insights**:
- **STRONGEST PREDICTOR**: Total_Trans_Ct (26% importance)
- Low transactions (<40/year) = 3.5x churn risk
- Declining activity (Chng_Q4_Q1 < 1) = strong churn signal
- Amount matters less than frequency

**Transaction Patterns**:
- **Churners**: Avg 45 transactions, $3,095 amount
- **Retained**: Avg 70 transactions, $4,404 amount

---

### 🎯 Target Variable

| Column | Type | Description | Values | Distribution |
|--------|------|-------------|--------|--------------|
| `Attrition_Flag` | Categorical | Customer churn status | Existing Customer, Attrited Customer | 83.9% / 16.1% |

**Encoding**:
- `Existing Customer` → 0 (Retained)
- `Attrited Customer` → 1 (Churned)

**Class Imbalance**: 
- Majority: 8,500 retained (83.9%)
- Minority: 1,627 churned (16.1%)
- **Ratio**: 5.2:1 (requires SMOTE)

---

### 🗑️ Columns to Remove

| Column | Reason |
|--------|--------|
| `CLIENTNUM` | Unique identifier, no predictive value |
| `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1` | Leakage - prediction from another model |
| `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2` | Leakage - prediction from another model |

---

## 🔧 Engineered Features

These features are created during preprocessing:

| Feature | Formula | Business Meaning |
|---------|---------|------------------|
| `Credit_Utilization_Ratio` | Revolving_Bal / Credit_Limit | How much of credit is being used |
| `Avg_Transaction_Value` | Total_Trans_Amt / Total_Trans_Ct | Average spend per transaction |
| `Monthly_Transaction_Frequency` | Total_Trans_Ct / 12 | Transactions per month |
| `Transaction_Change_Ratio` | Q4/Q1 transaction trend | Recent vs past activity |
| `Amount_Change_Ratio` | Q4/Q1 amount trend | Recent vs past spending |
| `Credit_To_Transaction_Ratio` | Credit_Limit / Total_Trans_Amt | Underutilization indicator |
| `Inactivity_Score` | Months_Inactive / 12 | Proportion of year inactive |
| `Relationship_Depth_Score` | Composite (products + tenure + activity) | Overall engagement score (0-100) |
| `Contact_Intensity` | Contacts_Count / 12 | Monthly customer service calls |

---

## 📈 Feature Importance Ranking

Based on XGBoost model:

1. **Total_Trans_Ct** (26%) - Transaction count is king
2. **Total_Trans_Amt** (22%) - Total spend matters
3. **Total_Revolving_Bal** (14%) - Balance indicates engagement
4. **Credit_Utilization_Ratio** (12%) - Engineered feature!
5. **Total_Relationship_Count** (9%) - Cross-selling works
6. **Months_Inactive_12_mon** (7%) - Inactivity = churn
7. **Customer_Age** (4%) - Age demographics
8. **Avg_Transaction_Value** (3%) - Engineered feature!
9. **Relationship_Depth_Score** (2%) - Engineered feature!
10. **Contact_Intensity** (1%) - Service issues

**Key Insight**: Engineered features account for 17% of total importance!

---

## 🔍 Data Quality

### Missing Values
✅ **NONE** - This dataset has no missing values

### Outliers
- `Credit_Limit`: Few customers with >$30K limits (premium segment)
- `Total_Trans_Ct`: Some customers with >120 transactions (power users)
- **Treatment**: Kept outliers as they represent valid customer segments

### Data Types
- **Numerical** (14): Age, Credit_Limit, Total_Trans_Amt, etc.
- **Categorical** (7): Gender, Education_Level, Card_Category, etc.

---

## 💡 Key Business Insights from Data

1. **Transaction behavior dominates**: Activity-based features are 2x more important than demographics
2. **Engagement matters more than wealth**: Transaction count > Income category
3. **Early warning signals**: Declining Q4/Q1 ratios, increasing inactivity
4. **Product strategy**: Multi-product customers are stickier
5. **Onboarding critical**: First 24 months determine long-term retention

---

## 🎯 Recommended Preprocessing Steps

1. ✅ Drop CLIENTNUM and Naive Bayes columns
2. ✅ Encode target: Existing Customer=0, Attrited=1
3. ✅ Label encode ordinal: Education_Level, Income_Category
4. ✅ One-hot encode nominal: Gender, Marital_Status, Card_Category
5. ✅ StandardScaler for numerical features
6. ✅ SMOTE for class imbalance (only on training set)
7. ✅ Stratified train-test split (80-20)

---

**This data dictionary is essential for:**
- Understanding feature meanings during EDA
- Communicating with business stakeholders
- Interview preparation (know your data!)
- Feature engineering decisions
