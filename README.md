# Credit Card Customer Churn Prediction

End-to-end machine learning project that predicts which credit card customers are likely to churn, built with Python, XGBoost, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Live Demo:** [bankchurnerscsv-guwhaaalen4z8x8gkkht5s.streamlit.app](https://bankchurnerscsv-guwhaaalen4z8x8gkkht5s.streamlit.app/)

---

## Project Overview

Customer churn is expensive. Acquiring a new banking customer costs 5 to 25 times more than retaining an existing one. This project builds a classification pipeline to flag at-risk customers before they leave, giving retention teams time to act.

The pipeline covers everything from data cleaning and feature engineering to model training and a live Streamlit dashboard. You can input a single customer's details and get back a churn probability with a risk level (Low, Medium, or High), or explore the full dataset through interactive charts and filters.

The best performing model is XGBoost, which achieved **0.99 ROC-AUC** and **0.90 F1-score** on the test set.

---

## Features

- Trains and compares 4 classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Custom feature engineering with 9 derived features like credit utilization ratio, transaction frequency, and relationship depth score
- Handles class imbalance using SMOTE and class weighting
- Streamlit web app with two pages: single-customer prediction and a full analytics dashboard
- Interactive Plotly charts with filters for gender, income, card type, age range, and risk level
- Batch predictions across the full dataset with downloadable CSV export
- Model persistence using joblib (model, preprocessor, and metadata saved separately)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Data Processing | pandas, NumPy |
| ML / Modeling | scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP |
| Web App | Streamlit |
| Serialization | joblib |

---

## Dataset

**Source:** [Credit Card Customers, Sakshi Goyal (Kaggle)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

- 10,127 customers, 23 features
- Target: `Attrition_Flag` (Existing Customer vs. Attrited Customer)
- Class split: roughly 84% retained, 16% churned
- The original dataset includes two Naive Bayes classifier columns at the end that leak the target. These are dropped during preprocessing.

---

## Project Workflow

```
Raw Data (Kaggle CSV)
    |
    v
Preprocessing
    - Drop leaky columns (Naive Bayes outputs)
    - Encode categoricals (ordinal for education/income, one-hot for gender/marital/card)
    - StandardScaler on all numeric features
    - Stratified train-test split
    - SMOTE on training set only
    |
    v
Feature Engineering (9 new features)
    - Credit_Utilization_Ratio (revolving balance / credit limit)
    - Avg_Transaction_Value (total amount / transaction count)
    - Monthly_Transaction_Frequency (transactions per month)
    - Relationship_Depth_Score (weighted score of products, tenure, activity; 0-100)
    - Inactivity_Score (months inactive / 12)
    - Contact_Intensity, Credit_To_Transaction_Ratio, and others
    |
    v
Model Training (4 classifiers)
    - Logistic Regression, Decision Tree, Random Forest, XGBoost
    - Compared on accuracy, precision, recall, F1, ROC-AUC
    - Best model auto-saved to models/
    |
    v
Streamlit Dashboard
    - Single customer prediction with gauge chart
    - Full dataset analytics with interactive filters
```

---

## Models Used

| Model | Why |
|-------|-----|
| Logistic Regression | Simple baseline, interpretable coefficients |
| Decision Tree | Non-linear splits, easy to visualize |
| Random Forest | Ensemble of trees, reduces overfitting |
| XGBoost | Gradient boosting, strong on tabular data |

All models use `class_weight='balanced'` or equivalent (`scale_pos_weight` for XGBoost) to handle the 84/16 class imbalance.

---

## Results

All metrics are from the held-out test set. These numbers come directly from `models/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **XGBoost** | 0.970 | 0.931 | 0.877 | 0.903 | **0.993** |
| Random Forest | 0.958 | 0.864 | 0.877 | 0.870 | 0.988 |
| Decision Tree | 0.931 | 0.758 | 0.840 | 0.797 | 0.914 |
| Logistic Regression | 0.877 | 0.582 | 0.831 | 0.684 | 0.943 |

XGBoost performed best across all metrics. The ROC-AUC of 0.99 is high, but I verified there is no data leakage: the Naive Bayes columns are dropped, and SMOTE is applied only after the train-test split. The features in this dataset are simply very predictive of churn.

**Top churn drivers (by feature importance):**
- Total transaction count and amount: customers who stop transacting are the most likely to leave
- Total revolving balance: lower balance correlates with lower engagement
- Credit utilization ratio: similar pattern

---

## Project Structure

```
CustomerChurn/
├── data/
│   ├── raw/                     # Original Kaggle CSV
│   └── processed/               # Cleaned + feature-engineered data
├── src/
│   ├── preprocessing.py         # Encoding, scaling, SMOTE, train-test split
│   ├── feature_engineering.py   # 9 derived features
│   ├── train.py                 # Trains 4 models, selects and saves the best
│   ├── inference.py             # Loads saved model and runs predictions
│   ├── eda.py                   # Exploratory data analysis helpers
│   └── __init__.py
├── app/
│   ├── app.py                   # Streamlit dashboard (prediction + analytics)
│   ├── assets/
│   └── components/
├── models/
│   ├── best_model.pkl           # Trained XGBoost model
│   ├── preprocessor.pkl         # Scaler + label encoders
│   ├── model_metadata.pkl       # Feature names, metrics, training date
│   └── model_comparison.csv     # All model results side by side
├── requirements.txt
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.9+ and pip.

```bash
git clone https://github.com/Chakradhar183/Credit-Card-Churn-Predictor.git
cd Credit-Card-Churn-Predictor

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

**Download the dataset:** Get it from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) and place the CSV in `data/raw/`.

Or use the Kaggle CLI:
```bash
kaggle datasets download -d sakshigoyal7/credit-card-customers -p data/raw/ --unzip
```

---

## Usage

### Train the model
```bash
python src/train.py
```
This preprocesses the data, engineers features, trains all four models, prints a comparison table, and saves the best model to `models/`.

### Launch the dashboard
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501`. The app has two pages:
- **Prediction** - enter customer details and get a churn probability with a gauge chart and risk level
- **Dashboard** - explore the dataset with filters for gender, income, card type, age, and risk level; view distribution charts and a table of the top 50 highest-risk customers

### Use in code
```python
from src.inference import load_model_and_preprocessor, predict_churn

model, preprocessor, metadata = load_model_and_preprocessor()

result = predict_churn({
    'Customer_Age': 45,
    'Gender': 'M',
    'Credit_Limit': 12000,
    'Total_Trans_Amt': 5000,
    'Total_Trans_Ct': 42,
    # ... other features
}, model, preprocessor, metadata)

print(result['churn_probability'])  # e.g. 0.73
print(result['risk_level'])         # "High"
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
