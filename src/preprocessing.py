"""
Data Preprocessing Module
Handles cleaning, encoding, scaling, imbalance handling and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def preprocess_pipeline(df, target_col='Attrition_Flag',
                        test_size=0.2,
                        apply_smote=True,
                        random_state=42):

    # Separate Features & Target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Drop Unnecessary Columns
    # Drop ID
    X = X.drop(columns=['CLIENTNUM'], errors='ignore')

    # Drop any Naive_Bayes probability columns
    X = X.loc[:, ~X.columns.str.startswith('Naive_Bayes')]

    # Identify Column Types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Handle Missing Values
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Encode Categorical Features

    # Ordinal features
    ordinal_features = ['Education_Level', 'Income_Category']
    label_encoders = {}

    for col in ordinal_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Nominal features (One-Hot Encoding)
    nominal_features = [col for col in categorical_cols if col not in ordinal_features]
    if nominal_features:
        X = pd.get_dummies(X, columns=nominal_features, drop_first=True)

    # Encode Target
    if y.dtype == 'object':
        y = y.map({'Existing Customer': 0,
                   'Attrited Customer': 1}).values
    else:
        y = y.values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle Class Imbalance (SMOTE)
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    # Save Preprocessing Objects
    preprocessor = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': X.columns.tolist()
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), preprocessor


def load_and_preprocess(filepath, apply_smote=True):
    """
    Load dataset and apply preprocessing pipeline
    """
    df = pd.read_csv(filepath)

    return preprocess_pipeline(df, apply_smote=apply_smote)


if __name__ == "__main__":
    print("Preprocessing module ready")