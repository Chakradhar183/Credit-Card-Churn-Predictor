"""
Model Training Module
Trains multiple ML models and selects the best performer
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).parent))
from preprocessing import preprocess_pipeline
from feature_engineering import apply_feature_engineering

MODELS_DIR = Path(__file__).parent.parent / "models"


def save_model(model, filename="best_model.pkl"):
    """Save trained model using joblib"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / filename)


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """Calculate model performance metrics"""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    return metrics


def print_evaluation_report(metrics_dict):
    """Print model metrics"""
    print(f"\n  {metrics_dict['model']} - Performance")
    print(f"  Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall:    {metrics_dict['recall']:.4f}")
    print(f"  F1-Score:  {metrics_dict['f1_score']:.4f}")
    if 'roc_auc' in metrics_dict:
        print(f"  ROC-AUC:   {metrics_dict['roc_auc']:.4f}")


def create_comparison_table(results_list):
    """Create comparison DataFrame from list of metrics"""
    df = pd.DataFrame(results_list)
    return df.sort_values('roc_auc', ascending=False)


def train_all_models(X_train, X_test, y_train, y_test):
    """Train and compare 4 ML models"""
    
    print("\nTRAINING MODELS\n")
    
    results = []
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    lr_metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
    print_evaluation_report(lr_metrics)
    
    models['Logistic Regression'] = lr_model
    results.append(lr_metrics)
    
    # Decision Tree
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42, class_weight='balanced')
    dt_model.fit(X_train, y_train)
    
    y_pred = dt_model.predict(X_test)
    y_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    
    dt_metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Decision Tree")
    print_evaluation_report(dt_metrics)
    
    models['Decision Tree'] = dt_model
    results.append(dt_metrics)
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    print_evaluation_report(rf_metrics)
    
    models['Random Forest'] = rf_model
    results.append(rf_metrics)
    
    # XGBoost
    print("Training XGBoost...")
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', scale_pos_weight=scale_weight
    )
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_metrics = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
    print_evaluation_report(xgb_metrics)
    
    models['XGBoost'] = xgb_model
    results.append(xgb_metrics)
    
    # Compare models
    comparison_df = create_comparison_table(results)
    
    print("\nMODEL COMPARISON\n")
    print(comparison_df.to_string(index=False))
    
    # Select best model
    best_result = max(results, key=lambda x: x['roc_auc'])
    best_name = best_result['model']
    best_model = models[best_name]
    
    print(f"Best Model: {best_name}")
    print(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
    
    return comparison_df, best_model, best_name, best_result


def main():
    """Main training pipeline"""
    
    print("\nCREDIT CARD CHURN PREDICTION - TRAINING\n")
    
    DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "BankChurners.csv"
    
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Feature engineering
    df_enhanced, feature_descriptions = apply_feature_engineering(df)
    
    # Save enhanced data
    processed_path = DATA_PATH.parent.parent / "processed" / "enhanced_data.csv"
    processed_path.parent.mkdir(exist_ok=True)
    df_enhanced.to_csv(processed_path, index=False)
    
    # Preprocessing
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_pipeline(df_enhanced, apply_smote=True)
    
    # Train models
    comparison_df, best_model, best_name, best_metrics = train_all_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_path = Path(__file__).parent.parent / "models" / "model_comparison.csv"
    results_path.parent.mkdir(exist_ok=True)
    comparison_df.to_csv(results_path, index=False)
    
    # Save best model
    model_path = Path(__file__).parent.parent / "models" / "best_model.pkl"
    save_model(best_model, model_path.name)
    
    # Save preprocessor
    preprocessor_path = Path(__file__).parent.parent / "models" / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_name,
        'metrics': best_metrics,
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    metadata_path = Path(__file__).parent.parent / "models" / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    
    print("\nTRAINING COMPLETE")
    print(f"\nBest Model: {best_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
