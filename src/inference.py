"""
Inference Module
Makes predictions on new customer data using trained model
"""

from pathlib import Path
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_model_and_preprocessor(model_path=None, preprocessor_path=None):
    """Load trained model and preprocessing tools"""
    
    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "best_model.pkl"
    if preprocessor_path is None:
        preprocessor_path = Path(__file__).parent.parent / "models" / "preprocessor.pkl"
    
    # Load model
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    
    # Load preprocessor
    if not Path(preprocessor_path).exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    # Load metadata
    metadata_path = Path(__file__).parent.parent / "models" / "model_metadata.pkl"
    metadata = None
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)

    
    return model, preprocessor, metadata


def predict_churn(customer_data, model, preprocessor, metadata):
    """
    Predict churn probability for new customer
    
    Returns: dict with prediction, probability, and risk level
    """
    
    # Convert to DataFrame
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()
    
    # Remove unnecessary columns
    cols_to_drop = ['CLIENTNUM', 'Attrition_Flag']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Encode ordinal features
    ordinal_features = ['Education_Level', 'Income_Category']
    for col in ordinal_features:
        if col in df.columns and col in preprocessor['label_encoders']:
            le = preprocessor['label_encoders'][col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    # Remove nominal features
    nominal_features = ['Gender', 'Marital_Status', 'Card_Category']
    for col in nominal_features:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Add expected features
    expected_features = metadata['feature_names']
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Set dummy columns based on input
    if isinstance(customer_data, dict):
        if 'Gender' in customer_data and customer_data['Gender'] == 'M':
            if 'Gender_M' in expected_features:
                df['Gender_M'] = 1
        
        if 'Marital_Status' in customer_data:
            status = customer_data['Marital_Status']
            if status == 'Married' and 'Marital_Status_Married' in expected_features:
                df['Marital_Status_Married'] = 1
            elif status == 'Single' and 'Marital_Status_Single' in expected_features:
                df['Marital_Status_Single'] = 1
        
        if 'Card_Category' in customer_data:
            card = customer_data['Card_Category']
            if card == 'Gold' and 'Card_Category_Gold' in expected_features:
                df['Card_Category_Gold'] = 1
            elif card == 'Platinum' and 'Card_Category_Platinum' in expected_features:
                df['Card_Category_Platinum'] = 1
            elif card == 'Silver' and 'Card_Category_Silver' in expected_features:
                df['Card_Category_Silver'] = 1
    
    # Reorder columns
    df = df[expected_features]
    
    # Scale features
    X = preprocessor['scaler'].transform(df)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    # Determine risk level
    if probability < 0.3:
        risk_level, risk_color = "Low", "green"
    elif probability < 0.7:
        risk_level, risk_color = "Medium", "orange"
    else:
        risk_level, risk_color = "High", "red"
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Churned' if prediction == 1 else 'Retained',
        'churn_probability': float(probability),
        'retention_probability': float(1 - probability),
        'risk_level': risk_level,
        'risk_color': risk_color
    }
    
    return result


def get_feature_importance(model_obj=None, top_n=10):
    """
    Get feature importance from model
    
    Returns: DataFrame with feature names and importance scores
    """
    if model_obj is None:
        return None
        
    if hasattr(model_obj, 'feature_importances_'):
        # Load metadata to get feature names
        metadata_path = Path(__file__).parent.parent / "models" / "model_metadata.pkl"
        if metadata_path.exists():
            import joblib
            metadata = joblib.load(metadata_path)
            feature_names = metadata['feature_names']
            
            importances = model_obj.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
    
    return None


