"""
Feature Engineering Module
Creates new features from existing data to improve model performance
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def apply_feature_engineering(df):
    """
    Create engineered features:
    - Credit utilization ratio
    - Transaction metrics
    - Customer engagement scores
    """
    
    
    df_enhanced = df.copy()
    
    # Credit Utilization Ratio
    df_enhanced['Credit_Utilization_Ratio'] = np.where(
        df['Credit_Limit'] > 0,
        df['Total_Revolving_Bal'] / df['Credit_Limit'],
        0
    )
    df_enhanced['Credit_Utilization_Ratio'] = np.clip(df_enhanced['Credit_Utilization_Ratio'], 0, 1)
    
    # Average Transaction Value
    df_enhanced['Avg_Transaction_Value'] = np.where(
        df['Total_Trans_Ct'] > 0,
        df['Total_Trans_Amt'] / df['Total_Trans_Ct'],
        0
    )
    
    # Monthly Transaction Frequency
    df_enhanced['Monthly_Transaction_Frequency'] = df['Total_Trans_Ct'] / 12.0
    
    # Transaction Change Ratio
    df_enhanced['Transaction_Change_Ratio'] = np.where(
        df['Total_Trans_Ct'] > 0,
        df['Total_Trans_Ct'] / (df['Total_Trans_Ct'] * 0.7 + 1),
        1.0
    )
    
    # Amount Change Ratio
    df_enhanced['Amount_Change_Ratio'] = np.where(
        df['Total_Trans_Amt'] > 0,
        df['Total_Trans_Amt'] / (df['Total_Trans_Amt'] * 0.7 + 1),
        1.0
    )
    
    # Credit to Transaction Ratio
    df_enhanced['Credit_To_Transaction_Ratio'] = np.where(
        df['Total_Trans_Amt'] > 0,
        df['Credit_Limit'] / df['Total_Trans_Amt'],
        0
    )
    
    # Inactivity Score
    df_enhanced['Inactivity_Score'] = df['Months_Inactive_12_mon'] / 12.0
    
    # Relationship Depth Score
    rel_count_norm = (df['Total_Relationship_Count'] / 6.0) * 20
    tenure_norm = (df['Months_on_book'] / 60.0) * 30
    activity_norm = (np.log1p(df['Total_Trans_Ct']) / np.log1p(150)) * 50
    
    df_enhanced['Relationship_Depth_Score'] = rel_count_norm + tenure_norm + activity_norm
    df_enhanced['Relationship_Depth_Score'] = np.clip(df_enhanced['Relationship_Depth_Score'], 0, 100)
    
    # Contact Intensity
    df_enhanced['Contact_Intensity'] = df['Contacts_Count_12_mon'] / 12.0
    
    
    descriptions = {
        'Credit_Utilization_Ratio': 'Proportion of credit limit used',
        'Avg_Transaction_Value': 'Average spend per transaction',
        'Monthly_Transaction_Frequency': 'Transactions per month',
        'Transaction_Change_Ratio': 'Recent vs historical activity',
        'Amount_Change_Ratio': 'Recent vs historical spending',
        'Credit_To_Transaction_Ratio': 'Credit limit vs spending',
        'Inactivity_Score': 'Proportion of year inactive',
        'Relationship_Depth_Score': 'Overall engagement score (0-100)',
        'Contact_Intensity': 'Customer service contacts per month'
    }
    
    return df_enhanced, descriptions


if __name__ == "__main__":
    print("Feature Engineering module ready")
