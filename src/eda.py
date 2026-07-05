"""
Exploratory Data Analysis (EDA) Module
Comprehensive data analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_overview(filepath):
    """Load data and display basic information"""
    print("\nDATA OVERVIEW\n")
    
    df = pd.read_csv(filepath)
    print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    
    return df


def check_missing_values(df):
    """Check for missing values"""
    print("\nMISSING VALUES\n")
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Percentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) == 0:
        print("No missing values found")
    else:
        print("Missing values found:")
        print(missing_df.to_string(index=False))
    
    return missing_df


def statistical_summary(df):
    """Statistical summary of numerical and categorical features"""
    print("\nSTATISTICAL SUMMARY\n")
    
    print("Numerical Features:")
    print(df.describe().T)
    
    print("\n\nCategorical Features:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())


def analyze_target(df, target_col='Attrition_Flag'):
    """Analyze target variable distribution"""
    print("\nTARGET VARIABLE ANALYSIS\n")
    
    print(f"Target: {target_col}\n")
    print("Distribution:")
    print(df[target_col].value_counts())
    
    print("\nPercentages:")
    percentages = df[target_col].value_counts(normalize=True) * 100
    for label, pct in percentages.items():
        print(f"  {label}: {pct:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    df[target_col].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Target Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Target Percentage', fontsize=14, fontweight='bold')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Check imbalance
    counts = df[target_col].value_counts()
    ratio = counts.max() / counts.min()
    print(f"\nImbalance Ratio: {ratio:.2f}:1")
    if ratio > 2:
        print("WARNING: Data is imbalanced - SMOTE recommended")


def plot_numerical_distributions(df, target_col='Attrition_Flag'):
    """Plot distributions of numerical features"""
    print("\nNUMERICAL FEATURES DISTRIBUTION\n")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    numerical_cols = [col for col in numerical_cols if 'CLIENTNUM' not in col]
    
    print(f"Plotting {len(numerical_cols)} features\n")
    
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_categorical_features(df, target_col='Attrition_Flag'):
    """Analyze categorical features"""
    print("\nCATEGORICAL FEATURES ANALYSIS\n")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    print(f"Analyzing {len(categorical_cols)} features\n")
    
    for col in categorical_cols:
        print(f"\n{col}:")

        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        
        for value, count in counts.items():
            pct = percentages[value]
            print(f"  {value}: {count} ({pct:.2f}%)")
    
    # Plot
    if len(categorical_cols) > 0:
        n_cols = 2
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            ax = axes[idx]
            df[col].value_counts().plot(kind='bar', ax=ax, color='coral')
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


def correlation_analysis(df, target_col='Attrition_Flag'):
    """Analyze feature correlations"""
    print("\nCORRELATION ANALYSIS\n")
    
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numerical_df.corr()
    
    print("Correlation Matrix:")
    print(correlation)
    
    # Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    if target_col in numerical_df.columns:
        print(f"\n\nTop 10 Features Correlated with {target_col}:")

        target_corr = correlation[target_col].abs().sort_values(ascending=False)
        print(target_corr.head(11))


def detect_outliers(df):
    """Detect outliers using boxplots"""
    print("\nOUTLIER DETECTION\n")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if 'CLIENTNUM' not in col]
    
    print("Using boxplots\n")
    
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        df.boxplot(column=col, ax=ax)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
    
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def bivariate_analysis(df, target_col='Attrition_Flag'):
    """Compare features against target variable"""
    print("\nBIVARIATE ANALYSIS (Features vs Target)\n")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in [target_col, 'CLIENTNUM']]
    
    print("Comparing numerical features across target classes\n")
    
    important_features = numerical_cols[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(important_features):
        ax = axes[idx]
        for label in df[target_col].unique():
            df[df[target_col] == label][col].hist(bins=20, alpha=0.6, label=label, ax=ax)
        ax.set_title(f'{col} by {target_col}', fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def run_complete_eda(filepath):
    """Run complete EDA pipeline"""
    print("\nEXPLORATORY DATA ANALYSIS (EDA)\n")
    
    df = load_and_overview(filepath)
    check_missing_values(df)
    statistical_summary(df)
    analyze_target(df)
    plot_numerical_distributions(df)
    analyze_categorical_features(df)
    correlation_analysis(df)
    detect_outliers(df)
    bivariate_analysis(df)
    
    print("\nEDA COMPLETE")
    print(f"\nTotal Customers: {len(df):,}")
    print(f"Total Features: {df.shape[1]}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print("\nTarget Distribution:")
    for label, count in df['Attrition_Flag'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")

    
    return df


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "BankChurners.csv"
    
    if DATA_PATH.exists():
        df = run_complete_eda(DATA_PATH)
    else:
        print(f"Data file not found: {DATA_PATH}")
