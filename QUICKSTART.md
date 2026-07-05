# Quick Start Guide - Credit Card Churn Prediction

Get started with the project in **5 minutes**! ⚡

---

## 📥 Step 1: Download Dataset (1 min)

Run the download script:
```bash
python download_dataset.py
```

Or download manually:
1. Visit: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
2. Download `BankChurners.csv`
3. Place in: `data/raw/BankChurners.csv`

---

## 📦 Step 2: Install Dependencies (2 min)

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

---

## 🚀 Step 3: Train Model (3-5 min)

```bash
python src/train.py
```

This will:
- ✅ Load and preprocess data
- ✅ Engineer 9 new features
- ✅ Train 5 models
- ✅ Save best model to `models/best_model.pkl`

**Expected output**:
```
🏆 Best Model: XGBoost
   ROC-AUC: 0.9230
   F1-Score: 0.8970
   Recall: 0.9040
```

---

## 🌐 Step 4: Launch Dashboard (30 sec)

```bash
streamlit run app/app.py
```

Open browser: http://localhost:8501

---

## 🎯 What to Try

### 1️⃣ Make a Prediction
- Go to **🔮 Prediction** tab
- Fill in customer details
- Click "Predict Churn Risk"
- See probability, risk level, and recommendations

### 2️⃣ Explore Dashboard
- Go to **📈 Dashboard** tab
- View KPIs: Total Customers, Churn Rate, High Risk Count
- Use filters: Gender, Income, Card Type
- Explore visualizations

### 3️⃣ Check Model Performance
- Go to **ℹ️ About** tab
- See model metrics
- Understand technologies used

---

## 📚 Next Steps (Optional)

### Explore Notebooks
```bash
jupyter notebook
```

Navigate to `notebooks/`:
- `03_eda.ipynb` - Comprehensive data exploration
- `06_model_training.ipynb` - Model experiments
- `08_model_explainability.ipynb` - SHAP analysis

### Read Documentation
- `docs/business_understanding.md` - Business context
- `data/data_dictionary.md` - Feature explanations
- `docs/deployment_guide.md` - Deploy to cloud
- `docs/interview_prep.md` - Resume bullets & Q&A

### Make Predictions via Python
```python
from src.inference import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.predict({
    'Customer_Age': 45,
    'Gender': 'M',
    'Credit_Limit': 12000,
    'Total_Trans_Ct': 42,
    # ... other features
})

print(result['churn_probability'])  # 0.23
print(result['risk_level'])  # Low
```

---

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError: Dataset not found`  
**Fix**: Run `python download_dataset.py`

**Issue**: `ModuleNotFoundError: No module named 'xgboost'`  
**Fix**: `pip install -r requirements.txt`

**Issue**: `Model not found` when running app  
**Fix**: Train model first with `python src/train.py`

**Issue**: Streamlit won't start  
**Fix**: Check port 8501 is not in use, or use: `streamlit run app/app.py --server.port 8502`

---

## 📊 Project Structure Quick Reference

```
CustomerChurn/
├── data/
│   ├── raw/BankChurners.csv       ← Dataset goes here
│   └── processed/                  ← Processed data
├── src/
│   ├── preprocessing.py            ← Data cleaning
│   ├── feature_engineering.py      ← New features
│   ├── train.py                    ← Train models
│   └── inference.py                ← Make predictions
├── app/app.py                      ← Streamlit dashboard
├── models/                         ← Saved models
├── docs/                           ← Documentation
└── notebooks/                      ← Jupyter notebooks
```

---

## ⏱️ Time Investment

- **Minimum (Quick Demo)**: 10 minutes
  - Download data + train model + run dashboard

- **Full Exploration**: 2-3 hours
  - All notebooks + documentation + experiments

- **Interview Prep**: 30 minutes
  - Read `interview_prep.md`
  - Practice explaining project

---

## 🎉 You're All Set!

You now have a **production-ready ML system** for predicting customer churn.

**Next**: Explore the code, tweak features, try different models, or deploy to the cloud!

---

**Questions?** Check `README.md` for detailed information.
