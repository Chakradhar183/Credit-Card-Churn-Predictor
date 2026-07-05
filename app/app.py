"""
Streamlit Web Application for Credit Card Churn Prediction

Features:
- Single customer churn prediction with input form
- Interactive analytics dashboard with dynamic charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import load_model_and_preprocessor, predict_churn, get_feature_importance
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Credit Card Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-medium {
        background-color: #ffc107;
        color: black;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-high {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 40px;
        border: none;
        border-radius: 25px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the trained model and preprocessor"""
    try:
        model, preprocessor, metadata = load_model_and_preprocessor()
        return model, preprocessor, metadata
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found. Please train the model first by running: `python src/train.py`")
        st.stop()


# Load data for dashboard
@st.cache_data
def load_dashboard_data():
    """Load processed data for dashboard visualizations"""
    data_path = Path(__file__).parent.parent / "data" / "processed" / "enhanced_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Encode target if needed
        if 'Attrition_Flag' in df.columns and df['Attrition_Flag'].dtype == 'object':
            df['Churn'] = df['Attrition_Flag'].map({
                'Existing Customer': 0,
                'Attrited Customer': 1
            })
        return df
    return None


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🏦 Credit Card Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Customer Retention Intelligence</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🔮 Prediction", "📊 Dashboard"]
    )
    
    predictor = load_predictor()
    model, preprocessor, metadata = predictor
    
    if page == "🔮 Prediction":
        prediction_page(model, preprocessor, metadata)
    else:
        dashboard_page(model, preprocessor, metadata)


def prediction_page(model, preprocessor, metadata):
    """Customer churn prediction page"""
    
    st.header("🔮 Predict Customer Churn")
    st.markdown("---")
    
    # Information box
    st.info("💡 Fill in the customer details below to predict churn probability and risk level.")
    
    # Create two columns for input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Demographic Info")
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent_count = st.number_input("Number of Dependents", min_value=0, max_value=10, value=3, step=1)
        education_level = st.selectbox(
            "Education Level",
            ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced", "Unknown"]
        )
        income_category = st.selectbox(
            "Income Category",
            ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
        )
    
    with col2:
        st.subheader("💳 Account Info")
        card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
        months_on_book = st.number_input("Months on Book (Tenure)", min_value=1, max_value=100, value=39, step=1)
        total_relationship_count = st.number_input("Total Products", min_value=1, max_value=6, value=5, step=1)
        months_inactive = st.number_input("Months Inactive (Last 12 months)", min_value=0, max_value=12, value=1, step=1)
        contacts_count = st.number_input("Contacts (Last 12 months)", min_value=0, max_value=10, value=3, step=1)
        credit_limit = st.number_input("Credit Limit ($)", min_value=1000, max_value=50000, value=12691, step=100)
    
    with col3:
        st.subheader("💰 Transaction Info")
        total_revolving_bal = st.number_input("Total Revolving Balance ($)", min_value=0, max_value=10000, value=777, step=10)
        avg_open_to_buy = st.number_input("Avg Open To Buy ($)", min_value=0, max_value=50000, value=11914, step=100)
        total_amt_chng = st.number_input("Total Amt Change Q4-Q1", min_value=0.0, max_value=5.0, value=1.335, step=0.001)
        total_trans_amt = st.number_input("Total Transaction Amount ($)", min_value=0, max_value=30000, value=1144, step=10)
        total_trans_ct = st.number_input("Total Transaction Count", min_value=0, max_value=200, value=42, step=1)
        total_ct_chng = st.number_input("Total Count Change Q4-Q1", min_value=0.0, max_value=5.0, value=1.625, step=0.001)
        avg_utilization = st.slider("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.061, step=0.001)
    
    # Predict button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🎯 Predict Churn Risk", width='stretch')
    
    if predict_button:
        # Calculate engineered features
        credit_util_ratio = total_revolving_bal / credit_limit if credit_limit > 0 else 0
        avg_trans_value = total_trans_amt / total_trans_ct if total_trans_ct > 0 else 0
        monthly_trans_freq = total_trans_ct / 12.0
        trans_change_ratio = 1.0  # Simplified
        amt_change_ratio = 1.0  # Simplified
        credit_to_trans_ratio = credit_limit / total_trans_amt if total_trans_amt > 0 else 0
        inactivity_score = months_inactive / 12.0
        relationship_depth = ((total_relationship_count / 6.0) * 20 + (months_on_book / 60.0) * 30 + (np.log1p(total_trans_ct) / np.log1p(150)) * 50)
        contact_intensity = contacts_count / 12.0
        
        # Prepare input data
        input_data = {
            'Customer_Age': customer_age,
            'Gender': gender,
            'Dependent_count': dependent_count,
            'Education_Level': education_level,
            'Marital_Status': marital_status,
            'Income_Category': income_category,
            'Card_Category': card_category,
            'Months_on_book': months_on_book,
            'Total_Relationship_Count': total_relationship_count,
            'Months_Inactive_12_mon': months_inactive,
            'Contacts_Count_12_mon': contacts_count,
            'Credit_Limit': float(credit_limit),
            'Total_Revolving_Bal': total_revolving_bal,
            'Avg_Open_To_Buy': float(avg_open_to_buy),
            'Total_Amt_Chng_Q4_Q1': total_amt_chng,
            'Total_Trans_Amt': total_trans_amt,
            'Total_Trans_Ct': total_trans_ct,
            'Total_Ct_Chng_Q4_Q1': total_ct_chng,
            'Avg_Utilization_Ratio': avg_utilization,
            'Credit_Utilization_Ratio': credit_util_ratio,
            'Avg_Transaction_Value': avg_trans_value,
            'Monthly_Transaction_Frequency': monthly_trans_freq,
            'Transaction_Change_Ratio': trans_change_ratio,
            'Amount_Change_Ratio': amt_change_ratio,
            'Credit_To_Transaction_Ratio': credit_to_trans_ratio,
            'Inactivity_Score': inactivity_score,
            'Relationship_Depth_Score': relationship_depth,
            'Contact_Intensity': contact_intensity
        }
        
        try:
            # Make prediction using functional approach
            result = predict_churn(input_data, model, preprocessor, metadata)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Create gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['churn_probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': result['risk_color']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, width='stretch')
            
            # Risk level badge
            risk_class = f"risk-{result['risk_level'].lower()}"
            st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <div class="{risk_class}">
                        Risk Level: {result['risk_level'].upper()}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
            with col2:
                st.metric("Retention Probability", f"{result['retention_probability']:.1%}")
            with col3:
                st.metric("Prediction", result['prediction_label'])
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommended Actions")
            
            if result['risk_level'] == "High":
                st.error("""
                **🚨 HIGH RISK - Immediate Action Required:**
                - Offer personalized retention incentive (e.g., bonus points, fee waiver)
                - Schedule priority customer service call
                - Review account for service issues
                - Consider credit limit increase if qualified
                """)
            elif result['risk_level'] == "Medium":
                st.warning("""
                **⚠️ MEDIUM RISK - Proactive Engagement Needed:**
                - Send targeted marketing campaign
                - Offer product upgrade or additional services
                - Monitor account activity closely
                - Survey customer satisfaction
                """)
            else:
                st.success("""
                **✅ LOW RISK - Maintain Engagement:**
                - Continue regular communication
                - Promote loyalty program benefits
                - Cross-sell relevant products
                - Thank customer for their business
                """)
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")


def _get_predictions_df(model, preprocessor, metadata):
    """Load data and run batch predictions. Returns (output_df, probabilities) or (None, None)."""
    df = load_dashboard_data()
    if df is None:
        return None, None

    display_cols = [c for c in df.columns if c not in ['Churn']]
    output_df = df[display_cols].copy()
    X_df = df.copy()

    for col in ['CLIENTNUM', 'Attrition_Flag']:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    ordinal_features = ['Education_Level', 'Income_Category']
    for col in ordinal_features:
        if col in X_df.columns and col in preprocessor['label_encoders']:
            le = preprocessor['label_encoders'][col]
            X_df[col] = X_df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    nominal_map = {
        'Gender': {'M': 'Gender_M'},
        'Marital_Status': {'Married': 'Marital_Status_Married',
                           'Single': 'Marital_Status_Single'},
        'Card_Category': {'Gold': 'Card_Category_Gold',
                          'Platinum': 'Card_Category_Platinum',
                          'Silver': 'Card_Category_Silver'},
    }
    expected_features = metadata['feature_names']
    for src_col, mapping in nominal_map.items():
        if src_col in X_df.columns:
            for value, dummy_col in mapping.items():
                if dummy_col in expected_features:
                    X_df[dummy_col] = (X_df[src_col] == value).astype(int)
            X_df = X_df.drop(columns=[src_col])

    for feat in expected_features:
        if feat not in X_df.columns:
            X_df[feat] = 0

    X_df = X_df[expected_features]
    X_scaled = preprocessor['scaler'].transform(X_df)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    output_df['Churn_Probability'] = np.round(probabilities * 100, 2)
    output_df['Risk_Level'] = np.where(
        probabilities < 0.3, 'Low',
        np.where(probabilities <= 0.7, 'Medium', 'High')
    )
    return output_df, probabilities


def dashboard_page(model, preprocessor, metadata):
    """Interactive analytics dashboard with dynamic Plotly charts"""

    st.header("📊 Churn Analytics Dashboard")
    st.markdown("---")

    # --- Load & predict ---
    with st.spinner("🔄 Loading predictions…"):
        output_df, probabilities = _get_predictions_df(model, preprocessor, metadata)

    if output_df is None:
        st.warning("⚠️ Data not available. Run `python src/train.py` first.")
        return

    df = output_df.copy()

    # ───────────── Sidebar Filters ─────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Dashboard Filters")

    # Gender filter
    gender_options = sorted(df['Gender'].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

    # Income filter
    income_options = sorted(df['Income_Category'].dropna().unique().tolist())
    selected_incomes = st.sidebar.multiselect("Income Category", income_options, default=income_options)

    # Card filter
    card_options = sorted(df['Card_Category'].dropna().unique().tolist())
    selected_cards = st.sidebar.multiselect("Card Category", card_options, default=card_options)

    # Risk filter
    risk_options = ['Low', 'Medium', 'High']
    selected_risks = st.sidebar.multiselect("Risk Level", risk_options, default=risk_options)

    # Age range
    age_min, age_max = int(df['Customer_Age'].min()), int(df['Customer_Age'].max())
    selected_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    # Apply filters
    mask = (
        df['Gender'].isin(selected_genders) &
        df['Income_Category'].isin(selected_incomes) &
        df['Card_Category'].isin(selected_cards) &
        df['Risk_Level'].isin(selected_risks) &
        df['Customer_Age'].between(selected_age[0], selected_age[1])
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No customers match the current filters. Adjust the sidebar filters.")
        return

    # ───────────── KPI Row ─────────────
    total = len(fdf)
    avg_prob = fdf['Churn_Probability'].mean()
    high_risk = (fdf['Risk_Level'] == 'High').sum()
    medium_risk = (fdf['Risk_Level'] == 'Medium').sum()
    low_risk = (fdf['Risk_Level'] == 'Low').sum()

    # Actual churn rate (from Attrition_Flag if available)
    if 'Attrition_Flag' in fdf.columns:
        actual_churned = (fdf['Attrition_Flag'] == 'Attrited Customer').sum()
        churn_rate = actual_churned / total * 100
    else:
        churn_rate = avg_prob

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Customers", f"{total:,}")
    k2.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
    k3.metric("🔴 High Risk", f"{high_risk:,}")
    k4.metric("🟡 Medium Risk", f"{medium_risk:,}")
    k5.metric("🟢 Low Risk", f"{low_risk:,}")

    st.markdown("---")

    # ───────────── Row 1: Risk Distribution + Churn Probability Histogram ─────────────
    col_left, col_right = st.columns(2)

    with col_left:
        risk_counts = fdf['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        colors_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        fig_donut = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.5,
            marker=dict(colors=[colors_map[r] for r in risk_counts.index]),
            textinfo='label+percent',
            hovertemplate='%{label}: %{value:,} customers (%{percent})<extra></extra>'
        ))
        fig_donut.update_layout(
            title=dict(text="Risk Level Distribution", font=dict(size=18)),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        fig_hist = px.histogram(
            fdf, x='Churn_Probability', nbins=40,
            color_discrete_sequence=['#667eea'],
            labels={'Churn_Probability': 'Churn Probability (%)'},
        )
        fig_hist.update_layout(
            title=dict(text="Churn Probability Distribution", font=dict(size=18)),
            yaxis_title="Number of Customers",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            bargap=0.05
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ───────────── Row 2: Churn by Card Category + Churn by Income ─────────────
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        card_agg = fdf.groupby('Card_Category')['Churn_Probability'].mean().sort_values(ascending=False).reset_index()
        fig_card = px.bar(
            card_agg, x='Card_Category', y='Churn_Probability',
            color='Churn_Probability',
            color_continuous_scale='RdYlGn_r',
            labels={'Card_Category': 'Card Category', 'Churn_Probability': 'Avg Churn Prob (%)'},
        )
        fig_card.update_layout(
            title=dict(text="Avg Churn Probability by Card Category", font=dict(size=18)),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_card, use_container_width=True)

    with col_right2:
        income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']
        income_agg = fdf.groupby('Income_Category')['Churn_Probability'].mean().reindex(
            [i for i in income_order if i in fdf['Income_Category'].unique()]
        ).reset_index()
        fig_income = px.bar(
            income_agg, x='Income_Category', y='Churn_Probability',
            color='Churn_Probability',
            color_continuous_scale='RdYlGn_r',
            labels={'Income_Category': 'Income Category', 'Churn_Probability': 'Avg Churn Prob (%)'},
        )
        fig_income.update_layout(
            title=dict(text="Avg Churn Probability by Income", font=dict(size=18)),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_income, use_container_width=True)

    # ───────────── Row 3: Gender Split + Age vs Churn Scatter ─────────────
    col_left3, col_right3 = st.columns(2)

    with col_left3:
        gender_agg = fdf.groupby('Gender').agg(
            Avg_Prob=('Churn_Probability', 'mean'),
            Count=('Churn_Probability', 'count')
        ).reset_index()
        fig_gender = px.bar(
            gender_agg, x='Gender', y='Avg_Prob',
            text='Count',
            color='Gender',
            color_discrete_map={'M': '#667eea', 'F': '#e84393'},
            labels={'Avg_Prob': 'Avg Churn Prob (%)', 'Count': 'Customers'},
        )
        fig_gender.update_traces(textposition='outside')
        fig_gender.update_layout(
            title=dict(text="Avg Churn Probability by Gender", font=dict(size=18)),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with col_right3:
        # Sample for performance if large
        scatter_df = fdf.sample(n=min(2000, len(fdf)), random_state=42)
        fig_scatter = px.scatter(
            scatter_df, x='Customer_Age', y='Churn_Probability',
            color='Risk_Level',
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
            opacity=0.6,
            labels={'Customer_Age': 'Age', 'Churn_Probability': 'Churn Prob (%)'},
            hover_data=['Gender', 'Card_Category', 'Income_Category']
        )
        fig_scatter.update_layout(
            title=dict(text="Age vs Churn Probability", font=dict(size=18)),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ───────────── Row 4: Top At-Risk Customers Table ─────────────
    st.markdown("---")
    st.subheader("🔥 Top 50 Highest-Risk Customers")

    top_cols = ['CLIENTNUM', 'Customer_Age', 'Gender', 'Income_Category',
                'Card_Category', 'Credit_Limit', 'Total_Trans_Ct',
                'Months_Inactive_12_mon', 'Churn_Probability', 'Risk_Level']
    available_cols = [c for c in top_cols if c in fdf.columns]
    top_risk = fdf.nlargest(50, 'Churn_Probability')[available_cols].reset_index(drop=True)
    top_risk.index = top_risk.index + 1  # 1-indexed

    st.dataframe(
        top_risk.style.background_gradient(
            subset=['Churn_Probability'], cmap='RdYlGn_r'
        ),
        use_container_width=True,
        height=500
    )

    # ───────────── Download filtered data ─────────────
    st.markdown("---")
    csv_bytes = fdf.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download Filtered Data ({len(fdf):,} rows)",
        data=csv_bytes,
        file_name="filtered_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
