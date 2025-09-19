import pandas as pd
import joblib
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import time
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import altair as alt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="MyHeartRisk - AI Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# MyHeartRisk\nAdvanced AI-powered heart disease risk assessment tool."
    }
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 2rem;
    }
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9Imd3LWdyYWRpZW50IiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIwJSI+PHN0b3Agc3RvcC1jb2xvcj0id2hpdGUiIHN0b3Atb3BhY2l0eT0iMC4yIiBvZmZzZXQ9IjAlIiAvPjxzdG9wIHN0b3AtY29sb3I9IndoaXRlIiBzdG9wLW9wYWNpdHk9IjAiIG9mZnNldD0iMTAwJSIgLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48cGF0aCBkPSJNMjUsMzAgUTUwLDVyIDc1LDMwIFQxMjUsNTAgVDg1LDgwIFQ1MCwxMjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0idXJsKCNndy1ncmFkaWVudCkiIHN0cm9rZS13aWR0aD0iMiIgLz48L3N2Zz4=');
        opacity: 0.15;
        z-index: 0;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        font-weight: 500;
        opacity: 0.95;
        margin-bottom: 1.25rem;
        position: relative;
        z-index: 1;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.75rem;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-left: 4px solid #6366F1;
        margin: 1.25rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #EF4444, #F87171);
        color: white;
        border-left: 4px solid #B91C1C;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #F59E0B, #FBBF24);
        color: white;
        border-left: 4px solid #B45309;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
        border-left: 4px solid #047857;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.85rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(99, 102, 241, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.2);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #F9FAFB;
    }
    
    section[data-testid="stSidebar"] .css-ng1t4o {
        background-color: #F9FAFB !important;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        border-radius: 100px;
    }
    
    /* Slider Styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    
    .stSlider [data-baseweb="thumb"] {
        background: #6366F1 !important;
        border-color: #6366F1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.25);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.08); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .slide-in-right {
        animation: slideInRight 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Mobile Responsive */
    @media (max-width: 991px) {
        .main-title {
            font-size: 2.5rem;
        }
        .main-subtitle {
            font-size: 1.2rem;
        }
        .main-header {
            padding: 2rem 1.5rem;
        }
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .main-subtitle {
            font-size: 1rem;
        }
        .metric-card {
            padding: 1.25rem;
            margin: 0.75rem 0;
        }
        .main-header {
            padding: 1.5rem 1rem;
        }
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: white;
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #6366F1;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transform: translateY(-5px);
    }
    
    .recommendation-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #1F2937;
    }
    
    .recommendation-content {
        color: #4B5563;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Feature Info Cards */
    .feature-info {
        padding: 1rem;
        border-radius: 12px;
        background: #F9FAFB;
        margin: 0.5rem 0;
        border-left: 3px solid #6366F1;
    }
    
    /* Heart Age Display */
    .heart-age-display {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border-radius: 50%;
        width: 180px;
        height: 180px;
        margin: 0 auto;
        box-shadow: 0 15px 30px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .heart-age-value {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .heart-age-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1F2937;
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Custom Loader */
    .custom-loader {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    
    .loader-icon {
        width: 50px;
        height: 50px;
        border: 3px solid #F3F4F6;
        border-radius: 50%;
        border-top: 3px solid #6366F1;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    .loader-text {
        color: #6366F1;
        font-weight: 500;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Select Box Styling */
    div[data-baseweb="select"] {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# UTILITY FUNCTIONS
# ===========================
def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def create_gauge_chart(value, title, color_scheme="blues"):
    """Create a beautiful gauge chart for confidence levels"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': '#2c3e50', 'family': 'Poppins'}},
        delta = {'reference': 50, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#4ecdc4"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#34495e"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#34495e",
            'steps': [
                {'range': [0, 25], 'color': '#4ecdc4'},
                {'range': [25, 50], 'color': '#f39c12'},
                {'range': [50, 75], 'color': '#e67e22'},
                {'range': [75, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#2c3e50", 'family': "Poppins"},
        height = 300,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    return fig

def create_feature_importance_chart(features, values):
    """Create an interactive feature importance chart"""
    df = pd.DataFrame({
        'feature_col': features,
        'importance_col': values
    }).sort_values('importance_col', ascending=False)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('importance_col:Q', title='Importance Score'),
        y=alt.Y('feature_col:N', sort='-x', title=None),
        color=alt.Color('importance_col:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['feature_col', 'importance_col']
    ).properties(
        title='Feature Importance in Prediction',
        height=400
    ).interactive()
    
    return chart

def create_radar_chart(user_data, avg_healthy, avg_risk):
    """Create radar chart comparing user data with healthy and risk averages"""
    categories = list(user_data.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[user_data[cat] for cat in categories],
        theta=categories,
        fill='toself',
        name='Your Data',
        line=dict(color='#6366F1', width=2),
        fillcolor='rgba(99, 102, 241, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[avg_healthy[cat] for cat in categories],
        theta=categories,
        fill='toself',
        name='Healthy Average',
        line=dict(color='#10B981', width=2),
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[avg_risk[cat] for cat in categories],
        theta=categories,
        fill='toself',
        name='At Risk Average',
        line=dict(color='#EF4444', width=2),
        fillcolor='rgba(239, 68, 68, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.5]
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins"),
        height=500,
        margin=dict(t=10, b=80)
    )
    
    return fig

def calculate_heart_age(user_data, model, features):
    """Calculate the heart age based on risk factors"""
    # Base calculation on chronological age and risk factors
    chronological_age = user_data['Age']
    risk_score = model.predict_proba(pd.DataFrame([user_data]))[0][1]
    
    # Adjust heart age based on risk score
    if risk_score > 0.7:
        heart_age = chronological_age + 15
    elif risk_score > 0.5:
        heart_age = chronological_age + 10
    elif risk_score > 0.3:
        heart_age = chronological_age + 5
    elif risk_score < 0.2:
        heart_age = chronological_age - 5
    else:
        heart_age = chronological_age
    
    # Cap heart age
    heart_age = max(20, min(heart_age, 90))
    
    return int(heart_age)

def get_recommendations(user_data, prediction_probability):
    """Generate personalized recommendations based on user data and prediction"""
    recommendations = []
    
    # Blood pressure recommendations
    if 'Systolic BP' in user_data and user_data['Systolic BP'] > 130:
        recommendations.append({
            "category": "Blood Pressure", 
            "advice": "Your systolic blood pressure is elevated. Consider reducing sodium intake, regular exercise, and stress management techniques.",
            "icon": "🩺"
        })
    
    # Cholesterol recommendations
    if 'Total Cholesterol' in user_data and user_data['Total Cholesterol'] > 200:
        recommendations.append({
            "category": "Cholesterol", 
            "advice": "Your cholesterol levels are above optimal range. Focus on a diet rich in fruits, vegetables, whole grains, and lean proteins.",
            "icon": "🍎"
        })
    
    # Age-related recommendations
    if 'Age' in user_data and user_data['Age'] > 50:
        recommendations.append({
            "category": "Age Management", 
            "advice": "Regular cardiovascular screenings recommended. Consider discussing aspirin therapy with your healthcare provider.",
            "icon": "🧬"
        })
    
    # Add general recommendations based on risk level
    if prediction_probability > 0.7:
        recommendations.append({
            "category": "High Risk Alert", 
            "advice": "Your risk profile indicates significant concern. Please consult a healthcare provider promptly for a comprehensive evaluation.",
            "icon": "⚠️"
        })
    elif prediction_probability > 0.3:
        recommendations.append({
            "category": "Moderate Risk", 
            "advice": "Consider lifestyle modifications and regular check-ups to monitor your cardiovascular health.",
            "icon": "⚖️"
        })
    else:
        recommendations.append({
            "category": "Prevention", 
            "advice": "Maintain your healthy lifestyle with regular exercise, balanced diet, and routine health screenings.",
            "icon": "✅"
        })
    
    return recommendations

# ===========================
# MAIN APPLICATION
# ===========================
def main():
    # Load custom CSS
    load_custom_css()
    
    # File paths setup
    mainpath = os.path.dirname(__file__)
    
    # Load Lottie animations
    heart_animation = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_5tl1xxnz.json")
    analysis_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_3rqwsqnj.json")

    # Header Section with Animation
    st.markdown("""
    <div class="main-header fade-in">
        <div class="main-title">❤️ MyHeartRisk</div>
        <div class="main-subtitle">AI-Powered Coronary Heart Disease Risk Assessment</div>
        <p style="font-size: 1.1rem; margin-top: 1.25rem; opacity: 0.9;">
            Advanced machine learning technology to predict cardiovascular risk based on clinical parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load dataset
    try:
        data_path = os.path.join(mainpath, r'Data_health1.xlsx')
        df = pd.read_excel(data_path)
        X = df.drop(columns=['Target'])
        y = df['Target']
        features = X.columns
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
    
    # Calculate average values for healthy and at-risk populations
    healthy_avg = X[y == 0].mean().to_dict()
    risk_avg = X[y == 1].mean().to_dict()
    
    # Create modern navigation
    selected_tab = option_menu(
        menu_title=None,
        options=["Risk Assessment", "Health Dashboard", "Model Insights", "Education", "About"],
        icons=["heart-pulse-fill", "clipboard-data", "graph-up", "book", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "margin-bottom": "2rem"},
            "icon": {"font-size": "1.1rem"},
            "nav-link": {"font-size": "1rem", "text-align": "center", "padding": "1rem", "border-radius": "10px", "margin-right": "0.5rem"},
            "nav-link-selected": {"background-color": "#6366F1", "font-weight": "600"},
        }
    )
    
    # Enhanced Sidebar with better styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #6366F1, #8B5CF6); border-radius: 16px; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0; font-weight: 600;">🏥 Health Parameters</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.95rem;">Enter your health data below</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_data = {}
        
        # Create input form with better organization
        with st.form("health_form"):
            # Age and demographic section
            st.markdown("<h3 style='font-size: 1.2rem; color: #1F2937;'>👤 Demographics</h3>", unsafe_allow_html=True)
            for col in features:
                if col.lower() in ['age', 'gender', 'ethnicity', 'race']:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        min_val = float(X[col].min())
                        max_val = float(X[col].max())
                        mean_val = float(X[col].mean())
                        
                        # Create help text for better UX
                        help_text = f"Normal range: {min_val:.1f} - {max_val:.1f}"
                        
                        if col.lower() == "age":
                            user_data[col] = st.slider(
                                f"🎂 {col.title()}", 
                                int(min_val), 
                                int(max_val), 
                                int(round(mean_val)),
                                help=help_text
                            )
                        else:
                            user_data[col] = st.slider(
                                f"📊 {col.title()}", 
                                min_val, 
                                max_val, 
                                mean_val,
                                help=help_text
                            )
                    else:
                        unique_vals = X[col].dropna().unique()
                        user_data[col] = st.selectbox(
                            f"👥 {col.title()}", 
                            unique_vals,
                            help=f"Select from available options"
                        )
            
            # Vital stats section
            st.markdown("<h3 style='font-size: 1.2rem; color: #1F2937;'>📊 Vital Statistics</h3>", unsafe_allow_html=True)
            vital_cols = [col for col in features if col.lower() in ['systolic bp', 'diastolic bp', 'heart rate', 'respiratory rate', 'temperature']]
            for col in vital_cols:
                if pd.api.types.is_numeric_dtype(X[col]):
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    mean_val = float(X[col].mean())
                    
                    user_data[col] = st.slider(
                        f"🩺 {col.title()}", 
                        min_val, 
                        max_val, 
                        mean_val,
                        help=f"Normal range: {min_val:.1f} - {max_val:.1f}"
                    )
                else:
                    unique_vals = X[col].dropna().unique()
                    user_data[col] = st.selectbox(
                        f"🩺 {col.title()}", 
                        unique_vals
                    )
            
            # Lab values section
            st.markdown("<h3 style='font-size: 1.2rem; color: #1F2937;'>🧪 Laboratory Values</h3>", unsafe_allow_html=True)
            lab_cols = [col for col in features if col not in vital_cols and col.lower() not in ['age', 'gender', 'ethnicity', 'race']]
            for col in lab_cols:
                if pd.api.types.is_numeric_dtype(X[col]):
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    mean_val = float(X[col].mean())
                    
                    user_data[col] = st.slider(
                        f"🧬 {col.title()}", 
                        min_val, 
                        max_val, 
                        mean_val,
                        help=f"Normal range: {min_val:.1f} - {max_val:.1f}"
                    )
                else:
                    unique_vals = X[col].dropna().unique()
                    user_data[col] = st.selectbox(
                        f"🧬 {col.title()}", 
                        unique_vals
                    )
            
            # Enhanced predict button
            predict_btn = st.form_submit_button(
                "🔍 Analyze Risk", 
                use_container_width=True
            )
        
        # Add info section
        st.markdown("""
        <div style="padding: 1.25rem; background: linear-gradient(135deg, #60A5FA, #3B82F6); border-radius: 16px; margin-top: 1.5rem;">
            <h4 style="color: white; margin: 0 0 0.75rem 0; font-size: 1.1rem; font-weight: 600;">ℹ️ How it works</h4>
            <p style="color: white; margin: 0; font-size: 0.95rem; line-height: 1.5; opacity: 0.95;">
                Our AI model analyzes your health parameters using advanced machine learning algorithms trained on real hospital data to predict coronary heart disease risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Convert input into DataFrame
    input_df = pd.DataFrame([user_data])

    # Load model
    try:
        model_path = os.path.join(mainpath, r'model.joblib')
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Risk Assessment Tab
    if selected_tab == "Risk Assessment":
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("<h2 style='font-size: 1.75rem; color: #1F2937;'>Your Heart Disease Risk Assessment</h2>", unsafe_allow_html=True)
            
            # Animated results section with Lottie
            if predict_btn:
                with st.spinner("Analyzing your health parameters..."):
                    # Add a slight delay for effect
                    time.sleep(1.5)
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
                    if prediction == 1:
                        risk_probability = probabilities[1] * 100
                        
                        if risk_probability > 75:
                            risk_class = "risk-high"
                            risk_level = "High Risk"
                            risk_icon = "⚠️"
                        elif risk_probability > 50:
                            risk_class = "risk-medium"
                            risk_level = "Moderate Risk"
                            risk_icon = "⚠️"
                        else:
                            risk_class = "risk-medium"
                            risk_level = "Low-Moderate Risk"
                            risk_icon = "⚠️"
                            
                        st.markdown(f"""
                        <div class="metric-card {risk_class} fade-in">
                            <h2 style="font-size: 1.5rem; margin: 0;">{risk_icon} {risk_level}</h2>
                            <p style="font-size: 1.1rem; margin: 0.5rem 0 1rem 0;">
                                Your parameters indicate a {risk_probability:.1f}% likelihood of coronary heart disease.
                            </p>
                            <p style="font-size: 0.95rem; opacity: 0.9; margin: 0;">
                                Please consult with a healthcare professional for a comprehensive evaluation.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        healthy_probability = probabilities[0] * 100
                        
                        st.markdown(f"""
                        <div class="metric-card risk-low fade-in">
                            <h2 style="font-size: 1.5rem; margin: 0;">✅ Low Risk</h2>
                            <p style="font-size: 1.1rem; margin: 0.5rem 0 1rem 0;">
                                Your parameters indicate a {healthy_probability:.1f}% likelihood of being in good cardiovascular health.
                            </p>
                            <p style="font-size: 0.95rem; opacity: 0.9; margin: 0;">
                                Continue maintaining a healthy lifestyle with regular check-ups.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate heart age
                    heart_age = calculate_heart_age(user_data, model, features)
                    age_difference = heart_age - user_data['Age']
                    
                    st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937; margin-top: 2rem;'>Your Heart Age</h3>", unsafe_allow_html=True)
                    
                    col_age1, col_age2 = st.columns([1, 2])
                    with col_age1:
                        st.markdown(f"""
                        <div class="heart-age-display pulse">
                            <div class="heart-age-value">{heart_age}</div>
                            <div class="heart-age-label">Heart Age</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_age2:
                        st.markdown("<div style='padding: 1rem 0;'>", unsafe_allow_html=True)
                        if age_difference > 5:
                            st.markdown(f"""
                            <div style="background: #FEF2F2; border-left: 4px solid #EF4444; padding: 1rem; border-radius: 8px;">
                                <p style="margin: 0; color: #B91C1C; font-weight: 500;">⚠️ Your heart age is {age_difference} years older than your actual age.</p>
                                <p style="margin: 0.5rem 0 0 0; color: #7F1D1D; font-size: 0.9rem;">This suggests your cardiovascular system may be aging faster than expected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif age_difference > 0:
                            st.markdown(f"""
                            <div style="background: #FFF7ED; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 8px;">
                                <p style="margin: 0; color: #B45309; font-weight: 500;">⚠️ Your heart age is {age_difference} years older than your actual age.</p>
                                <p style="margin: 0.5rem 0 0 0; color: #92400E; font-size: 0.9rem;">This suggests some mild cardiovascular aging.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: #ECFDF5; border-left: 4px solid #10B981; padding: 1rem; border-radius: 8px;">
                                <p style="margin: 0; color: #047857; font-weight: 500;">✅ Your heart age is {abs(age_difference)} years younger than your actual age.</p>
                                <p style="margin: 0.5rem 0 0 0; color: #064E3B; font-size: 0.9rem;">This suggests excellent cardiovascular health!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Personalized recommendations
                    st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937; margin-top: 1.5rem;'>Personalized Recommendations</h3>", unsafe_allow_html=True)
                    
                    recommendations = get_recommendations(user_data, probabilities[1])
                    
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation-card slide-in-right">
                            <div class="recommendation-title">
                                <span>{rec['icon']}</span>
                                <span>{rec['category']}</span>
                            </div>
                            <div class="recommendation-content">
                                {rec['advice']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Lottie animation when no prediction yet
                st_lottie(heart_animation, height=300, key="heart_anim")
                
                st.markdown("""
                <div style="text-align: center; padding: 1rem; margin-top: 1rem;">
                    <h3 style="color: #4B5563; font-size: 1.3rem; font-weight: 500;">Enter your health parameters in the sidebar</h3>
                    <p style="color: #6B7280; font-size: 1rem;">Click "Analyze Risk" to get your personalized heart health assessment</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if predict_btn:
                # Show confidence gauge
                st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937;'>Risk Assessment Confidence</h3>", unsafe_allow_html=True)
                
                if prediction == 1:
                    risk_probability = probabilities[1] * 100
                    confidence_chart = create_gauge_chart(risk_probability, "Risk Probability")
                else:
                    healthy_probability = probabilities[0] * 100
                    confidence_chart = create_gauge_chart(healthy_probability, "Health Confidence")
                
                st.plotly_chart(confidence_chart, use_container_width=True)
                
                # Key Risk Factors
                st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937; margin-top: 1rem;'>Your Key Risk Factors</h3>", unsafe_allow_html=True)
                
                # Calculate feature importance for this prediction
                try:
                    # Get feature importance
                    feature_importance = np.abs(model.coef_[0])
                    top_features_idx = feature_importance.argsort()[-5:][::-1]
                    top_features = [features[i] for i in top_features_idx]
                    top_importance = [feature_importance[i] for i in top_features_idx]
                    
                    # Normalize to 0-100 scale
                    top_importance = [float(i)/sum(feature_importance)*100 for i in top_importance]
                    
                    # Display importance
                    feature_chart = create_feature_importance_chart(top_features, top_importance)
                    st.altair_chart(feature_chart, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Couldn't calculate feature importance: {e}")
                    
    # Health Dashboard Tab            
    elif selected_tab == "Health Dashboard":
        st.markdown("<h2 style='font-size: 1.75rem; color: #1F2937;'>Your Health Dashboard</h2>", unsafe_allow_html=True)
        
        if not 'Age' in user_data:
            st.warning("Please enter your health parameters in the sidebar to view your personalized dashboard.")
        else:
            # Create comparison charts
            st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937;'>Health Parameters Comparison</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #4B5563;'>See how your values compare to healthy individuals and those at risk</p>", unsafe_allow_html=True)
            
            # Select numerical features only
            numerical_features = {}
            for col in features:
                if pd.api.types.is_numeric_dtype(X[col]):
                    numerical_features[col] = user_data[col]
            
            # Create normalized versions for radar chart
            def normalize_dict(d, X):
                result = {}
                for k, v in d.items():
                    if pd.api.types.is_numeric_dtype(X[k]):
                        min_val = X[k].min()
                        max_val = X[k].max()
                        if max_val > min_val:
                            result[k] = (v - min_val) / (max_val - min_val)
                        else:
                            result[k] = 0.5
                return result
            
            # Take only top 8 features for radar to avoid overcrowding
            top_features = list(features)[:8]
            user_data_normalized = normalize_dict({k: user_data[k] for k in top_features if k in user_data}, X)
            healthy_avg_normalized = normalize_dict({k: healthy_avg[k] for k in top_features if k in healthy_avg}, X)
            risk_avg_normalized = normalize_dict({k: risk_avg[k] for k in top_features if k in risk_avg}, X)
            
            radar_chart = create_radar_chart(user_data_normalized, healthy_avg_normalized, risk_avg_normalized)
            st.plotly_chart(radar_chart, use_container_width=True)
            
            # Create Health Metrics
            st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937; margin-top: 1rem;'>Your Health Metrics</h3>", unsafe_allow_html=True)
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            # Create function to determine color based on value
            def get_metric_color(value, normal_range, reverse=False):
                min_val, max_val = normal_range
                if reverse:
                    if value > max_val:
                        return "#EF4444"  # Red
                    elif value < min_val:
                        return "#10B981"  # Green
                    else:
                        return "#F59E0B"  # Yellow
                else:
                    if value < min_val:
                        return "#EF4444"  # Red
                    elif value > max_val:
                        return "#EF4444"  # Red
                    else:
                        return "#10B981"  # Green
            
            # Define normal ranges for common health metrics
            normal_ranges = {
                "Systolic BP": (90, 120),
                "Diastolic BP": (60, 80),
                "Heart Rate": (60, 100),
                "Total Cholesterol": (125, 200),
                "LDL": (0, 100),
                "HDL": (40, 60),
                "Triglycerides": (0, 150),
                "Glucose": (70, 99)
            }
            
            # Display metrics in columns
            metric_keys = list(normal_ranges.keys())
            for i, key in enumerate(metric_keys):
                if key in user_data:
                    col = [metrics_col1, metrics_col2, metrics_col3][i % 3]
                    with col:
                        value = user_data[key]
                        is_reverse = key == "HDL"  # HDL is "good cholesterol", so higher is better
                        color = get_metric_color(value, normal_ranges[key], reverse=is_reverse)
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 4px solid {color};">
                            <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">{key}</p>
                            <h3 style="color: #1F2937; margin: 0.25rem 0; font-size: 1.5rem; font-weight: 600;">{value}</h3>
                            <p style="color: #6B7280; margin: 0; font-size: 0.8rem;">Normal: {normal_ranges[key][0]} - {normal_ranges[key][1]}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Model Insights Tab
    elif selected_tab == "Model Insights":
        st.markdown("<h2 style='font-size: 1.75rem; color: #1F2937;'>AI Model Performance & Insights</h2>", unsafe_allow_html=True)
        
        insight_col1, insight_col2 = st.columns([1, 1])
        
        with insight_col1:
            # Display model performance with upgraded UI
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <h3 style="font-size: 1.4rem; color: #1F2937; margin: 0 0 1rem 0;">Model Performance</h3>
                <p style="color: #4B5563; margin-bottom: 1.5rem;">
                    Our model uses advanced machine learning algorithms to predict heart disease risk with high accuracy.
                </p>
                
                <div style="margin-bottom: 1rem;">
                    <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Accuracy</p>
                    <div style="height: 8px; width: 100%; background: #E5E7EB; border-radius: 4px; margin: 0.5rem 0;">
                        <div style="height: 8px; width: 94%; background: linear-gradient(90deg, #6366F1, #8B5CF6); border-radius: 4px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #6B7280; font-size: 0.8rem;">0%</span>
                        <span style="color: #4B5563; font-weight: 500; font-size: 0.9rem;">94%</span>
                        <span style="color: #6B7280; font-size: 0.8rem;">100%</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Sensitivity</p>
                    <div style="height: 8px; width: 100%; background: #E5E7EB; border-radius: 4px; margin: 0.5rem 0;">
                        <div style="height: 8px; width: 92%; background: linear-gradient(90deg, #6366F1, #8B5CF6); border-radius: 4px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #6B7280; font-size: 0.8rem;">0%</span>
                        <span style="color: #4B5563; font-weight: 500; font-size: 0.9rem;">92%</span>
                        <span style="color: #6B7280; font-size: 0.8rem;">100%</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Specificity</p>
                    <div style="height: 8px; width: 100%; background: #E5E7EB; border-radius: 4px; margin: 0.5rem 0;">
                        <div style="height: 8px; width: 95%; background: linear-gradient(90deg, #6366F1, #8B5CF6); border-radius: 4px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #6B7280; font-size: 0.8rem;">0%</span>
                        <span style="color: #4B5563; font-weight: 500; font-size: 0.9rem;">95%</span>
                        <span style="color: #6B7280; font-size: 0.8rem;">100%</span>
                    </div>
                </div>
                
                <div>
                    <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">MCC (Matthews Correlation Coefficient)</p>
                    <div style="height: 8px; width: 100%; background: #E5E7EB; border-radius: 4px; margin: 0.5rem 0;">
                        <div style="height: 8px; width: 88%; background: linear-gradient(90deg, #6366F1, #8B5CF6); border-radius: 4px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #6B7280; font-size: 0.8rem;">0</span>
                        <span style="color: #4B5563; font-weight: 500; font-size: 0.9rem;">0.88</span>
                        <span style="color: #6B7280; font-size: 0.8rem;">1</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add PCA visualization if available
            if os.path.exists(os.path.join(mainpath, r'PCA.jpg')):
                st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937; margin-top: 1.5rem;'>Data Distribution</h3>", unsafe_allow_html=True)
                st.image(pca, caption='PCA Plot Showing Distribution of Cases and Controls (Cumulative Variance: 37%)')
        
        with insight_col2:
            # Model comparison chart if available
            if os.path.exists(os.path.join(mainpath, r'Comparison.jpg')):
                st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937;'>Model Comparison</h3>", unsafe_allow_html=True)
                st.image(compare, caption='Model Performance Comparison (10 Models with 5-Fold Stratified CV)')
            
            # Feature contribution to prediction
            st.markdown("<h3 style='font-size: 1.4rem; color: #1F2937; margin-top: 1.5rem;'>Feature Contribution Analysis</h3>", unsafe_allow_html=True)
            
            feature_tabs = st.tabs(["Protective Factors", "Risk Factors"])
            
            with feature_tabs[0]:
                if os.path.exists(os.path.join(mainpath, r'neg.jpg')):
                    st.image(neg, caption='Features Contributing to Lower Risk (Protective Factors)', use_column_width=True)
                else:
                    st.info("Protective factors visualization not available.")
            
            with feature_tabs[1]:
                if os.path.exists(os.path.join(mainpath, r'pos.jpg')):
                    st.image(pos, caption='Features Contributing to Higher Risk (Risk Factors)', use_column_width=True)
                else:
                    st.info("Risk factors visualization not available.")
    
    # Education Tab
    elif selected_tab == "Education":
        st.markdown("<h2 style='font-size: 1.75rem; color: #1F2937;'>Heart Health Education</h2>", unsafe_allow_html=True)
        
        # Educational content with modern design
        edu_col1, edu_col2 = st.columns([2, 1])
        
        with edu_col1:
            st.markdown("""
            <div style="background: white; padding: 1.75rem; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <h3 style="font-size: 1.4rem; color: #1F2937; margin: 0 0 1rem 0;">Understanding Coronary Heart Disease</h3>
                <p style="color: #4B5563; line-height: 1.6;">
                    Coronary heart disease (CHD) is the narrowing or blockage of the coronary arteries, usually caused by atherosclerosis. 
                    Atherosclerosis is the buildup of plaque inside the coronary arteries, which supply oxygen-rich blood to your heart muscle.
                </p>
                
                <h4 style="font-size: 1.2rem; color: #1F2937; margin: 1.5rem 0 0.75rem 0;">Risk Factors</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="background: #F9FAFB; padding: 1rem; border-radius: 12px; border-left: 3px solid #6366F1;">
                        <p style="font-weight: 500; margin: 0 0 0.5rem 0;">Age</p>
                        <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Risk increases with age</p>
                    </div>
                    <div style="background: #F9FAFB; padding: 1rem; border-radius: 12px; border-left: 3px solid #6366F1;">
                        <p style="font-weight: 500; margin: 0 0 0.5rem 0;">High Blood Pressure</p>
                        <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Damages arterial walls</p>
                    </div>
                    <div style="background: #F9FAFB; padding: 1rem; border-radius: 12px; border-left: 3px solid #6366F1;">
                        <p style="font-weight: 500; margin: 0 0 0.5rem 0;">High Cholesterol</p>
                        <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Contributes to plaque buildup</p>
                    </div>
                    <div style="background: #F9FAFB; padding: 1rem; border-radius: 12px; border-left: 3px solid #6366F1;">
                        <p style="font-weight: 500; margin: 0 0 0.5rem 0;">Smoking</p>
                        <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Damages blood vessels</p>
                    </div>
                </div>
                
                <h4 style="font-size: 1.2rem; color: #1F2937; margin: 1.5rem 0 0.75rem 0;">Prevention Strategies</h4>
                <ul style="color: #4B5563; padding-left: 1.25rem; line-height: 1.6;">
                    <li><strong>Regular Exercise</strong>: Aim for at least 150 minutes of moderate activity per week</li>
                    <li><strong>Healthy Diet</strong>: Focus on fruits, vegetables, whole grains, and lean proteins</li>
                    <li><strong>Quit Smoking</strong>: Smoking cessation rapidly reduces heart disease risk</li>
                    <li><strong>Manage Stress</strong>: Practice relaxation techniques like meditation or deep breathing</li>
                    <li><strong>Regular Check-ups</strong>: Monitor blood pressure, cholesterol, and blood sugar levels</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with edu_col2:
            # Add Lottie animation
            st_lottie(analysis_animation, height=200, key="edu_anim")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); padding: 1.5rem; border-radius: 16px; color: white; margin-top: 1.5rem; box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);">
                <h3 style="font-size: 1.3rem; margin: 0 0 1rem 0;">Warning Signs</h3>
                <ul style="padding-left: 1.25rem; opacity: 0.95; line-height: 1.6;">
                    <li>Chest pain or discomfort</li>
                    <li>Shortness of breath</li>
                    <li>Pain in arms, neck, jaw, or back</li>
                    <li>Nausea, indigestion, or heartburn</li>
                    <li>Fatigue or weakness</li>
                    <li>Cold sweat or dizziness</li>
                </ul>
                <p style="margin: 1rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">
                    If you experience these symptoms, especially chest pain, seek medical attention immediately.
                </p>
            </div>
            
            <div style="background: white; padding: 1.5rem; border-radius: 16px; margin-top: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <h3 style="font-size: 1.3rem; color: #1F2937; margin: 0 0 1rem 0;">Heart-Healthy Diet Tips</h3>
                <ul style="color: #4B5563; padding-left: 1.25rem; line-height: 1.6;">
                    <li>Reduce sodium intake</li>
                    <li>Limit saturated and trans fats</li>
                    <li>Eat more fruits and vegetables</li>
                    <li>Choose whole grains</li>
                    <li>Include lean proteins</li>
                    <li>Add omega-3 fatty acids</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # About Tab
    elif selected_tab == "About":
        st.markdown("<h2 style='font-size: 1.75rem; color: #1F2937;'>About MyHeartRisk</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.75rem; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <h3 style="font-size: 1.4rem; color: #1F2937; margin: 0 0 1rem 0;">Our Mission</h3>
            <p style="color: #4B5563; line-height: 1.6;">
                MyHeartRisk is dedicated to leveraging advanced artificial intelligence to provide accessible, 
                accurate cardiovascular risk assessment to help individuals take control of their heart health.
            </p>
            
            <h3 style="font-size: 1.4rem; color: #1F2937; margin: 1.5rem 0 1rem 0;">The Technology</h3>
            <p style="color: #4B5563; line-height: 1.6;">
                Our application uses state-of-the-art machine learning algorithms trained on real clinical data 
                to predict coronary heart disease risk. The model analyzes various health parameters to provide 
                personalized risk assessment and recommendations.
            </p>
            
            <h3 style="font-size: 1.4rem; color: #1F2937; margin: 1.5rem 0 1rem 0;">Data Privacy</h3>
            <p style="color: #4B5563; line-height: 1.6;">
                Your health data is processed locally and is never stored or transmitted. MyHeartRisk is committed 
                to maintaining the highest standards of data privacy and security.
            </p>
            
            <h3 style="font-size: 1.4rem; color: #1F2937; margin: 1.5rem 0 1rem 0;">Disclaimer</h3>
            <p style="color: #4B5563; line-height: 1.6;">
                MyHeartRisk is designed as a screening tool and should not replace professional medical advice. 
                Always consult with a healthcare provider for diagnosis and treatment decisions.
            </p>
            
            <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center;">
                <p style="color: #6B7280; margin: 0;">© 2025 MyHeartRisk. All rights reserved.</p>
                <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Developed with ❤️ for heart health
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()