import pandas as pd
import joblib
import os
import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


# Configure page
st.set_page_config(
    page_title="MyHeartRisk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Rotating dots HTML
rotating_dots = """
<div class="loading-dots">
  <span></span><span></span><span></span><span></span>
</div>

<style>
.loading-dots {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80px;
}
.loading-dots span {
  width: 12px;
  height: 12px;
  margin: 0 6px;
  background: #3498db;
  border-radius: 50%;
  animation: loading-dots 1s infinite ease-in-out;
}
.loading-dots span:nth-child(1) { animation-delay: 0s; }
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }
.loading-dots span:nth-child(4) { animation-delay: 0.6s; }

@keyframes loading-dots {
  0%, 80%, 100% { transform: scale(0.8); opacity: 0.6; }
  40% { transform: scale(1.2); opacity: 1; }
}
</style>
"""

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff6b6b, #ee5a6f);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .instruction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .positive-result {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #51cf66, #69db7c);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar .stSelectbox label, .sidebar .stSlider label {
        font-weight: 600;
        color: #2d3436;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        border-radius: 12px;
        background-color: white;
        border: 2px solid #e9ecef;
        color: #495057;
        font-weight: 700;
        padding: 0 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
        border-color: #007bff;
        box-shadow: 0 4px 8px rgba(0,123,255,0.3);
        font-weight: 700;
    }
    
    .stTabs [role="tabpanel"] {
        padding-top: 0.5rem;
    }
    
    .dataset-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .model-performance {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .interpretation-section {
        background: #f1f3f4;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

mainpath = os.path.dirname(__file__)
heartimage = os.path.join(mainpath, r'heart.png')
neg = os.path.join(mainpath, r'neg.jpg')
pos = os.path.join(mainpath, r'pos.jpg')
compare = os.path.join(mainpath, r'Comparison.jpg')
pcaimage = os.path.join(mainpath, r'PCA.jpg')

# Header section with titles on left, heart image on right
col1, col2 = st.columns([3, 1])
with col1:
    st.title("MyHeartRisk")
    st.subheader("Standardized WebApp to Predict Coronary Artery Disease Based on Real World Hospital Case/Control Data")
with col2:
    st.image(heartimage, width=120)

# Load dataset first to get predict_btn status
data_path = os.path.join(mainpath, r'Data_health1.xlsx')
df = pd.read_excel(data_path)

# Separate features and target
X = df.drop(columns=['Target'])  # make sure 'Target' is the correct target column
features = X.columns

# Modern Sidebar Design
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #ff6b6b, #ee5a6f); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Health Parameters</h2>
        <p style="color: white; opacity: 0.9; margin: 0;">Enter your health data below</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create sections for better organization
    st.markdown("### 👤 Personal Information")
    user_data = {}
    
    # Handle age separately for better UX
    age_col = None
    for col in features:
        if col.lower() == "age":
            age_col = col
            break
    
    if age_col:
        min_val = int(X[age_col].min())
        max_val = int(X[age_col].max())
        mean_val = int(round(X[age_col].mean()))
        user_data[age_col] = st.slider(
            f"**{age_col} (years)**", 
            min_val, max_val, mean_val,
            help="Your current age in years"
     
        )
    
  
    
   
    
    # Handle categorical features
    categorical_features = []
    for col in features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            categorical_features.append(col)
    
    if categorical_features:
        for col in categorical_features:
            unique_vals = X[col].dropna().unique()
            user_data[col] = st.selectbox(
                f"**{col}**", 
                unique_vals,
                help=f"Select your {col.lower()}"
                
            )
    
    st.markdown("---")
    
    # Modern prediction button
    predict_btn = st.button(
        "🔍 Analyze Risk", 
        type="primary", 
        use_container_width=True,
        help="Click to get your heart disease risk assessment"
    )

# Show instruction card only when analysis hasn't started
if not predict_btn:
    st.markdown("""
    <div class="instruction-card">
        <h3>🎯 How to Use This App</h3>
        <p>This advanced machine learning application predicts your Coronary Artery Disease risk using hospital-validated data. Simply:</p>
        <ul>
            <li><strong>Step 1:</strong> Adjust your health parameters in the sidebar</li>
            <li><strong>Step 2:</strong> Click the "🔍 Analyze Risk" button</li>
            <li><strong>Step 3:</strong> Review your personalized risk assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Convert input into DataFrame
input_df = pd.DataFrame([user_data])

# Load model
model_path = os.path.join(mainpath, r'model.joblib')
model = joblib.load(model_path)

# Show tabs and content only when analysis starts
tab1, tab2, tab3, tab4 = st.tabs(["Report Dashboard", "Dataset", "Model Performance", "Feature Analysis"])

if predict_btn:
  
    placeholder = st.empty()
    with placeholder.container():
        components.html(rotating_dots, height=100)
    
    # long processing
    time.sleep(2)
    
    # remove the whole container
    placeholder.empty()

   
    
    # Prediction only on button click
    with tab1:
        st.markdown("## Report Dashboard")
        
        with st.spinner(" Analyzing your health parameters..."):
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="positive-result">
                    <h2>⚠️ Higher Risk Detected</h2>
                    <p style="font-size: 1.2rem; margin-bottom: 1rem;">Your health parameters show similarities to patients diagnosed with Coronary Artery Disease.</p>
                    <p style="font-size: 1rem; opacity: 0.9;">
                        <strong>Recommendation:</strong> Please consult with a healthcare professional for a comprehensive evaluation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                confidence = probabilities[1] * 100
                risk_level = "HIGH"
                risk_color = "#ff6b6b"
            else:
                st.markdown("""
                <div class="negative-result">
                    <h2>✅ Lower Risk Indicated</h2>
                    <p style="font-size: 1.2rem; margin-bottom: 1rem;">Your health parameters suggest a lower likelihood of Coronary Artery Disease.</p>
                    <p style="font-size: 1rem; opacity: 0.9;">
                        <strong>Recommendation:</strong> Continue maintaining a healthy lifestyle and regular check-ups.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                confidence = probabilities[0] * 100
                risk_level = "LOW"
                risk_color = "#51cf66"
        
      
        
        
        with col2:
            # Confidence metrics card
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {risk_color}; margin-bottom: 1rem;">🎯 Confidence Score</h3>
                <div style="font-size: 2.5rem; font-weight: bold; color: {risk_color};">
                    {confidence:.1f}%
                </div>
                <p style="margin-top: 0.5rem; color: #666;">
                    Risk Level: <strong style="color: {risk_color};">{risk_level}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar with modern styling
        st.markdown("### Detailed Analysis")
        progress_col1, progress_col2 = st.columns([3, 1])
        
        with progress_col1:
            st.progress(confidence/100, text=f"Model Confidence: {confidence:.2f}%")
        
        with progress_col2:
            if confidence >= 80:
                st.markdown("🟢 **High Confidence**")
            elif confidence >= 60:
                st.markdown("🟡 **Medium Confidence**")
            else:
                st.markdown("🟠 **Low Confidence**")
        
        # Additional insights
        st.markdown("---")
        df = pd.read_excel(data_path)
        
        # -----------------------
        # Add user input
        # -----------------------
        # user_data comes from Streamlit inputs
        user_df = pd.DataFrame([user_data])
        user_df["Target"] = -1  # special label for USER
        df = pd.concat([df, user_df], ignore_index=True)
        
        # -----------------------
        # Identify numerical and categorical columns
        # -----------------------
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'Target' in numerical_cols:
            numerical_cols.remove('Target')
        
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        # -----------------------
        # Preprocess
        # -----------------------
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        df_processed = preprocessor.fit_transform(df)
        
        # -----------------------
        # PCA
        # -----------------------
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_processed)
        explained_var = pca.explained_variance_ratio_ * 100
        
        # -----------------------
        # Plot in Streamlit
        # -----------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Controls (0)
        ax.scatter(pca_result[df["Target"] == 0, 0],
                   pca_result[df["Target"] == 0, 1],
                   c="blue", label="Controls", alpha=0.6)
        
        # Cases (1)
        ax.scatter(pca_result[df["Target"] == 1, 0],
                   pca_result[df["Target"] == 1, 1],
                   c="red", label="Cases", alpha=0.6)
        
        # User (-1)
        ax.scatter(pca_result[df["Target"] == -1, 0],
                   pca_result[df["Target"] == -1, 1],
                   c="gold", s=200, edgecolor="black", marker="*", label="You")
        
        ax.set_xlabel(f"PCA 1 ")
        ax.set_ylabel(f"PCA 2")
        ax.set_title("PCA Projection")
        ax.legend()
        st.markdown("""
          <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px;">
            <p>The Star Represents Your Data Point in Parameteric Space (Cummulative Variance = 37 %)</p>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig, width=800)
      
                        
        
with tab1:
       st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #ff6b6b, #ee5a6f); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Welcome to MyHeartRisk</h2>
        <p style="color: white; opacity: 0.9; margin: 0; margin-bottom: 2rem;">
            Enter your health parameters in the sidebar to get started.
        </p>
        <div style="background: white; padding: 1.5rem; border-radius: 15px; margin: 1rem auto; max-width: 700px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <p style="color: #495057; margin: 0;">
               <b> Interpretable Parameters • Hospital-Grade Data • 94% Accuracy </b>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
   
   


    
with tab2:
    st.markdown("""
    <div class="dataset-section">
        <h2>Dataset Overview</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            Our model is trained on carefully curated hospital data to ensure real-world accuracy and reliability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Samples",
            value="120 patients",
            delta="60 cases + 60 controls",
            help="Balanced dataset for optimal learning"
        )
    
    with col2:
        st.metric(
            label="External Validation",
            value="20 samples",
            delta="10 cases + 10 controls",
            help="Independent test set for validation"
        )
    
    with col3:
        st.metric(
            label="Data Quality",
            value="Hospital Grade",
            delta="Real-world clinical data",
            help="Professional medical data collection"
        )
    
    st.markdown("---")
    
    st.markdown("### Principal Component Analysis")
    st.image(pcaimage, caption='PCA Visualization: Distribution of Cases vs Controls (Cumulative Variance: 37%)', width=800)
    
    st.markdown("""
    <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h4> What This Chart Shows:</h4>
        <p>The PCA plot demonstrates clear separation between CAD cases and healthy controls, 
        indicating that the health parameters used in our model are effective discriminators. 
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="dataset-section">
        <h2> Deployed Model</h2>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Logistic Regression
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        ### Comparative Analysis
        *5 K-Fold Stratified Cross Validation*
        """)
    st.image(compare, caption=' Comprehensive Model Comparison (10 Models with 5-Fold Stratified Cross-Validation)', width=800)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Internal Validation Results
        *5x2 K-Fold Stratified Cross Validation*
        """)
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric(" Accuracy", "0.94 ± 0.01")
            st.metric(" Sensitivity", "0.92 ± 0.04")
        with metrics_col2:
            st.metric(" Specificity", "0.95 ± 0.03")
            st.metric(" MCC", "0.88 ± 0.02")
    
    with col2:
        st.markdown("""
        ###  External Validation Results
        *Independent Test Set (20 Samples)*
        """)
        
        ext_col1, ext_col2 = st.columns(2)
        with ext_col1:
            st.metric(" Accuracy", "100%")
            st.metric(" Sensitivity", "100%")
        with ext_col2:
            st.metric(" Specificity", "100%")
            st.metric(" MCC", "100%")
    
    st.markdown("""
    <div style="background: #e8f5e8; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
        <h4>Important to Note: </h4>
        <p> The External Validation Set Consisted of 20 Data Points , Hence 100% Metrics in External Validation is Expected Due to Possible Congeneric Parameters</p>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="interpretation-section">
        <h2> Understanding the Predictions</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem;">
            Learn how different health factors influence coronary artery disease.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Protective Factors")
        st.image(neg, caption='Features Associated with Lower Risk (Controls)')
        st.markdown("""
        <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px;">
            <p><strong>Bars</strong> represent health parameters that, when present, 
            are associated with <strong>lower risk</strong> of coronary artery disease.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔴 Risk Factors")
        st.image(pos, caption='Features Associated with Higher Risk (Cases)')
        st.markdown("""
        <div style="background: #ffeaea; padding: 1rem; border-radius: 8px;">
            <p><strong>Bars</strong> represent health parameters that, when present, 
            are associated with <strong>higher risk</strong> of coronary artery disease.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
























































