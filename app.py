import pandas as pd
import joblib
import os
import streamlit as st



mainpath = os.path.dirname(__file__)
heartimage = os.path.join(mainpath, r'heart.png')
neg = os.path.join(mainpath, r'neg.jpg')
pos = os.path.join(mainpath, r'pos.jpg')
compare = os.path.join(mainpath, r'Comparison.jpg')
pca = os.path.join(mainpath, r'PCA.jpg')

# Title
st.image(heartimage, width=100)
st.title('MyHeartRisk')
st.subheader('Standardized WebApp to Predict Heart Disease Based on Real World Hospital Case/Control Data')
st.write('**Instruction**')
st.write('This app uses machine learning to predict the likelihood of heart disease based on user-provided health parameters. Adjust the options in the sidebar to input your health data and click "Check" to see your risk assessment.')
# Load dataset
tab1, tab2, tab3, tab4 = st.tabs(["Report", "Dataset", "Model", "Interpretation"])

data_path = os.path.join(mainpath, r'Data_health1.xlsx')
df = pd.read_excel(data_path)

# Separate features and target
X = df.drop(columns=['Target'])  # make sure 'Target' is the correct target column
features = X.columns

# Sidebar - User input
st.sidebar.header('User Input Bar')
st.sidebar.write("Give Your Parameters and Click 'Check' to Predict Heart Disease Risk.")
predict_btn = st.sidebar.button('Check')
user_data = {}
for col in features:
    if pd.api.types.is_numeric_dtype(X[col]):
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())

        if col.lower() == "age":  # special case for Age → integer slider
            user_data[col] = st.sidebar.slider(f"{col}", int(min_val), int(max_val), int(round(mean_val)))
        else:
            user_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
    else:
        unique_vals = X[col].dropna().unique()
        user_data[col] = st.sidebar.selectbox(f"{col}", unique_vals)

# Convert input into DataFrame
input_df = pd.DataFrame([user_data])

# Load model
model_path = os.path.join(mainpath, r'model.joblib')
model = joblib.load(model_path)

# Prediction only on button click
with tab1:
    if predict_btn:
        
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        st.subheader('Results')
        if prediction == 1:
            st.write("**⚠️ Your Parameters Resemble That of Patient Cases With Heart Disease. Take Care of Yourself. Heart Disease Predicted**")
            confidence = probabilities[1] * 100
        else:
            st.write("**✅ Your Parameters Don’t Resemble Patient Cases With Heart Disease. No Heart Diseases Predicted**")
            confidence = probabilities[0] * 100

        st.progress(int(confidence), text=f"Confidence Level: {confidence:.2f}%", width=400)


with tab2:
    st.subheader('Dataset Overview')
    st.write('120 Datapoints were collected as per sample size statistics consisiting of 60 cases and controls each. The External set consists of 10 cases and 10 controls.')
    st.image(pca, caption='PCA Plot Showing Distribution of Cases and Controls (Cummalative Variance: 39%)')
    st.write('The PCA plot above shows the distribution of cases and controls based on the health parameters provided. The separation indicates that the features used are effective in distinguishing between the two groups.')
 
   


with tab3:
    st.subheader('Model Performance')
    st.write('The model used is Logistic Regression for prediction, trained on a real-world dataset of hospital case/control data. Below are the performance metrics of the model:')
    
    st.image(compare, caption='Model Performance Comparison (10 Models with 5 Fold Stratified CV)')
   
    st.write("""
    Internal Validation Metrics ( 5x2 K-Fold Stratified Cross Validation ):
    - **Accuracy**: 0.92 ± 0.05
    - **Sensitivity**: 0.90 ± 0.06
    - **Specificity**: 0.95 ± 0.06
    - **MCC**: 0.89 ± 0.02
    
    External Validation Metrics:
    - **Accuracy**: 0.90
    - **Sensitivity**: 1.00
    - **Specificity**: 0.80
    - **MCC**: 0.81
             
    These metrics indicate that the model is quite effective at predicting heart disease based on the provided health parameters.
    """)

with tab4:
    st.subheader('Interpretation: Logistic Regression Coefficients')
    st.write('This section provides insights into how different health parameters influence the prediction of heart disease. Understanding these factors can help in making informed health decisions.')
    st.image(neg, caption='Features Contributing to Controls')
    st.image(pos, caption='Features Contributing to Cases')
   






