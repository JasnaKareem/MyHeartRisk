import pandas as pd
import joblib
import os
import streamlit as st



mainpath = os.path.dirname(__file__)
heartimage = os.path.join(mainpath, r'heart.png')
# Title
st.image(heartimage, width=100)
st.title('MyHeartRisk')
st.subheader('Standardized WebApp to Predict Heart Disease Based on Real World Hospital Case/Control Data')
st.write('**Instruction**')
st.write('This app uses machine learning to predict the likelihood of heart disease based on user-provided health parameters. Adjust the options in the sidebar '>>' to input your health data and click "Check" to see your risk assessment.')
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
   

