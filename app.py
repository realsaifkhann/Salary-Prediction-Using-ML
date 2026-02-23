import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Page Config (Must for premium look)
# -----------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .stButton>button {
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
            font-weight: bold;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #262730;
            text-align: center;
            animation: fadeIn 1.5s ease-in;
        }
        @keyframes fadeIn {
            0% {opacity: 0;}
            100% {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("lasso_salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title Section
# -----------------------------
st.title("💰 Employee Salary Prediction")
st.write("Predict employee monthly salary using Machine Learning (Lasso Regression)")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Employee Details")

age = st.sidebar.slider("Age", 20, 60, 30)
experience = st.sidebar.slider("Years of Experience", 0, 35, 5)
performance = st.sidebar.slider("Performance Rating", 1, 5, 3)
hours = st.sidebar.slider("Work Hours / Week", 30, 60, 40)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
department = st.sidebar.selectbox("Department", ["HR", "IT", "Sales", "Finance", "Operations"])
jobrole = st.sidebar.selectbox("Job Role", ["Manager", "Analyst", "Engineer", "Executive", "Consultant"])
education = st.sidebar.selectbox("Education Level", ["Bachelor", "Master", "PhD"])

# -----------------------------
# Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "YearsExperience": [experience],
    "PerformanceRating": [performance],
    "WorkHoursPerWeek": [hours],
    "Gender": [gender],
    "Department": [department],
    "JobRole": [jobrole],
    "EducationLevel": [education]
})

# Encode
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale
input_scaled = scaler.transform(input_encoded)

# -----------------------------
# Layout Columns
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# Prediction Section
# -----------------------------
with col1:
    st.subheader("🎯 Salary Prediction")

    if st.button("Predict Salary"):
        prediction = model.predict(input_scaled)

        st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Monthly Salary</h2>
                <h1 style="color:#00FFAA;">₹ {int(prediction[0])}</h1>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Input Summary Section
# -----------------------------
with col2:
    st.subheader("📊 Input Summary")

    st.write("**Employee Profile:**")
    st.write(f"- Age: {age}")
    st.write(f"- Experience: {experience} years")
    st.write(f"- Performance Rating: {performance}")
    st.write(f"- Work Hours: {hours} / week")
    st.write(f"- Role: {jobrole}")
    st.write(f"- Department: {department}")
    st.write(f"- Education: {education}")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📈 Feature Influence (Model Coefficients)")

coef = pd.Series(model.coef_, index=scaler.feature_names_in_)
important_features = coef[coef != 0].sort_values()

st.bar_chart(important_features)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("✅ Model: Lasso Regression | Built with Streamlit")