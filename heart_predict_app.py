import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("heart_disease_knn_model.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom CSS for simple modern style + center blood sugar input
st.markdown("""
<style>
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: #fafafa;
    color: #333333;
}
h1 {
    color: #b22222; /* firebrick red */
    text-align: center;
    font-weight: 600;
    margin-bottom: 0;
}
p.description {
    text-align: center;
    color: #555555;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}
.form-container {
    max-width: 850px;
    margin: auto;
    background: white;
    padding: 30px 40px;
    border-radius: 12px;
    box-sizing: border-box;
}
.stButton>button {
    background-color: #b22222;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 14px 28px;
    border-radius: 25px;
    border: none;
    transition: background-color 0.25s ease;
    margin-top: 20px;
    width: 100%;
}
.stButton>button:hover {
    background-color: #7a1414;
    cursor: pointer;
}
label {
    font-weight: 500;
    margin-bottom: 6px;
    display: block;
}

/* Center the fasting blood sugar input */
.center-blood-sugar > div {
    margin-left: auto !important;
    margin-right: auto !important;
    max-width: 220px;  /* adjust width as needed */
}
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1>Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Fill out the form below to assess your heart disease risk using Machine Learning.</p>", unsafe_allow_html=True)

# Landing image
st.image("Landing.png", use_container_width=True)

# Form with two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    cholesterol = st.number_input("Cholesterol (mg/dL)", value=200)
    blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)", value=120)
    heart_rate = st.number_input("Heart Rate (bpm)", value=70)
    exercise_hours = st.slider("Exercise Hours Per Week", 0, 20, 3)

with col2:
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])
    family_history = st.selectbox("Family History of Heart Disease", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    obesity = st.selectbox("Obesity", ["Yes", "No"])
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    
    # Center the blood sugar input
    st.markdown('<div class="center-blood-sugar">', unsafe_allow_html=True)
    blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", value=100)
    st.markdown('</div>', unsafe_allow_html=True)

chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

# Form container end
st.markdown('</div>', unsafe_allow_html=True)

def preprocess_input():
    gender_val = 1 if gender == "Male" else 0
    family_history_val = 1 if family_history == "Yes" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    obesity_val = 1 if obesity == "Yes" else 0
    angina_val = 1 if angina == "Yes" else 0

    smoking_current = 1 if smoking == "Current" else 0
    smoking_former = 1 if smoking == "Former" else 0
    smoking_never = 1 if smoking == "Never" else 0

    alcohol_heavy = 1 if alcohol == "Heavy" else 0
    alcohol_moderate = 1 if alcohol == "Moderate" else 0
    alcohol_unknown = 1 if alcohol == "None" else 0

    cp_asymptomatic = 1 if chest_pain == "Asymptomatic" else 0
    cp_atypical = 1 if chest_pain == "Atypical Angina" else 0
    cp_non_anginal = 1 if chest_pain == "Non-anginal Pain" else 0
    cp_typical = 1 if chest_pain == "Typical Angina" else 0

    return np.array([[age, gender_val, cholesterol, blood_pressure, heart_rate,
                      exercise_hours, family_history_val, diabetes_val, obesity_val,
                      stress_level, blood_sugar, angina_val,
                      smoking_current, smoking_former, smoking_never,
                      alcohol_heavy, alcohol_moderate, alcohol_unknown,
                      cp_asymptomatic, cp_atypical, cp_non_anginal, cp_typical]])

if st.button("Predict Heart Disease Risk"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Heart Disease ({prob*100:.1f}% confidence)")
    else:
        st.success(f"Low Risk of Heart Disease ({(1 - prob)*100:.1f}% confidence)")

st.markdown(
    """
    <div style="text-align:center; margin-top: 3rem; font-size: 0.8rem; color: #777;">
    ⚠️ This tool is for educational purposes and not a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
