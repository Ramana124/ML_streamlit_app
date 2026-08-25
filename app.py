import streamlit as st
import pandas as pd
import joblib


# Page Configuration
st.set_page_config(
    page_title="Teen Mental Health Prediction",
    layout="centered"
)

# Load Model and Preprocessing Objects
@st.cache_resource
def load_model():
    model = joblib.load("best_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    gender_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, gender_encoder

model, scaler, gender_encoder = load_model()

# Title
st.title("Teen Social Media & Mental Health")
st.subheader("Depression Risk Prediction")

st.write(
    "Enter the teenager's social media usage, lifestyle, "
    "academic and mental health information to predict "
    "the estimated depression risk."
)

# Input Form
with st.form("prediction_form"):

    # Personal Information
    st.markdown("### Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=13,
            max_value=19,
            value=17,
            step=1
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["male", "female"]
        )

    # Social Media Usage
    st.markdown("### Social Media Usage")

    daily_social_media_hours = st.number_input(
        "Daily Social Media Hours",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.1
    )

    platform_usage = st.selectbox(
        "Primary Social Media Platform",
        ["Instagram", "Other", "TikTok"]
    )

    # Lifestyle Factors
    st.markdown("### Lifestyle & Academic Factors")

    col1, col2 = st.columns(2)

    with col1:

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.1
        )

        screen_time_before_sleep = st.number_input(
            "Screen Time Before Sleep (Hours)",
            min_value=0.0,
            max_value=24.0,
            value=1.5,
            step=0.1
        )

        physical_activity = st.number_input(
            "Physical Activity (Hours)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.1
        )

    with col2:

        academic_performance = st.number_input(
            "Academic Performance",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.01
        )

        social_interaction_level = st.selectbox(
            "Social Interaction Level",
            ["Low", "Medium", "High"]
        )

    # Mental Health Factors
    st.markdown("### Mental Health Factors")
    col1, col2 = st.columns(2)

    with col1:

        stress_level = st.slider(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:

        anxiety_level = st.slider(
            "Anxiety Level",
            min_value=1,
            max_value=10,
            value=5
        )

    # Prediction Button
    submitted = st.form_submit_button(
        "Predict Depression Risk",
        use_container_width=True
    )

# Prediction
if submitted:
    # Encode Gender
    gender_encoded = gender_encoder.transform([gender])[0]

    # Encode Social Interaction
    social_interaction_encoded = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }[social_interaction_level]

    # One-Hot Encode Platform
    platform_Instagram = 1 if platform_usage == "Instagram" else 0
    platform_Other = 1 if platform_usage == "Other" else 0
    platform_TikTok = 1 if platform_usage == "TikTok" else 0

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender_encoded],
        "daily_social_media_hours": [daily_social_media_hours],
        "sleep_hours": [sleep_hours],
        "screen_time_before_sleep": [screen_time_before_sleep],
        "academic_performance": [academic_performance],
        "physical_activity": [physical_activity],
        "social_interaction_level": [social_interaction_encoded],
        "stress_level": [stress_level],
        "anxiety_level": [anxiety_level],
        "platform_usage_Instagram": [platform_Instagram],
        "platform_usage_Other": [platform_Other],
        "platform_usage_TikTok": [platform_TikTok]
    })

    # Apply Same Scaling Used During Training
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Convert numerical prediction to label
    risk_labels = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    predicted_risk = risk_labels[prediction]

    # Display Result
    st.markdown("---")

    st.subheader("Prediction Result")

    if predicted_risk == "Low":

        st.success(
            f"### Depression Risk: {predicted_risk}"
        )

    elif predicted_risk == "Medium":

        st.warning(
            f"### Depression Risk: {predicted_risk}"
        )

    else:

        st.error(
            f"### Depression Risk: {predicted_risk}"
        )

    st.caption(
        "This prediction is generated by the trained machine learning "
        "model and should not be considered a clinical diagnosis."
    )