import streamlit as st
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the XGBoost sepsis model
sepsis_model = joblib.load('sepsis_xgb_model.pkl')

# Load fine-tuned GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./gpt2-sepsis")
gpt2_model = AutoModelForCausalLM.from_pretrained("./gpt2-sepsis")
generator = pipeline("text-generation", model=gpt2_model, tokenizer=tokenizer)

# App title
st.title("Sepsis Risk Prediction & Explanation")

# User inputs
st.sidebar.header("Patient Vitals")
features = {
    'Heart Rate': st.sidebar.number_input("Heart Rate", value=90),
    'Temperature': st.sidebar.number_input("Temperature (°C)", value=37.0),
    'WBC': st.sidebar.number_input("White Blood Cell Count", value=7.0),
    'Blood Pressure': st.sidebar.number_input("Blood Pressure", value=120),
    'Respiratory Rate': st.sidebar.number_input("Respiratory Rate", value=18),
    'Lactate': st.sidebar.number_input("Lactate", value=1.0),
    'Glucose': st.sidebar.number_input("Glucose", value=90),
    'Age': st.sidebar.number_input("Age", value=40)
}


if st.button("Predict Sepsis Risk"):
    # Convert to feature vector
    X_input = [list(features.values())]
    
    # Predict risk
    risk_score = sepsis_model.predict_proba(X_input)[0][1]
    
    # Display risk score
    st.write(f"### Sepsis Risk Score: **{risk_score:.2f}**")
    
    # Generate medical explanation using GPT-2
    vitals_text = ', '.join(f"{key}: {value}" for key, value in features.items())
    prompt = f"Patient has the following vitals: {vitals_text}. The model predicts a sepsis risk score of {risk_score:.2f}. Provide a brief medical explanation."
    explanation = generator(prompt, max_new_tokens=60, num_return_sequences=1)
    
    # Show explanation
    st.write("### Explanation")
    st.write(explanation[0]['generated_text'])
