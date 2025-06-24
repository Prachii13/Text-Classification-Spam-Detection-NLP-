import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📨 Spam Classifier")

msg = st.text_area("Enter a message")

if st.button("Predict"):
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    st.success("✅ Not Spam" if pred == 0 else "❌ Spam")
