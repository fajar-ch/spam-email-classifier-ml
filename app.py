import streamlit as st
import joblib

# Load model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Email Spam Classifier (CS + AI System)")
st.write("Enter email text to check if it's SPAM or NOT")

email = st.text_area("Enter Email Content")

if st.button("Check"):
    if email:
        vec = vectorizer.transform([email])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("⚠ SPAM EMAIL")
        else:
            st.success("NOT SPAM (Safe Email)")
    else:
        st.warning("Please enter email text")