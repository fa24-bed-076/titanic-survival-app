import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# Label encoders are recreated exactly like the notebook (Step 5.1), using the same fixed category lists
pclass_label_encoder = LabelEncoder().fit(np.ravel(["First", "Second", "Third"]))
gender_label_encoder = LabelEncoder().fit(np.ravel(["Male", "Female"]))
sibling_label_encoder = LabelEncoder().fit(np.ravel(["Zero", "One", "Two", "Three"]))
embarked_label_encoder = LabelEncoder().fit(np.ravel(["Southampton", "Cherbourg", "Queenstown"]))


@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter the passenger details and the model will predict whether the passenger would have survived.")

pclass_input = st.selectbox("Passenger class (PClass)", ["First", "Second", "Third"], index=2)
gender_input = st.selectbox("Gender", ["Male", "Female"])
sibling_input = st.selectbox("Number of siblings / spouses aboard (Sibling)", ["Zero", "One", "Two", "Three"])
embarked_input = st.selectbox("Port of embarkation (Embarked)", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict"):
    features = pd.DataFrame({
        "PClass": [pclass_label_encoder.transform([pclass_input])[0]],
        "Gender": [gender_label_encoder.transform([gender_input])[0]],
        "Sibling": [sibling_label_encoder.transform([sibling_input])[0]],
        "Embarked": [embarked_label_encoder.transform([embarked_input])[0]],
    })
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("Prediction: this passenger would have SURVIVED.")
    else:
        st.error("Prediction: this passenger would NOT have survived.")
