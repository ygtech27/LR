import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression  # Include this to help pickle resolve class

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 50.0)

if st.button("Predict"):
    sex_val = 0 if sex == "male" else 1
    features = np.array([[pclass, sex_val, age, sibsp, parch, fare]])
    prediction = model.predict(features)
    st.write("Prediction: ", "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive")
