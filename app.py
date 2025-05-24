# stress_app.py

import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('gaussian_naive_bayes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🧘‍♀️ Stress Level Prediction App")

st.sidebar.header("📝 Enter Your Details:")

age = st.sidebar.slider("Age", 10, 80, 25)
income = st.sidebar.number_input("Monthly Income (INR)", value=20000)
expenses = st.sidebar.number_input("Medical Expenses per Month (INR)", value=2000)
sleep = st.sidebar.slider("Sleep Hours per Day", 0.0, 12.0, 6.0)
support = st.sidebar.slider("Support System Score (0-10)", 0, 10, 5)

if st.sidebar.button("Predict Stress Level"):
    input_data = pd.DataFrame([[age, income, expenses, sleep, support]],
                              columns=['Age', 'Monthly_Income_INR', 'Medical_Expenses_per_Month_INR',
                                       'Sleep_Hours_per_Day', 'Support_System_Score'])

    prediction = model.predict(input_data)[0]

    st.subheader("🔍 Prediction:")
    if prediction == 0:
        st.success("Low Stress 😌")
        st.write("You're doing great! It’s wonderful to see you in a calm and steady place."
                 " Even low stress can sometimes build up if ignored, so remember to"
                 " keep nurturing yourself—rest well, stay connected with the people you "
                 "love, and do what brings you joy. Keep this balance going, because your "
                 "peace of mind is a gift worth protecting")
    elif prediction == 1:
        st.warning("Moderate Stress 😐")
        st.write("You’re doing okay, and it’s perfectly normal to feel a bit overwhelmed "
                 "sometimes. Life gets busy, and stress sneaks in—but don’t forget to pause,"
                 " breathe, and take care of yourself. Small breaks, good sleep, and talking"
                 " to someone you trust can make a big difference. You’ve got this, one step "
                 "at a time")
    elif prediction == 2:
        st.error("High Stress 😣")
        st.write("Hey… it’s okay to not be okay right now. You’re under a lot, and your "
                 "mind and heart are trying hard to carry it all. But please remember — "
                 "you don’t have to go through it alone. Take a deep breath, even if it "
                 "feels heavy. Reach out to someone you trust. Talk, rest, cry if you need "
                 "to — it's all valid. You're stronger than this moment, and brighter days "
                 "are waiting. One breath at a time — you will get through this. 💙")
    else:
        st.info("Unknown Stress Level")

st.markdown("---")

