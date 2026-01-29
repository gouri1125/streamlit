import streamlit as st
import numpy as np
import pickle

with open("model.pickle","rb") as f:
    model = pickle.load(f)


    st.title("Iris species prediction App ")
    sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
    sepal_width  = st.number_input("Sepal Width", 0.0, 10.0, 5.0)
    petal_length = st.number_input("Petal Length", 0.0, 10.0, 5.0)
    petal_width  = st.number_input("Petal Width", 0.0, 10.0, 5.0)
    predict = st.button("Predict Feature")

if predict:
    input_data =np.array([[ sepal_length,sepal_width,petal_length,petal_width]])

    prediction = model.predict(input_data)

    st.success(f"predicted species is: {prediction[0]}")

