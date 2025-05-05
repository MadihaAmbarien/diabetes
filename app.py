import streamlit as st
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Diabetes Progression Prediction")
st.write("Enter the values for the following features to predict disease progression:")

# Collect user input for each feature
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0, format="%.4f")
    user_input.append(val)

# Convert input to numpy array
input_data = np.array(user_input).reshape(1, -1)

# Predict and show result
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted disease progression score: {prediction[0]:.2f}")
