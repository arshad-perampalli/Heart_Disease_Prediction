import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

heart_df =pd.read_csv('https://github.com/arshad-perampalli/Heart_Disease_Prediction/blob/c7b47320395377eaa1c4fbfa435d9e791c912c14/heart.csv')

X=heart_df.drop('target',axis=1)
y=heart_df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,y_test)


# Assuming `model` is already loaded and `heart_df`, `training_data_accuracy`, `test_data_accuracy` are defined

# Title and input fields
st.subheader("SMART HEALTH: ADVANCED HEART DISEASE PREDICTION USING MACHINE LEARNING")

# Input fields
input_text1 = st.text_input('Enter age')  # Numeric
input_text2 = st.selectbox('Select sex', options=['Female', 'Male'])  # Categorical
input_text3 = st.selectbox(
    'Select chest pain type',
    options=['High heart pain', 'Low heart pain', 'Other pain', 'No pain']  # Categorical
)
input_text4 = st.text_input('Enter resting blood pressure')  # Numeric
input_text5 = st.text_input('Enter serum cholesterol (mg/dl)')  # Numeric
input_text6 = st.selectbox('Is fasting blood sugar > 120 mg/dl?', options=['No', 'Yes'])  # Categorical
input_text7 = st.selectbox(
    'Select resting electrocardiographic results',
    options=['Normal', 'ST-T wave abnormality', 'Definite left ventricular hypertrophy']  # Categorical
)
input_text8 = st.text_input('Enter maximum heart rate achieved')  # Numeric
input_text9 = st.selectbox('Exercise-induced angina (heart pain)?', options=['No', 'Yes'])  # Categorical
input_text10 = st.text_input('Enter ST depression induced by exercise')  # Numeric
input_text11 = st.selectbox(
    'Select slope of the peak exercise ST segment',
    options=['Upsloping', 'Flat', 'Downsloping']  # Categorical
)
input_text12 = st.number_input('Enter number of major vessels (0-3) colored by fluoroscopy', min_value=0, max_value=3, step=1)  # Numeric
input_text13 = st.selectbox(
    'Select thalassemia type',
    options=['Normal', 'Fixed defect', 'Reversible defect']  # Categorical
)

if st.button('Submit'):
    try:
        # Mapping categorical values to numeric equivalents
        sex_map = {'Female': 0, 'Male': 1}
        cp_map = {'High heart pain': 0, 'Low heart pain': 1, 'Other pain': 2, 'No pain': 3}
        fbs_map = {'No': 0, 'Yes': 1}
        restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Definite left ventricular hypertrophy': 2}
        exang_map = {'No': 0, 'Yes': 1}
        slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
        thal_map = {'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}

        # Converting inputs to numeric values
        sprated_input = [
            float(input_text1),  # Age
            sex_map[input_text2],  # Sex
            cp_map[input_text3],  # Chest Pain
            float(input_text4),  # Resting Blood Pressure
            float(input_text5),  # Serum Cholesterol
            fbs_map[input_text6],  # Fasting Blood Sugar
            restecg_map[input_text7],  # Resting ECG
            float(input_text8),  # Max Heart Rate
            exang_map[input_text9],  # Exercise Angina
            float(input_text10),  # ST Depression
            slope_map[input_text11],  # Slope
            int(input_text12),  # Number of Major Vessels
            thal_map[input_text13]  # Thalassemia
        ]

        # Converting to NumPy array and reshaping for prediction
        np_df = np.asarray(sprated_input).reshape(1, -1)
        prediction = model.predict(np_df)

        # Display prediction result
        if prediction[0] == 0:
            st.write("This person doesn't have heart disease.")
        else:
            st.write("This person has heart disease.")

    except ValueError as e:
        st.error(f"Error: {e}. Please enter all values properly.")

# Display additional information
st.subheader("About data")
st.write(heart_df)  # Assuming `heart_df` is defined
st.subheader("Model Performance on Training Data")
st.write(training_data_accuracy)  # Assuming it's defined
st.subheader("Model Performance on Test Data")
st.write(test_data_accuracy)  # Assuming it's defined

