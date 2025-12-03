# Heart Disease Prediction using Machine Learning

This project leverages machine learning to predict the presence of heart disease using patient health data. Built with Python and Streamlit, it provides an interactive web app for users to enter medical parameters and receive a heart disease risk prediction, alongside detailed model performance metrics.

---


## Features

- **Streamlit Web Interface**: Easy-to-use GUI for entering patient data.
- **Logistic Regression Model**: Trained and tested on the classic Heart Disease dataset.
- **Live Prediction**: Get immediate feedback on heart disease likelihood based on user input.
- **Model Metrics**: Displays training and test set accuracy for transparency.
- **Data Visibility**: Explore the dataset directly within the app.

---


## Dataset

- File: `heart.csv`
- Features include: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, max heart rate achieved, exercise-induced angina, ST depression, slope of peak exercise ST segment, number of major vessels, thalassemia, and target variable (presence of heart disease).

---

## How It Works

1. **Load Data**: Patient data is read from `heart.csv`.
2. **ML Model**: A logistic regression model is trained on historical data.
3. **User Inputs**: Enter values for all health parameters in the web form.
4. **Data Mapping**: Categorical fields are mapped to model-ready numeric values.
5. **Prediction**: The model predicts the likelihood of heart disease for the entered values.
6. **Results**: See prediction results and inspect model accuracy on training/test data.

---

## Getting Started

### Prerequisites

- Python 3.7+
- Python packages: `numpy`, `pandas`, `scikit-learn`, `streamlit`, `matplotlib`

### Installation

```bash
git clone https://github.com/Arshad-5068/heart_disease_prediction.git
cd heart_disease_prediction
pip install -r requirements.txt
```

> **Note**: If `requirements.txt` is not present, install manually:
> ```
> pip install numpy pandas scikit-learn streamlit matplotlib
> ```

### Running the App

```bash
streamlit run code.py
```

Open the provided local URL (e.g., http://localhost:8501) in your browser.

---

## Model Details

- **Algorithm**: Logistic Regression
- **Training Accuracy**: ~0.86
- **Test Accuracy**: ~0.89

---

## File Overview

- `code.py`: Main Streamlit app and model logic.
- `heart.csv`: Tabular heart disease dataset, used for training/testing.

---

## Screenshot Reference

1. ![Input Form](https://github.com/arshad-perampalli/Heart_Disease_Prediction/blob/main/output1.png?raw=true)  
   App input fields for prediction.

2. ![Results](https://github.com/arshad-perampalli/Heart_Disease_Prediction/blob/main/output3.png?raw=true)  
   Data display and model performance results.

3. ![Confusion Matrix](https://github.com/arshad-perampalli/Heart_Disease_Prediction/blob/main/confusion%20matrix.png?raw=true)

4. ![Model Performance](https://github.com/arshad-perampalli/Heart_Disease_Prediction/blob/main/Model%20Performance.png?raw=true)

---


## Acknowledgements

- Dataset source: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Streamlit for rapid web app development.
- scikit-learn for machine learning.
