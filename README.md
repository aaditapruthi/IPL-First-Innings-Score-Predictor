# IPL First Innings Score Predictor

## Overview

This project predicts the final first innings score of an IPL match using Machine Learning.

The model is trained on historical IPL match data and compares multiple regression algorithms to select the best-performing model.

---

## Features

- Predict IPL first innings scores
- Compare multiple ML algorithms
- Interactive Streamlit web application
- Model performance visualization
- Real-time score prediction

---

## Machine Learning Models Used

- Linear Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- MLP Regressor (Neural Network)

---

## Evaluation Metrics

The models are compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Pickle

---

## Project Structure

```text
IPL_SCORE_PREDICTOR/
│
├── app.py
├── train_model.py
├── data.csv
├── best_model.pkl
├── model_results.csv
├── requirements.txt
├── README.md
└── assets/
```

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

### Run the Web App

```bash
streamlit run app.py
```

---

## Results

The best model is selected automatically based on R² Score.

The application also displays:

- Model Comparison Chart
- Detailed Metrics Table
- Score Interpretation

---

## Author

Aadita Pruthi

AI/ML Internship Project