# Sulfur Recovery Unit (SRU) Emission Prediction using Machine Learning

This project focuses on building a **data-driven soft sensor** to predict emissions from a Sulfur Recovery Unit (SRU) using machine learning techniques. The primary objective is to model sulfur emissions based on process and environmental parameters using a **Just-in-Time (JIT) modeling approach** and regression-based learning.

---

## Problem Statement

Accurate real-time measurement of sulfur emissions in SRUs is often challenging due to sensor delays, noise, or hardware limitations. This project proposes a soft sensor framework that predicts emissions using available upstream data, enabling better control and regulatory compliance.

---

## Dataset

The dataset includes:
- **Categorical features**: `State`, `City`, `Station`
- **Numerical features**: Temperature, Humidity, Wind Speed, etc.
- **Target variable**: Sulfur-related Air Quality Index (AQI) or equivalent emission metric

---

## Project Highlights

- Applies **Just-in-Time (JIT) model learning** for adaptive and local prediction
- Compares multiple traditional regression methods:
  - Linear Regression
  - Polynomial Regression (degree = 2)
  - Decision Tree Regression
- Evaluates models on both training and testing data
- Plans to implement **Neural Networks** in future work

---

## Evaluation Metrics

Each model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R² Score)
- Root Mean Squared Log Error (RMSLE)

---

## Tools and Libraries

- **Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn

---

## How to Run

1. Clone this repository
2. Install required dependencies using `pip install -r requirements.txt`
3. Place your dataset in the `data set/` folder
4. Run the script:  
   ```bash
   python sru_prediction.py
