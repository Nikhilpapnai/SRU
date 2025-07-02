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
- Implements and compares multiple regression models:
  - Linear Regression
  - Polynomial Regression (degree = 2)
  - Decision Tree Regression
  - Random Forest Regression
  - XGBoost Regression
- Evaluates model performance on both training and testing datasets
- Plans to implement Neural Networks in future work

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
- **Libraries**:
  - NumPy
  - Pandas
  - Scikit-learn
  - XGBoost
  - Matplotlib (optional for visualization)

---

## Folder Structure

