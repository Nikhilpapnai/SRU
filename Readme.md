# Sulfur Recovery Unit (SRU) Emission Prediction using Machine Learning

**Author:** Nikhil Papnai
**Supervisor:** Prof. Jayaram Valluru

This project focuses on building a **data-driven soft sensor** to predict emissions from a Sulfur Recovery Unit (SRU) using advanced machine learning and dynamic modeling techniques. The primary objective is to model sulfur emissions based on process and environmental parameters using **Just-in-Time (JIT) modeling**, regression-based learning, and advanced hybrid models including deep learning and graph-based methods.

---

## Problem Statement

Sulfur Recovery Units are critical in controlling sulfur emissions from industrial processes. However, **accurate real-time measurement of emissions** is often limited by sensor delays, noise, and hardware constraints.

This project develops a **predictive soft sensor** that uses upstream process data to estimate sulfur emissions in real time. This allows:

* Improved **process optimization**
* Compliance with **environmental regulations**
* Enhanced understanding of **dynamic chemical processes**

---

## Dataset

The dataset includes:

* **Categorical features:** `State`, `City`, `Station`
* **Numerical features:** Temperature, Humidity, Wind Speed, etc.
* **Target variable:** Sulfur-related Air Quality Index (AQI) or equivalent emission metric

**Preprocessing Example:**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

for col in ['State', 'City', 'Station']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Methodology

### 1. Initial Regression Models

Implemented baseline regression models:

* Linear Regression
* Polynomial Regression (degree = 2)
* Decision Tree Regression
* Random Forest Regression
* XGBoost Regression

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 2. Neural Networks

Tested various neural network architectures to capture non-linear and temporal dependencies:

* ANN (Artificial Neural Networks)
* DNN (Deep Neural Networks)
* RNN (Recurrent Neural Networks)

```python
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)
```

### 3. Advanced Graph & Temporal Modeling

Currently focusing on **hybrid models** for spatial and temporal process dynamics:

* **Graph Neural Networks (GNN):** Captures spatial relationships
* **LSTM (Long Short-Term Memory):** Captures temporal dependencies
* **Gaussian Process Regression (GPR):** Provides uncertainty estimates
* **Moving Horizon Estimation (MHE):** Updates predictions dynamically

```python
from torch_geometric.nn import GCNConv

class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

---

## Evaluation Metrics

Each model is evaluated using:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Coefficient of Determination (R² Score)
* Root Mean Squared Log Error (RMSLE)

---

## Tools & Libraries

* **Language:** Python
* **Libraries:**

  * NumPy, Pandas, Scikit-learn
  * XGBoost
  * PyTorch, PyTorch Geometric
  * Matplotlib, Seaborn

---

## Project Highlights

* Developed a **robust soft sensor framework** for SRU emission prediction
* Explored **multiple machine learning and deep learning models**
* Integrated **GNN + LSTM + GPR + MHE pipeline** for dynamic, adaptive predictions
* Work is ongoing and **preparation for publication is in progress**

---

## Future Work

* Complete evaluation and **optimization of the hybrid pipeline**
* Extend the approach to **multi-unit SRU datasets**
* Improve computational efficiency for **real-time deployment**
* Prepare findings for **peer-reviewed publication**

```
```
