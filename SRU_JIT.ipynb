import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load data
input_file = r"C:\Users\Hp\OneDrive\Desktop\Cleaned_OUT_Table.csv"
output_file = r"C:\Users\Hp\OneDrive\Desktop\Cleaned_IN_Table.csv"

data1 = pd.read_csv(input_file, header=None).values
data2 = pd.read_csv(output_file, header=None).values
data = np.hstack((data1, data2))  # Combine horizontally

# Input/output config
input_dim = 5
output_dim = 2

input_matrix = data[:, :input_dim]
output_matrix = data[:, input_dim:input_dim+output_dim]

print(f"The length of output matrix is {len(output_matrix)}")

# Split into 70% train, 30% validation
n_samples = len(output_matrix)
train_size = int(0.7 * n_samples)
all_indices = np.random.permutation(n_samples)
train_idx = all_indices[:train_size]
valid_idx = all_indices[train_size:]

train_input = input_matrix[train_idx]
train_output = output_matrix[train_idx]
valid_input = input_matrix[valid_idx]
valid_output = output_matrix[valid_idx]

train_data = np.hstack((train_input, train_output))
valid_data = np.hstack((valid_input, valid_output))

# Just-in-Time modeling with k-NN + Linear Regression
k = 60
y_predict_lin = np.zeros((len(valid_output), output_dim))

# Mahalanobis distance requires covariance matrix
V = np.cov(train_input.T)

for i in range(len(valid_output)):
    query_pt = valid_input[i].reshape(1, -1)
    
    knn = NearestNeighbors(n_neighbors=k, metric='mahalanobis', metric_params={'V': V})
    knn.fit(train_input)
    _, indices = knn.kneighbors(query_pt)
    
    neighbor_input = train_input[indices[0]]
    neighbor_output = train_output[indices[0]]
    
    if len(neighbor_input) == 0 or len(neighbor_output) == 0:
        y_predict_lin[i, :] = np.nan
        continue

    model1 = LinearRegression().fit(neighbor_input, neighbor_output[:, 0])
    model2 = LinearRegression().fit(neighbor_input, neighbor_output[:, 1])

    y_predict_lin[i, 0] = model1.predict(query_pt)
    y_predict_lin[i, 1] = model2.predict(query_pt)

# Mean squared error
mean_error = np.nanmean((valid_output - y_predict_lin)**2, axis=0)
print("The mean error for trained model is:", mean_error)

# R² score
residual_sum_squares = np.nansum((valid_output - y_predict_lin)**2, axis=0)
total_sum_squares = np.nansum((valid_output - np.nanmean(valid_output, axis=0))**2, axis=0)
R_squared = 1 - residual_sum_squares / total_sum_squares
print("The R-squared for trained model is:", R_squared)

# Error from mean
error = output_matrix - mean_error

# Plot error
plt.figure()
plt.plot(np.arange(len(output_matrix)), error, 'b', linewidth=1.5)
plt.xlabel("Index")
plt.ylabel("Error from Mean Error")
plt.title("Error of each value from the Mean Error")
plt.grid(True)

# True vs predicted for Output 1
plt.figure()
plt.plot(valid_output[:, 0], 'r', label="True - Output 1")
plt.plot(y_predict_lin[:, 0], 'k', label="Predicted - Output 1")
plt.xlabel("Validation Index")
plt.ylabel("Output 1")
plt.title("True vs Predicted - Output 1")
plt.legend()

# True vs predicted for Output 2
plt.figure()
plt.plot(valid_output[:, 1], 'r', label="True - Output 2")
plt.plot(y_predict_lin[:, 1], 'k', label="Predicted - Output 2")
plt.xlabel("Validation Index")
plt.ylabel("Output 2")
plt.title("True vs Predicted - Output 2")
plt.legend()

plt.show()
