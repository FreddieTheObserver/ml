import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([300.0, 500.0, 680.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

i = 1 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title 
plt.title("Housing Prices")
# Set the  y-axis label
plt.ylabel('Price in (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')

plt.show()

w = 200
b = 100     
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
      """
      Computes the prediction of a linear model
      Args:
            x (ndarray (m,)): Data, m examples
            w, b (scalar)   : model parameters
      Returns
            f_wb (ndarray (m,)): model prediction
      """
      m = x.shape[0]
      f_wb = np.zeros(m)
      for i in range(m):
            f_wb[i] = w * x[i] + b
      return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot out model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# Prediction for a new house
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"{cost_1200sqft:.0f} thousand dollars")

# Recompute the model line and plot everything together
tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.scatter(x_i, cost_1200sqft, marker='o', c='g', label='Prediction (1200 sqft)')

plt.title("Housing Prices — With New Prediction")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()

## Vectorize compute_model_output
def compute_model_output_vectorized(x, w, b):
      """
      Computes the prediction of a linear model
      Args:
            x (ndarray (m,)): Data, m examples
            w, b (scalar): model parameters
      Returns:
            f_wb (ndarray (m,)): model prediction
      """
      f_wb = w * x + b
      return f_wb

tmp_f_wb_loop = compute_model_output(x_train, w, b)
tmp_f_wb_vec = compute_model_output_vectorized(x_train, w, b)

print(np.array_equal(tmp_f_wb_loop, tmp_f_wb_vec))