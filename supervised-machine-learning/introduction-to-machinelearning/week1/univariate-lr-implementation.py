import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Training data
# -----------------------------------------------------------------------------

# x_train is the input variable: size in 1000 square feet.
# y_train is the target variable: price in 1000s of dollars.
x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([300.0, 500.0, 680.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


# -----------------------------------------------------------------------------
# 2. Number of training examples
# -----------------------------------------------------------------------------

print(f"x_train.shape: {x_train.shape}")

m = x_train.shape[0]
print(f"Number of training examples is: {m}")


# -----------------------------------------------------------------------------
# 3. Access one training example
# -----------------------------------------------------------------------------

i = 1
x_i = x_train[i]
y_i = y_train[i]

print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# -----------------------------------------------------------------------------
# 4. Plot the training data
# -----------------------------------------------------------------------------

plt.scatter(x_train, y_train, marker="x", c="r")

plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (in 1000s of dollars)")

plt.show()


# -----------------------------------------------------------------------------
# 5. Set model parameters
# -----------------------------------------------------------------------------

w = 200
b = 100

print(f"w: {w}")
print(f"b: {b}")


# -----------------------------------------------------------------------------
# 6. Compute model output with a loop
# -----------------------------------------------------------------------------

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model.

    Args:
        x (ndarray (m,)): Data, m examples
        w, b (scalar): Model parameters

    Returns:
        f_wb (ndarray (m,)): Model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b)


# -----------------------------------------------------------------------------
# 7. Plot the model prediction
# -----------------------------------------------------------------------------

plt.plot(x_train, tmp_f_wb, c="b", label="Our Prediction")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")

plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (in 1000s of dollars)")
plt.legend()

plt.show()


# -----------------------------------------------------------------------------
# 8. Make a prediction for a new house
# -----------------------------------------------------------------------------

x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"{cost_1200sqft:.0f} thousand dollars")


# -----------------------------------------------------------------------------
# 9. Plot the new prediction with the model line
# -----------------------------------------------------------------------------

tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c="b", label="Our Prediction")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")
plt.scatter(x_i, cost_1200sqft, marker="o", c="g", label="Prediction (1200 sqft)")

plt.title("Housing Prices - With New Prediction")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (in 1000s of dollars)")
plt.legend()

plt.show()


# -----------------------------------------------------------------------------
# 10. Vectorized model output
# -----------------------------------------------------------------------------

def compute_model_output_vectorized(x, w, b):
    """
    Computes the prediction of a linear model using vectorized NumPy operations.

    Args:
        x (ndarray (m,)): Data, m examples
        w, b (scalar): Model parameters

    Returns:
        f_wb (ndarray (m,)): Model prediction
    """
    f_wb = w * x + b
    return f_wb


tmp_f_wb_loop = compute_model_output(x_train, w, b)
tmp_f_wb_vec = compute_model_output_vectorized(x_train, w, b)

print(np.array_equal(tmp_f_wb_loop, tmp_f_wb_vec))


# -----------------------------------------------------------------------------
# 11. Compute cost
# -----------------------------------------------------------------------------

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): Target values
        w, b (scalar): Model parameters

    Returns:
        total_cost (float): Cost of using w and b to fit the data points
    """
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost
