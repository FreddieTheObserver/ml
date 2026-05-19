import math
import numpy as np 
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852,  2, 1, 35]])
y_train = np.array([460, 232, 178]) 

m = X_train.shape[0]
n = X_train.shape[1]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"m: {m}, n: {n}")

def predict_single_loop(x, w, b):
      """
      Single predict using linear regression

      Args:
            x (ndarray): Shape (n,) example with multiple features
            w (ndarray): Shape (n,) model parameters
            b (scalar):  model parameter

      Returns:
            p (scalar): prediction
      """
      n = x.shape[0]
      p = 0

      for j in range(n):
            p_j = w[j] * x[j]
            p = p + p_j
      p = p + b 
      return p

def predict(x, w, b):
      """
      Single predict using linear regression

      Args:
            x (ndarray): Shape (n,) example with multiple features
            w (ndarray): Shape (n,) model parameters
            b (scalar):  model parameter

      Returns:
            p (scalar): prediction
      """
      p = np.dot(w, x) + b 
      return p

w_init = np.zeros(n)
b_init = 0
x_vec = X_train[0, :]

f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb: {f_wb}")

f_wb_vec = predict(x_vec, w_init, b_init)
print(f"f_wb_vec: {f_wb_vec}")

def compute_cost(X, y, w, b):
      """
      Compute cost for linear regression with multiple variables

      Args:
            X (ndarray): Shape (m, n), m examples with n features
            y (ndarray): Shape (m,), target values
            w (ndarray): Shape (n,), model parameters
            b (scalar):  model parameter

      Returns:
            total_cost (float): cost
      """
      m = X.shape[0]

      f_wb = X @ w + b
      err = f_wb - y 
      total_cost = (1 / (2 * m)) * np.dot(err, err)
      
      return total_cost

cost = compute_cost(X_train, y_train, np.zeros(n), 0)
print(f"Cost at initial w and b: {cost}")


def compute_gradient(X, y, w, b):
      """
      Computes the gradient for linear regression with multiple variables

      Args:
            X (ndarray): Shape (m, n), m examples with n features
            y (ndarray): Shape (m,), target values
            w (ndarray): Shape (n,), model parameters
            b (scalar):  model parameter

      Returns:
            dj_dw (ndarray): Shape (n,), gradient of cost with respect to w
            dj_db (scalar): gradient of cost with respect to b
      """
      m = X.shape[0]

      f_wb = X @ w + b
      err = f_wb - y
      
      dj_dw = (X.T @ err) / m
      dj_db = np.sum(err) / m

      return dj_dw, dj_db

dj_dw, dj_db = compute_gradient(X_train, y_train, np.zeros(n), 0)

print(f"dj_dw: {dj_dw}")
print(f"dj_dw shape: {dj_dw.shape}")
print(f"dj_db: {dj_db}")

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
      """
      Performs gradient descent to fit w,b

      Args:
            X (ndarray): Shape (m, n), m examples with n features
            y (ndarray): Shape (m,), target values
            w_in (ndarray): Shape (n,), initial model parameters
            b_in (scalar): initial model parameter
            alpha (float): learning rate
            num_iters (int): number of iterations
            cost_function: function to call to produce cost
            gradient_function: function to call to produce gradient

      Returns:
            w (ndarray): Shape (n,), updated model parameters
            b (scalar): updated model parameter
            J_history (list): history of cost values
      """
      J_history = []
      w = copy.deepcopy(w_in)
      b = b_in

      for i in range(num_iters):
            dj_dw, dj_db = gradient_function(X, y, w, b)

            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            if i < 100000:
                  J_history.append(cost_function(X, y, w, b))
            
            if i % math.ceil(num_iters / 10) == 0:
                  print(f"Iteration {i:4}: Cost {J_history[-1]:8.2f}")
      return w, b, J_history

initial_w = np.zeros(n)
initial_b = 0
iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(
      X_train, y_train, initial_w, initial_b,
      alpha, iterations, compute_cost, compute_gradient
)

print(f"w_final: {w_final}")
print(f"b_final: {b_final}")
print(f"Final cost: {J_hist[-1]}")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(J_hist)
ax.set_title("Cost vs. iteration")
ax.set_xlabel("iteration step")
ax.set_ylabel("Cost")
plt.show()

x_house = np.array([1200, 3, 1, 40])
x_house_predict = predict(x_house, w_final, b_final)

print(f"1200 sqft, 3 bedroom, 1 floor, 40 years old house prediction: {x_house_predict:0.1f} Thousand dollars")