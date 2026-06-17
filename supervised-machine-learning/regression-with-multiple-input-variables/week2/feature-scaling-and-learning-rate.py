import math
import copy
import numpy as np
import matplotlib.pyplot as plt 

X_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852,  2, 1, 35],
    [1534, 3, 2, 30],
    [1940, 4, 2, 15],
    [2400, 4, 1, 10],
    [1100, 2, 1, 50],
    [1860, 3, 2, 25],
    [1300, 3, 1, 38],
    [2200, 4, 2, 8],
])
y_train = np.array([460, 232, 178, 315, 420, 540, 195, 390, 245, 510])

X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Peak-to-peak range per feature: {np.ptp(X_train, axis=0)}")

def compute_cost(X, y, w, b):
      """
      Compute the cost for linear regression
      Args:
            X (ndarray): Shape (m,n) matrix of examples with n features
            y (ndarray): Shape (m,) vector of target values
            w (ndarray): Shape (n,) vector of model parameters
            b (scalar): Model parameter
      Returns:
            cost (scalar): The cost of the model
      """
      m = X.shape[0]
      f_wb = X @ w + b
      err = f_wb - y 
      total_cost = (1 / (2 * m)) * np.dot(err, err) 
      return total_cost

def compute_gradient(X, y, w, b):
      """
      Compute the gradient for linear regression
      Args:
            X (ndarray): Shape (m,n) matrix of examples with n features
            y (ndarray): Shape (m,) vector of target values
            w (ndarray): Shape (n,) vector of model parameters
            b (scalar): Model parameter
      Returns:
            dj_dw (ndarray): Shape (n,) vector of gradient of the cost with respect to the parameters w
            dj_db (scalar): Gradient of the cost with respect to the parameter b
      """ 
      m = X.shape[0]
      f_wb = X @ w + b 
      err = f_wb - y  
      dj_dw = (X.T @ err) / m
      dj_db = np.sum(err) / m
      return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
      """
      Run batch gradient descent.

      Returns:
            w, b : learned parameters
            J_history : History of cost values
      """
      J_history = []
      w = copy.deepcopy(w_in)
      b = b_in

      for i in range(num_iters):
            dj_dw, dj_db = gradient_function(X, y, w, b)

            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            J_history.append(cost_function(X, y, w, b))

            if i % math.ceil(num_iters / 10) == 0:
                  cw = ", ".join(f"{v: 0.3e}" for v in w)
                  print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2e}  "
                        f"dj_dw: [{', '.join(f'{v: 0.3e}' for v in dj_dw)}]  "
                        f"dj_db: {dj_db: 0.3e}  "
                        f"w: [{cw}]  b: {b: 0.5e}")
      return w, b, J_history

def run_gradient_descent(X, y, iterations, alpha):
    """Convenience wrapper: zero-init w and b, run GD, return everything."""
    m, n = X.shape
    initial_w = np.zeros(n)
    initial_b = 0.0

    w_out, b_out, J_hist = gradient_descent(
        X, y, initial_w, initial_b,
        compute_cost, compute_gradient,
        alpha, iterations,
    )
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.4f}")
    return w_out, b_out, J_hist

print("\n--- alpha = 9.9e-7 (expect divergence) ---")
_w, _b, _ = run_gradient_descent(X_train, y_train, iterations=10, alpha=9.9e-7)

print("\n--- alpha = 1e-7 (slow but stable convergence) ---")
w_slow, b_slow, J_slow = run_gradient_descent(X_train, y_train, iterations=1000, alpha=1e-7)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(J_slow)
ax[0].set_title("Cost vs iteration (alpha=1e-7)")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("cost")

ax[1].plot(100 + np.arange(len(J_slow[100:])), J_slow[100:])
ax[1].set_title("Cost vs iteration (zoomed, after iter 100)")
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("cost")
fig.suptitle("Unscaled features: fast initial drop, painfully slow tail")
plt.tight_layout()
plt.show()

def zscore_normalize_features(X):
    """
    Z-score normalize each feature column.

    Returns:
        X_norm (ndarray): normalized features, shape (m, n)
        mu (ndarray): mean of each feature, shape (n,)
        sigma (ndarray): standard deviation of each feature, shape (n,)
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

print("\n--- z-score normalization ---")
X_norm, mu, sigma = zscore_normalize_features(X_train)
print(f"mu: {mu}")
print(f"sigma: {sigma}")
print(f"Peak-to-peak range per normalized feature: {np.ptp(X_norm, axis=0)}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[0].set_title("Raw features")

ax[1].scatter(X_norm[:, 0], X_norm[:, 3])
ax[1].set_xlabel(X_features[0])
ax[1].set_ylabel(X_features[3])
ax[1].set_title("Z-score normalized")
fig.suptitle("Size vs age: thin strip → balanced cloud")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 4, figsize=(14, 6))
for i in range(4):
    ax[0, i].hist(X_train[:, i], bins=8, edgecolor="black")
    ax[0, i].set_title(X_features[i])
    ax[1, i].hist(X_norm[:, i], bins=8, edgecolor="black")
    ax[1, i].set_title(f"{X_features[i]} (normalized)")
ax[0, 0].set_ylabel("raw")
ax[1, 0].set_ylabel("normalized")
fig.suptitle("Before / after z-score normalization")
plt.tight_layout()
plt.show()

print("\n--- alpha = 1e-1 on z-score normalized features ---")
w_norm, b_norm, J_norm = run_gradient_descent(X_norm, y_train, iterations=1000, alpha=1.0e-1)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(J_norm)
ax.set_title("Cost vs iteration (alpha=0.1, normalized features)")
ax.set_xlabel("iteration")
ax.set_ylabel("cost")
plt.tight_layout()
plt.show()

x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - mu) / sigma
price = np.dot(x_house_norm, w_norm) + b_norm
print(f"\n1200 sqft, 3 bedroom, 1 floor, 40 years old — predicted price: {price:0.1f} thousand dollars")
print(f"  (${price * 1000:0.0f})")