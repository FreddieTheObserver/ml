import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=2)

X_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35],
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

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

print(f"Peak-to-peak range, raw        X: {np.ptp(X_train, axis=0)}")
print(f"Peak-to-peak range, normalized X: {np.ptp(X_norm, axis=0)}")

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm, y_train)

print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")


b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w = {w_norm}, b = {b_norm}")

y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm

print(f"predict() vs manual w·x+b match: {(y_pred == y_pred_sgd).all()}")
print(f"Prediction on training set:\n{y_pred[:4]}")
print(f"Target values:\n{y_train[:4]}")

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='target')
    ax[i].scatter(X_train[:, i], y_pred, color='orange', label='predict')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('Price')
ax[0].legend()
fig.suptitle('target vs prediction using z-score normalized model')
plt.tight_layout()
plt.show()