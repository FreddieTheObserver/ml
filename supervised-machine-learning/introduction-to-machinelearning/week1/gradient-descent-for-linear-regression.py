import math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
      m = x.shape[0]
      cost = 0

      for i in range(m):
            f_wb = w * x[i] + b
            cost = cost + (f_wb - y[i])**2
      total_cost = 1 / (2 * m) * cost
      return total_cost

def compute_gradient(x, y, w, b):
      """
      Computes the gradient for linear regression
      Args:
            x (ndarray (m, )): Data, m examples
            y (ndarray (m, )): target values
            w, b (scalar): model parameters
      Returns
            dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
            dj_db (scalar): The gradient of the cost w.r.t. the parameters b
      """

      # Number of training examples
      m = x.shape[0]
      dj_dw = 0
      dj_db = 0

      for i in range(m):
            f_wb = w * x[i] + b
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = (f_wb - y[i])
            dj_db += dj_db_i
            dj_dw += dj_dw_i
      dj_dw = dj_dw / m
      dj_db = dj_db / m

      return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
      """
      Performs gradient descent to fit w,b. Updates w,b by taking 
      num_iters gradient steps with learning rate alpha
    
      Args:
            x (ndarray (m,))  : Data, m examples 
            y (ndarray (m,))  : target values
            w_in,b_in (scalar): initial values of model parameters  
            alpha (float):     Learning rate
            num_iters (int):   number of iterations to run gradient descent
            cost_function:     function to call to produce cost
            gradient_function: function to call to produce gradient
      
      Returns:
            w (scalar): Updated value of parameter after running gradient descent
            b (scalar): Updated value of parameter after running gradient descent
            J_history (List): History of cost values
            p_history (list): History of parameters [w,b] 
      """

      # An array to store cost J and w's at each iteration primarily for graphing later
      J_history = []
      p_history = []
      b = b_in
      w = w_in

      for i in range(num_iters):
            # Calculate the grandient and update the parameters using gradient_function
            dj_dw, dj_db = gradient_function(x, y, w, b)

            # Update the Parameters using equations
            b = b - alpha * dj_db
            w = w - alpha * dj_dw

            # Save cost J at each iteration
            if i < 100000:
                  J_history.append(cost_function(x, y, w, b))     
                  p_history.append([w, b])
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                  print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                        f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                        f"w: {w: 0.3e}, b:{b: 0.5e}")
      return w, b, J_history, p_history 

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist  = gradient_descent(
      x_train, y_train, w_init, b_init, tmp_alpha,
      iterations, compute_cost, compute_gradient
)
print(f"(w, b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration (start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

# build a 2D grid of (w, b) values around the optimum
w_range = np.arange(0, 400, 5)
b_range = np.arange(-100, 200, 5)
W, B = np.meshgrid(w_range, b_range)

# compute the cost at every grid cell
Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])

print(f"grid shape: {W.shape}, cost range: {Z.min():.1f} to {Z.max():.0f}")

# draw the contour plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.contour(W, B, Z, levels=[5, 10, 50, 100, 500, 1000, 5000, 25000, 100000],
           colors='black', alpha=0.5)
ax.set_title('Cost contour with gradient descent path')
ax.set_xlabel('w')
ax.set_ylabel('b')

# overlay the descent path
p_hist_array = np.array(p_hist)
ax.plot(p_hist_array[:, 0], p_hist_array[:, 1], 'r-', linewidth=1, alpha=0.5)

# add arrows at intervals along the path to show direction
step = 100
for i in range(0, len(p_hist) - step, step):
    ax.annotate('', xy=p_hist_array[i + step], xytext=p_hist_array[i],
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

plt.show()

# --- divergence demo: a too-large learning rate ---
# reinitialize parameters
w_init = 0
b_init = 0
# huge alpha, only a few iterations (cost overflows quickly)
iterations = 10
tmp_alpha = 8.0e-1

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha,
    iterations, compute_cost, compute_gradient
)

# plot the diverging cost on a log scale
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(J_hist, marker='o')
ax.set_yscale('log')
ax.set_title(f'Diverging cost with alpha = {tmp_alpha}')
ax.set_xlabel('iteration step')
ax.set_ylabel('Cost (log scale)')
plt.show()
