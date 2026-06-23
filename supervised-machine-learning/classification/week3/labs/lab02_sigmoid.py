"""
Week 3 - Optional Lab 02: Sigmoid (Logistic) Function
=====================================================

Self-contained, runnable version of the Coursera lab (no `lab_utils_common`,
no `%matplotlib widget`). It:

  1. Implements the sigmoid function and shows how np.exp works.
  2. Prints sigmoid(z) for z = -10 .. 10 and plots the S-curve.
  3. Fits a logistic regression model to the tumor dataset with batch
     gradient descent (the lab's "Run Logistic Regression" button), then
     applies a 0.5 threshold to turn the probabilities into 0/1 predictions.

Run it:

    python lab02_sigmoid.py

Figures are written next to this file in ./figures/ and also shown on screen
(if your matplotlib backend supports a window).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Save figures next to this script, regardless of the current directory.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. The sigmoid function
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Compute the sigmoid of z.

    Args:
        z (ndarray | float): a scalar or numpy array of any size.

    Returns:
        g (ndarray | float): sigmoid(z) = 1 / (1 + e^-z), same shape as z,
                             always in the open interval (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-z))


def draw_vthresh(ax, x):
    """Draw a vertical threshold line at `x` and shade the two prediction
    regions (y=0 to the left, y=1 to the right) -- mirrors the lab helper."""
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color="tab:blue")
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color="tab:orange")
    ax.annotate("z >= 0", xy=(x, 0.5), xytext=(x + 0.5, 0.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="tab:orange"))
    ax.annotate("z < 0", xy=(x, 0.5), xytext=(x - 3.0, 0.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="tab:blue"))
    ax.axvline(x, color="k", linestyle="--", linewidth=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def demo_exp():
    """Show that np.exp works element-wise on arrays and on single numbers."""
    input_array = np.array([1, 2, 3])
    print("Input to exp:", input_array)
    print("Output of exp:", np.exp(input_array))

    input_val = 1
    print("Input to exp:", input_val)
    print("Output of exp:", np.exp(input_val))


def show_sigmoid_table_and_plot():
    """Print sigmoid(z) for z in -10..10 and plot the S-shaped curve."""
    z_tmp = np.arange(-10, 11)
    y = sigmoid(z_tmp)

    np.set_printoptions(precision=3)
    print("\nInput (z), Output (sigmoid(z))")
    print(np.c_[z_tmp, y])

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(z_tmp, y, c="b")
    ax.set_title("Sigmoid function")
    ax.set_ylabel("sigmoid(z)")
    ax.set_xlabel("z")
    draw_vthresh(ax, 0)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lab02_sigmoid_curve.png")
    fig.savefig(out, dpi=110)
    print(f"\nSaved sigmoid curve -> {out}")


# ---------------------------------------------------------------------------
# 2. Logistic regression on the tumor data (the lab's interactive widget,
#    reproduced as a static fit + threshold)
# ---------------------------------------------------------------------------
def fit_logistic(x, y, w=0.0, b=0.0, alpha=0.1, iters=10000):
    """Fit f(x) = sigmoid(w*x + b) with batch gradient descent.

    Single feature, so w and b are scalars. Returns the trained w, b.
    """
    m = x.shape[0]
    for _ in range(iters):
        f = sigmoid(w * x + b)            # predicted probabilities
        err = f - y                        # dJ/dz for log loss
        dw = np.dot(err, x) / m
        db = np.sum(err) / m
        w -= alpha * dw
        b -= alpha * db
    return w, b


def logistic_regression_demo():
    """Fit logistic regression to the tumor data, plot the sigmoid fit and the
    underlying z line, then apply the 0.5 threshold to get 0/1 predictions."""
    x_train = np.array([0.0, 1, 2, 3, 4, 5])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    w, b = fit_logistic(x_train, y_train)
    print(f"\nTrained logistic regression:  w = {w:.3f},  b = {b:.3f}")

    # Probabilities and thresholded predictions for the training points.
    probs = sigmoid(w * x_train + b)
    preds = (probs >= 0.5).astype(int)
    print("x:    ", x_train)
    print("P(y=1):", np.round(probs, 3))
    print("y_hat:", preds, " (threshold 0.5)")
    print("y:    ", y_train, " (true labels)")

    # Smooth curve for plotting the fitted model.
    xs = np.linspace(-1, 6, 200)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(x_train[y_train == 1], y_train[y_train == 1],
               marker="x", c="red", s=80, label="y = 1 (malignant)")
    ax.scatter(x_train[y_train == 0], y_train[y_train == 0],
               marker="o", facecolors="none", edgecolors="blue", s=80,
               label="y = 0 (benign)")
    ax.plot(xs, sigmoid(w * xs + b), c="blue",
            label=r"$f(x)=g(wx+b)$ (probability)")
    ax.plot(xs, w * xs + b, c="orange", linestyle="--",
            label=r"$z = wx+b$ (linear part)")
    ax.axhline(0.5, color="purple", linestyle=":", linewidth=1, label="0.5 threshold")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("tumor size (x)")
    ax.set_ylabel("y / probability")
    ax.set_title("Logistic regression fit (tumor data)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lab02_logistic_fit.png")
    fig.savefig(out, dpi=110)
    print(f"Saved logistic fit -> {out}")


def main():
    print("=" * 60)
    print("Lab 02: Sigmoid / Logistic function")
    print("=" * 60)
    demo_exp()
    show_sigmoid_table_and_plot()
    logistic_regression_demo()
    plt.show()  # no-op under headless backends (e.g. Agg)


if __name__ == "__main__":
    main()
