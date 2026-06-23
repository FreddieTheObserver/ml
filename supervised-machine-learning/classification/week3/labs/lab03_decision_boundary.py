"""
Week 3 - Optional Lab 03: Logistic Regression, Decision Boundary
================================================================

Self-contained, runnable version of the Coursera lab (no `lab_utils_common`,
no `%matplotlib widget`). It:

  1. Defines the small 2-feature dataset and plots it (red x = 1, blue o = 0).
  2. Recaps the sigmoid function with a threshold plot at z = 0.
  3. Plots the decision boundary for the trained model f(x) = g(-3 + x0 + x1),
     i.e. the line x1 = 3 - x0, shading the region predicted as
      y = 0.

Run it:

    python lab03_decision_boundary.py

Figures are written next to this file in ./figures/ and also shown on screen
(if your matplotlib backend supports a window).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Local helpers (replacements for the Coursera `lab_utils_common` imports)
# ---------------------------------------------------------------------------
def sigmoid(z):
    """sigmoid(z) = 1 / (1 + e^-z), in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def plot_data(X, y, ax):
    """Plot 2-feature data: y=1 as red crosses, y=0 as blue circles."""
    y = y.reshape(-1)
    pos = y == 1
    neg = y == 0
    ax.scatter(X[pos, 0], X[pos, 1], marker="x", c="red", s=80, label="y = 1")
    ax.scatter(X[neg, 0], X[neg, 1], marker="o", facecolors="none",
               edgecolors="blue", s=80, label="y = 0")


def draw_vthresh(ax, x):
    """Vertical threshold line at z = x, shading the two prediction regions."""
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color="tab:blue")
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color="tab:orange")
    ax.axvline(x, color="k", linestyle="--", linewidth=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)


def plot_dataset():
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_data(X, y, ax)
    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_0$")
    ax.set_title("Training data")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lab03_data.png")
    fig.savefig(out, dpi=110)
    print(f"Saved dataset plot -> {out}")


# ---------------------------------------------------------------------------
# 2. Sigmoid refresher
# ---------------------------------------------------------------------------
def plot_sigmoid_refresher():
    z = np.arange(-10, 11)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(z, sigmoid(z), c="b")
    ax.set_title("Sigmoid function")
    ax.set_ylabel("sigmoid(z)")
    ax.set_xlabel("z")
    draw_vthresh(ax, 0)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lab03_sigmoid.png")
    fig.savefig(out, dpi=110)
    print(f"Saved sigmoid refresher -> {out}")


# ---------------------------------------------------------------------------
# 3. Decision boundary for f(x) = g(-3 + x0 + x1)
# ---------------------------------------------------------------------------
def plot_decision_boundary():
    """The trained model predicts y=1 when -3 + x0 + x1 >= 0.
    The boundary -3 + x0 + x1 = 0 is the line x1 = 3 - x0."""
    # A couple of sanity-check predictions printed to the console.
    w = np.array([1.0, 1.0])
    b = -3.0
    for xi in (np.array([1.0, 1.0]), np.array([2.0, 2.0])):
        z = np.dot(w, xi) + b
        print(f"x={xi}  z={z:+.1f}  P(y=1)={sigmoid(z):.3f}  "
              f"-> y_hat={int(sigmoid(z) >= 0.5)}")

    x0 = np.arange(0, 6)
    x1 = 3 - x0  # boundary line

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(x0, x1, c="b", label=r"$x_1 = 3 - x_0$ (boundary)")
    ax.axis([0, 4, 0, 3.5])
    ax.fill_between(x0, x1, alpha=0.2)  # region below the line -> predict 0
    plot_data(X, y, ax)
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_0$")
    ax.set_title("Decision boundary:  f(x) = g(-3 + x0 + x1)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lab03_decision_boundary.png")
    fig.savefig(out, dpi=110)
    print(f"Saved decision boundary -> {out}")


def main():
    print("=" * 60)
    print("Lab 03: Logistic Regression, Decision Boundary")
    print("=" * 60)
    plot_dataset()
    plot_sigmoid_refresher()
    print("\nModel f(x) = g(-3 + x0 + x1) predicts y=1 when -3 + x0 + x1 >= 0:")
    plot_decision_boundary()
    plt.show()  # no-op under headless backends (e.g. Agg)


if __name__ == "__main__":
    main()
