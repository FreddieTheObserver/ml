# Week 1: Introduction to Machine Learning

## What is Machine Learning?

> "Field of study that gives computers the ability to learn without being explicitly programmed."
> — Arthur Samuel (1959)

## Machine Learning Algorithms

**Primary types:**
- Supervised Learning
- Unsupervised Learning

**Other algorithms:**
- Recommender Systems
- Reinforcement Learning

## Supervised Learning

Supervised learning refers to algorithms that learn **x → y (input → output)** mappings.

- You train the model with labeled examples: input `x` + correct label `y`
- After training, the model takes a new input `x` it has never seen and predicts the output `y`

> ~99% of the economic value created by ML today comes from supervised learning.

### Real-World Examples

| Input (x) | Output (y) | Application |
|---|---|---|
| Email | Spam / Not spam | Spam filter |
| Audio clip | Text transcript | Speech recognition |
| English text | Another language | Machine translation |
| Ad + user info | Click or not | Online advertising |
| Sensor data + image | Position of other cars | Self-driving cars |
| Photo of product | Defect or not | Visual inspection |

### Types of Supervised Learning

#### Regression
Predicting a **number** from infinitely many possible values.

**Example — Housing Price Prediction:**
- Input `x`: size of house (sq ft)
- Output `y`: price (in $1,000s)
- You can fit a **straight line** or a **curve** to the data — choosing the right fit is something the algorithm learns to do systematically

#### Classification

Predicting a **category** from a **small, finite set** of possible outputs (classes/categories).

> Classification predicts discrete categories — not all possible numbers in between like 0.5 or 1.7.

**Example — Breast Cancer Detection**

| Input (x) | Output (y) |
|---|---|
| Tumor size | 0 = Benign, 1 = Malignant |

- The learning algorithm finds a **boundary** that separates classes in the data.
- Unlike regression, there are only a **small number of possible output categories** (e.g. 0 or 1).

**More than two output categories:**

The algorithm can also predict multiple cancer types, e.g.:
- `0` — Benign
- `1` — Malignant Type 1
- `2` — Malignant Type 2

**Multiple inputs:**

You can use more than one input feature to improve predictions:

| Input Features (x) | Output (y) |
|---|---|
| Tumor size + Patient age | 0 = Benign, 1 = Malignant |

With two inputs, the learning algorithm finds a **boundary line** through 2D space to separate classes.

In practice, many more inputs may be used (e.g. tumor clump thickness, cell size uniformity, cell shape uniformity, etc.).

## Unsupervised Learning

In unsupervised learning, data has **no output labels `y`**. The algorithm must find **structure, patterns, or something interesting** in the data entirely on its own.

| | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Data | Labeled (x + y) | Unlabeled (x only) |
| Goal | Predict correct output | Discover structure/patterns |
| Example | Benign vs. Malignant (labeled) | Group patients by similarity (no labels) |

> We call it "unsupervised" because we don't supervise the algorithm with right answers — it figures out patterns by itself.

### Types of Unsupervised Learning

#### Clustering

A **clustering algorithm** groups unlabeled data into clusters (groups) automatically, without being told what the groups are in advance.

**Example 1 — Google News**

- Every day, Google News scans hundreds of thousands of articles on the internet.
- A clustering algorithm groups related stories together based on similar words (e.g. "panda", "twins", "zoo").
- No human tells the algorithm which words define a cluster — it figures this out on its own, every single day.

**Example 2 — DNA Microarray Data**

- Each column = one person's genetic data; each row = a particular gene.
- Colors (red, green, gray, etc.) show the degree to which a gene is active in each individual.
- Clustering groups individuals into types (e.g. Type 1, Type 2, Type 3) based on gene expression patterns.
- The algorithm is not told what the types are in advance — it discovers them automatically.

**Example 3 — Market Segmentation**

- Given a large customer database, clustering can automatically group customers into market segments.
- Example: the deeplearning.ai community was clustered into distinct groups:
  - **Group 1** — Seeking knowledge to grow their skills
  - **Group 2** — Looking to develop their career (promotion, new job)
  - **Group 3** — Wanting to stay updated on how AI impacts their field

### Formal Definition of Unsupervised Learning

- **Supervised learning** — data comes with both inputs `x` and output labels `y`.
- **Unsupervised learning** — data comes with **only inputs `x`**, no labels `y`. The algorithm must find structure, patterns, or something interesting in the data on its own.

#### Anomaly Detection

Used to detect **unusual events** — data points that don't fit normal patterns.

- **Fraud detection** in the financial system — unusual transactions can be signs of fraud.
- Useful across many other applications where rare/abnormal events matter.

#### Dimensionality Reduction

Takes a **big dataset** and compresses it into a **much smaller dataset** while losing as little information as possible.

> Almost magically shrinks data while preserving its essence — useful for visualization, storage, and speeding up other algorithms.

### Check Your Understanding — Supervised vs. Unsupervised

| Problem | Type | Why |
|---|---|---|
| Spam filtering (emails labeled spam / not spam) | **Supervised** | Labeled data: `x` = email, `y` = spam/not spam |
| Grouping news articles (e.g. Google News) | **Unsupervised** | No labels — cluster similar stories |
| Market segmentation from customer data | **Unsupervised** | Algorithm discovers segments automatically |
| Diagnosing diabetes (diabetic / not diabetic) | **Supervised** | Labeled data, just like benign/malignant tumor classification |

### Summary of Unsupervised Learning Types Covered

1. **Clustering** — group similar data points together
2. **Anomaly detection** — find unusual data points
3. **Dimensionality reduction** — compress data while keeping information

> Clustering, anomaly detection, and dimensionality reduction will all be explored more deeply later in the specialization.

---

## Linear Regression Model

**Linear regression** = fitting a **straight line** to your data. It's probably the most widely used learning algorithm in the world today, and the concepts here carry over to other ML models later in the specialization.

### Motivating Example — Predicting House Prices (Portland, USA)

- **Goal**: estimate the price of a client's house based on its size.
- **Dataset**: house sizes and prices from Portland.
- **Plot**: horizontal axis = size (sq ft), vertical axis = price ($1,000s). Each cross = one house.
- **Use**: fit a straight line to the data, then read off the predicted price for a new house (e.g. 1250 sq ft → ~$220,000).

> This is **supervised learning** because the training data includes "right answers" — each house comes with its actual sale price.

### Regression vs. Classification (Recap)

| | Regression | Classification |
|---|---|---|
| Output type | A **number** | A **category** |
| Possible outputs | **Infinitely many** | **Small, finite set** |
| Examples | $220,000 / 1.5 / -33.2 | Cat vs. dog, disease vs. no disease |

> Linear regression is **one example** of a regression model — others exist and will appear in Course 2.

### Training Set

The **training set** is the dataset used to train the model.

- The client's house is **not** in the training set — its price is unknown (that's what we're predicting).
- Workflow: train the model on the training set → use the trained model to predict on new inputs.

### Standard Notation

This notation is standard across AI/ML — you'll see it repeatedly throughout the specialization.

| Symbol | Meaning | Also called |
|---|---|---|
| `x` | Input variable | Feature / input feature |
| `y` | Output variable | Target variable |
| `m` | Total number of training examples | — |
| `(x, y)` | A **single** training example | — |
| `(x⁽ⁱ⁾, y⁽ⁱ⁾)` | The **i-th** training example (row `i`) | — |

**Example — Portland dataset:**

| i | x⁽ⁱ⁾ (size, sq ft) | y⁽ⁱ⁾ (price, $1,000s) |
|---|---|---|
| 1 | 2,104 | 400 |
| ... | ... | ... |
| 47 | ... | ... |

Here `m = 47`, and for the first training example `x⁽¹⁾ = 2104`, `y⁽¹⁾ = 400`.

> ⚠️ **The superscript `(i)` is NOT exponentiation.** `x⁽²⁾` means "the second training example" — not `x²` (x squared). It's just an **index** into the training set.

---

## How Supervised Learning Works

### The Workflow

```
Training set  ──►  Learning algorithm  ──►  Function  f  (the model)
(features x +
 targets y)

New input  x  ──►  f  ──►  ŷ  (prediction / estimate of y)
```

- Feed the **training set** (input features `x` + output targets `y`) into the **learning algorithm**.
- The algorithm produces a **function `f`** — this is the **model**.
- Historically `f` was called a *hypothesis*; in this course we just call it the **function f**.
- Given a new input `x`, the model outputs `ŷ` (read as **"y-hat"**) — the prediction.

### `y` vs. `ŷ` — Target vs. Prediction

| Symbol | Meaning |
|---|---|
| `y` | The **actual true value** (target) — from the training set |
| `ŷ` | The model's **estimate / prediction** for `y` — may or may not equal the true value |

> When helping a client sell a house, the **true price** is unknown until it sells. The model's `ŷ` is the **estimated price** given the house's features.

### How Do We Represent `f`?

A key design question: **what math formula computes `f`?**

For now, we stick with `f` as a **straight line**:

$$
f_{w,b}(x) = wx + b
$$

- `w` and `b` are **numbers** (parameters); their values determine the prediction `ŷ`.
- Shorthand: write `f(x)` instead of `f_{w,b}(x)` — same meaning.
- Plot: `x` on horizontal axis, `y` on vertical axis. The algorithm finds the **best-fit straight line** through the training points.

### Why a Linear Function?

- A line is **simple and easy to work with**.
- It serves as a **foundation** for more complex non-linear models (curves, parabolas, etc.) introduced later.

### Naming the Model

| Name | Meaning |
|---|---|
| **Linear regression** | The model `f(x) = wx + b` |
| **Linear regression with one variable** | A single input feature `x` (e.g. just house size) |
| **Univariate linear regression** | Same thing — *uni* = one (Latin), *variate* = variable |

> Later: **multivariate** linear regression — predicting from multiple features (size, # bedrooms, etc.).

---

## Optional Lab — Notation Cheat Sheet

| Math | Description | Python |
|---|---|---|
| `a` (non-bold) | scalar | `a` |
| **a** (bold) | vector | `a` (numpy array) |
| `x` | training feature values (size in 1000 sqft) | `x_train` |
| `y` | training targets (price in $1000s) | `y_train` |
| `x⁽ⁱ⁾`, `y⁽ⁱ⁾` | i-th training example | `x_i`, `y_i` |
| `m` | number of training examples | `m` |
| `w` | parameter — **weight** | `w` |
| `b` | parameter — **bias** | `b` |
| `f_{w,b}(x⁽ⁱ⁾)` | model output at `x⁽ⁱ⁾` | `f_wb` |

> Implementation lives in [univariate-lr-implementation.py](univariate-lr-implementation.py).

---

## Cost Function

To implement linear regression, the first key step is to define a **cost function** — a way to measure **how well the model is doing** so we can adjust it to do better.

### Recap — Model and Parameters

We have a training set of input features `x` and output targets `y`, and we fit a linear function:

$$
f_{w,b}(x) = wx + b
$$

- `w` and `b` are the **parameters** of the model.
- **Parameters** = variables you can adjust during training to improve the model.
- Also called **coefficients** or **weights**.

### What `w` and `b` Do

Different values of `w` and `b` produce different functions `f(x)` — different lines on the graph.

| `w` | `b` | `f(x)` | Behavior |
|---|---|---|---|
| 0 | 1.5 | `f(x) = 1.5` | Horizontal line — always predicts 1.5. `b` is the **y-intercept** (where the line crosses the y-axis). |
| 0.5 | 0 | `f(x) = 0.5x` | Passes through origin. `w` is the **slope** (rise/run = 0.5). |
| 0.5 | 1 | `f(x) = 0.5x + 1` | Crosses y-axis at `b = 1`. Slope is still `w = 0.5`. |

> `w` controls the **slope**; `b` controls the **y-intercept**.

### Goal of Linear Regression

Choose values of `w` and `b` so that the line `f(x) = wx + b` **fits the training data well** — i.e. passes through or close to the training examples.

### Predictions vs. Targets

| Symbol | Meaning |
|---|---|
| `(x⁽ⁱ⁾, y⁽ⁱ⁾)` | The **i-th training example** — `y⁽ⁱ⁾` is the true target |
| `ŷ⁽ⁱ⁾` | The model's **prediction** for `x⁽ⁱ⁾`: `ŷ⁽ⁱ⁾ = f_{w,b}(x⁽ⁱ⁾) = wx⁽ⁱ⁾ + b` |

**Question:** how do we find `w` and `b` so that `ŷ⁽ⁱ⁾` is close to `y⁽ⁱ⁾` for many (ideally all) training examples?

### Building the Cost Function — Step by Step

1. **Error for one example** — how far off the prediction is from the target:

   ```math
   \text{error}^{(i)} = \hat{y}^{(i)} - y^{(i)}
   ```

2. **Squared error** — square it so positive and negative errors don't cancel out, and large errors are penalized more:

   ```math
   (\hat{y}^{(i)} - y^{(i)})^2
   ```

3. **Sum across all training examples** (`i = 1` to `m`):

   ```math
   \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
   ```

   ⚠️ Problem: more training examples → bigger sum, even if the model is equally good. We don't want the cost to grow with dataset size.

4. **Average squared error** — divide by `m`:

   ```math
   \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
   ```

5. **By convention, divide by `2m`** — the extra factor of 2 makes later derivative calculations cleaner. The cost function works either way.

### The Squared Error Cost Function

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
        = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

- `J(w, b)` is the **cost function**.
- Called the **squared error cost function** because we square the error terms.
- **By far the most common** cost function for linear regression and regression problems in general.

### Intuition

| `J(w, b)` value | Meaning |
|---|---|
| Small | Predictions are close to targets — model fits the data well |
| Large | Predictions are far from targets — model fits the data poorly |

> Our goal: find values of `w` and `b` that **minimize** `J(w, b)`.

---

## Cost Function — Intuition

The cost function `J` measures the gap between the model's predictions and the actual targets. Linear regression's job is to find `w` and `b` that make `J` as small as possible:

$$
\underset{w,b}{\text{minimize}} \; J(w, b)
$$

### Simplified Model — Set `b = 0`

To build intuition visually, drop the `b` parameter (set `b = 0`). The model becomes:

$$
f_w(x) = wx
$$

- Only **one** parameter to tune: `w`.
- The line is forced to pass through the **origin** (when `x = 0`, `f(x) = 0`).
- Cost function simplifies to a function of `w` alone:

$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} \left( w \cdot x^{(i)} - y^{(i)} \right)^2
$$

**Goal:** find the `w` that minimizes `J(w)`.

### Two Graphs Side-by-Side

There are **two different plots** to keep straight — they're related but show different things:

| Plot | Horizontal axis | Vertical axis | What it shows |
|---|---|---|---|
| **Model** `f_w(x)` | `x` (input feature) | `y` (output) | A straight line for a **fixed** `w` |
| **Cost** `J(w)` | `w` (parameter) | `J` (cost) | How cost varies as `w` changes |

> For each value of `w`, you get **one line** on the left plot and **one point** on the right plot.

### Worked Example — Training Set `(1,1), (2,2), (3,3)`

Three points lying perfectly on the line `y = x`. Let's compute `J(w)` for several values of `w`.

#### Case 1 — `w = 1`

The line `f(x) = x` passes through all three points exactly.

| i | x⁽ⁱ⁾ | y⁽ⁱ⁾ | f(x⁽ⁱ⁾) = 1·x⁽ⁱ⁾ | error² |
|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 0² |
| 2 | 2 | 2 | 2 | 0² |
| 3 | 3 | 3 | 3 | 0² |

$$
J(1) = \frac{1}{2 \cdot 3} (0 + 0 + 0) = 0
$$

> Perfect fit → cost is zero.

#### Case 2 — `w = 0.5`

The line `f(x) = 0.5x` has a shallower slope. Each prediction undershoots the target.

| i | x⁽ⁱ⁾ | y⁽ⁱ⁾ | f(x⁽ⁱ⁾) = 0.5·x⁽ⁱ⁾ | error² |
|---|---|---|---|---|
| 1 | 1 | 1 | 0.5 | (0.5 − 1)² = 0.25 |
| 2 | 2 | 2 | 1.0 | (1 − 2)² = 1 |
| 3 | 3 | 3 | 1.5 | (1.5 − 3)² = 2.25 |

$$
J(0.5) = \frac{1}{2 \cdot 3} (0.25 + 1 + 2.25) = \frac{3.5}{6} \approx 0.58
$$

#### Case 3 — `w = 0`

The line `f(x) = 0` is horizontal, sitting on the x-axis.

$$
J(0) = \frac{1}{2 \cdot 3} (1^2 + 2^2 + 3^2) = \frac{14}{6} \approx 2.33
$$

#### Case 4 — `w = −0.5`

Downward-sloping line — predictions are way off.

$$
J(-0.5) \approx 5.25
$$

### Plotting `J(w)`

Repeating this calculation across a range of `w` values traces out the shape of `J(w)`:

| `w` | `J(w)` |
|---|---|
| −0.5 | ~5.25 |
| 0 | ~2.33 |
| 0.5 | ~0.58 |
| 1 | 0 |
| 1.5 | ~0.58 |
| 2 | ~2.33 |

![Side-by-side plot of f_w(x) lines and the resulting J(w) parabola](images/cost_function.png)

The plot is a **U-shaped (parabolic) curve** with its minimum at `w = 1`.

### Key Insight

> Each value of `w` corresponds to:
> - a different **line** on the model plot, **and**
> - a single **point** on the cost plot.
>
> The `w` that minimizes `J(w)` produces the line that best fits the training data.

For this training set, `w = 1` minimizes `J` and gives the best-fitting line — exactly what we'd expect.

### Generalizing Back to Two Parameters

In the full linear regression problem with **both** `w` and `b`:

- Goal: find `w` and `b` that minimize `J(w, b)`.
- Same idea — the cost surface is now 3D (a function of two parameters) instead of a 2D curve.
- Visualized as a **3D bowl-shaped surface** or as **contour plots**.

---

## Cost Function Visualization With `w` and `b`

Previously, we simplified the visualization by setting `b = 0`, so the cost function depended only on `w`:

$$
J(w)
$$

Now return to the full linear regression model:

$$
f_{w,b}(x) = wx + b
$$

The cost function depends on both parameters:

$$
J(w, b)
$$

![Cost function visualizations as a 2D curve, 3D surface, and contour plot](images/cost-function-vis-examples.png)

### Example Model Choice

For the housing-price training set, suppose we choose:

$$
w = 0.06, \quad b = 50
$$

Then:

$$
f(x) = 0.06x + 50
$$

This line is a poor fit for the training data because it consistently underestimates house prices. That means this choice of `w` and `b` gives a relatively large value of `J(w, b)`.

### From a U-Shaped Curve to a 3D Bowl

When there was only one parameter, `w`, the cost function `J(w)` could be drawn as a 2D U-shaped curve.

With two parameters, `w` and `b`, the cost function becomes a 3D surface:

- The horizontal axes are `w` and `b`.
- The vertical axis is the cost `J(w, b)`.
- Each point on the surface represents one particular choice of `w` and `b`.
- The height of the surface at that point is the cost for that choice.

![3D bowl-shaped surface plot of the cost function J(w, b)](images/3d-cost-func.png)

For example, if:

$$
w = -10, \quad b = -15
$$

then the surface height above that point is:

$$
J(-10, -15)
$$

The surface is shaped like a bowl, curved plate, or hammock. The lowest point of the bowl is the minimum cost.

> Linear regression's goal is to find the values of `w` and `b` at the bottom of this bowl.

### Contour Plots

A 3D surface can be hard to read, so we can show the same cost function using a **contour plot**.

A contour plot is like a topographic map:

![Contour plot example showing curves of equal height](images/contour-plot-example.png)

- Imagine slicing the 3D cost surface horizontally at different heights.
- Each slice gives all points with the same cost value.
- Viewed from above, these slices appear as ovals or ellipses.

In the contour plot:

| Axis | Meaning |
|---|---|
| Horizontal axis | `w` |
| Vertical axis | `b` |
| Each oval | All `(w, b)` pairs with the same `J(w, b)` |
| Center of the ovals | Minimum of the cost function |

So, different points on the same oval can have different values of `w` and `b`, and therefore different model lines `f(x)`, but they all have the same cost.

### Key Insight

> A contour plot is a 2D, top-down view of the same 3D cost surface.

The smallest oval is closest to the bottom of the bowl. Its center marks the parameter values that minimize `J(w, b)` and give the best-fitting line for the training data.

---
