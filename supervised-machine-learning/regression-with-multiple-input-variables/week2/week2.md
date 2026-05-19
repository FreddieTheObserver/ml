# Week 2: Regression With Multiple Input Variables

## Multiple Features (Multiple Linear Regression)

Previously: one feature `x` (house size) → predict `y` (price). Model was `f_w,b(x) = wx + b`.

Now: multiple features. For a house we might know **size, # bedrooms, # floors, age**. More information → better predictions.

### Notation

![Housing price data with 4 features (size, bedrooms, floors, age) labeled X_1..X_4. The i=2 row is highlighted, showing x⁽²⁾ = [1416, 3, 2, 40] and x⁽²⁾_3 = 2. Annotations on the right show j = 1..4 and n = 4.](images/multiple-features-notation.png)

| Symbol | Meaning |
|---|---|
| `x_j` | The j-th feature |
| `n` | Total number of features (in this example, `n = 4`) |
| `x⁽ⁱ⁾` | The i-th training example — a **vector** of all n features |
| `x⁽ⁱ⁾_j` | The j-th feature of the i-th training example |

Concrete example with the housing data:

```
x⁽²⁾ = [1416, 3, 2, 40]      ← row vector of all features for example 2
x⁽²⁾_3 = 2                    ← 3rd feature (# floors) of example 2
```

> The arrow `→` is sometimes drawn above a symbol (e.g. `x⃗`) just to emphasize "this is a vector, not a scalar." It's optional notation.

### The Model with Multiple Features

With 4 features:

```
f_w,b(x) = w_1·x_1 + w_2·x_2 + w_3·x_3 + w_4·x_4 + b
```

Generalized to `n` features:

```
f_w,b(x) = w_1·x_1 + w_2·x_2 + ... + w_n·x_n + b
```

### Interpreting the Parameters — Concrete Example

![Linear regression model with multiple features. Top line shows the previous single-feature form f_w,b(x) = wx + b. Middle line shows the 4-feature generalization f_w,b(x) = w_1·x_1 + w_2·x_2 + w_3·x_3 + w_4·x_4 + b. Bottom line shows a concrete example f_w,b(x) = 0.1·x_1 + 4·x_2 + 10·x_3 − 2·x_4 + 80, with each term annotated: x_1 = size, x_2 = #bedrooms, x_3 = #floors, x_4 = years, and 80 = base price.](images/multi-feature-model.png)

Suppose the learned model is:

```
f(x) = 0.1·x_1 + 4·x_2 + 10·x_3 − 2·x_4 + 80
```

(Price in thousands of dollars.)

| Param | Meaning |
|---|---|
| `b = 80` | Base price ≈ \$80,000 (a hypothetical house with size 0, 0 bedrooms, 0 floors, 0 years old) |
| `w_1 = 0.1` | Each additional sqft adds 0.1 × \$1,000 = **\$100** |
| `w_2 = 4` | Each additional bedroom adds **\$4,000** |
| `w_3 = 10` | Each additional floor adds **\$10,000** |
| `w_4 = −2` | Each additional year of age **subtracts \$2,000** |

### Vector Notation — Compact Form

Collect the parameters and features into vectors:

```
w = [w_1, w_2, ..., w_n]      ← vector of weights
x = [x_1, x_2, ..., x_n]      ← vector of features
b                              ← still a single scalar
```

Then the model becomes:

```
f_w,b(x) = w · x + b
```

where `·` is the **dot product** from linear algebra:

```
w · x = w_1·x_1 + w_2·x_2 + w_3·x_3 + ... + w_n·x_n
```

Same math as before, just fewer characters to write. This compact form is also what enables the **vectorization** trick (next lesson) that makes the code dramatically faster.

### Row Vectors vs Column Vectors

A **vector** is just an ordered list of numbers. The same list can be written two ways:

**Row vector** — written horizontally, shape `(1, n)`:

```
x = [ x_1   x_2   x_3   x_4 ]          ← shape (1, 4)
```

**Column vector** — written vertically, shape `(n, 1)`:

```
         ⎡ x_1 ⎤
x  =   ⎢ x_2 ⎥                         ← shape (4, 1)
         ⎢ x_3 ⎥
         ⎣ x_4 ⎦
```

#### Does the orientation matter?

For the **dot product** `w · x`: **no**. As long as both vectors have the same length `n`, you sum the elementwise products either way.

```
w · x = w_1·x_1 + w_2·x_2 + ... + w_n·x_n     (same number, regardless of row/column form)
```

For **matrix multiplication**: **yes**, because the inner dimensions have to match. Convention in linear algebra:

- A single training example's features → usually written as a **column vector** `(n, 1)`
- A whole dataset of `m` examples → a **matrix** `X` of shape `(m, n)`, where each **row** is one example
- Predictions for all `m` examples: `X · w = (m, n) · (n, 1) = (m, 1)` → one prediction per row

That's the shape arithmetic you'll see in the next lessons.

#### In NumPy

NumPy distinguishes three shapes:

| Code | Shape | Name |
|---|---|---|
| `np.array([1, 2, 3])` | `(3,)` | **1D array** — neither row nor column |
| `np.array([[1, 2, 3]])` | `(1, 3)` | Row vector (2D) |
| `np.array([[1], [2], [3]])` | `(3, 1)` | Column vector (2D) |

For most of this course, **1D arrays are fine** — `np.dot` works on them without caring about orientation. The 2D row/column forms only become necessary once you're doing matrix products with `@` or stacking many examples into a feature matrix `X`.

> The transcript's `x⁽²⁾ = [1416, 3, 2, 40]` is a row vector in math notation, but in code it'd be a NumPy 1D array — and the gradient descent code from week 1 works on exactly that shape.

### Naming — Watch Out

| Term | Meaning |
|---|---|
| **Univariate** regression | One input feature (week 1's model) |
| **Multiple** linear regression | Multiple input features (this model) &check; |
| ~~Multivariate regression~~ | Different concept entirely — **don't use this name here** |

> Even though "multiple features" sounds like "multivariate," the term *multivariate regression* refers to predicting multiple outputs at once, not multiple inputs. The course uses **multiple linear regression**.

## Vectorization

![Four-quadrant comparison of vectorization. Top-left: parameters and features — w = [w_1, w_2, w_3], b is a scalar, x = [x_1, x_2, x_3], with NumPy code and a note that linear algebra counts from 1 while Python counts from 0. Bottom-left: without vectorization, hard-coded sum f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b — marked with a sad face since it can't scale to n = 100,000. Top-right: without vectorization, for loop using range(0, n) — marked with a neutral face since it scales but is slow. Bottom-right: vectorization, f = np.dot(w, x) + b — marked with a happy face since it is one line and uses parallel hardware.](images/vectorization-comparison.png)

**Vectorization** = writing code that operates on whole vectors/arrays at once instead of element-by-element. It has **two distinct benefits**:

1. **Shorter code** — easier to write, easier to read.
2. **Much faster execution** — modern numerical libraries (NumPy) and hardware (CPU SIMD, GPU) can perform the operation in parallel instead of one element at a time.

### Setup — Concrete Example

Say `n = 3`, with parameters and features:

```
w = [w_1, w_2, w_3]
x = [x_1, x_2, x_3]
b   (a scalar)
```

In Python with NumPy:

```python
import numpy as np
w = np.array([1.0, 2.5, -3.3])
b = 4.0
x = np.array([10, 20, 30])
```

### Indexing — 1-Based Math vs 0-Based Code

| Math notation | Python code |
|---|---|
| `w_1` (first element) | `w[0]` |
| `w_2` | `w[1]` |
| `w_n` (last element, n = 3) | `w[2]` (i.e. `w[n-1]`) |

Most programming languages — including Python — count from **0**. The math in lectures counts from **1**. Same vector, different indexing convention.

### Three Implementations of `f(x) = w · x + b`

#### 1. Hard-coded sum (no vectorization, no loop)

```python
f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b
```

&cross; Doesn't scale. If `n = 100,000` you'd need 100,000 terms typed out.

#### 2. For loop (no vectorization)

```python
f = 0
for j in range(0, n):       # j = 0, 1, ..., n-1
    f = f + w[j] * x[j]
f = f + b
```

In math, this is the summation:

```
f(x) = ( Σ_{j=1..n}  w_j · x_j ) + b
```

&check; Scales to any `n`. &cross; Still slow — Python executes the loop **sequentially**, one iteration at a time, with per-iteration interpreter overhead.

> Note: `range(0, n)` produces `0, 1, ..., n-1` — it **does not include `n`**. The `0` is usually omitted (`range(n)` means the same thing).

#### 3. Vectorized with `np.dot` (the right way)

```python
f = np.dot(w, x) + b
```

&check; One line. &check; Scales. &check; Fast.

`np.dot(w, x)` computes `w_1·x_1 + w_2·x_2 + ... + w_n·x_n` — exactly the math definition of the dot product.

### Why Vectorized Code Is So Much Faster

`np.dot` doesn't just hide the loop — it dispatches the work to **highly optimized parallel hardware**:

- On a **CPU**: SIMD instructions (Single Instruction, Multiple Data) let one CPU operation multiply 4, 8, or 16 numbers simultaneously.
- On a **GPU**: thousands of small cores can multiply many pairs in parallel — designed originally for graphics, now widely used for ML.

The Python `for` loop in version #2 can use **none** of this. It runs strictly sequentially with overhead from the interpreter on every iteration.

| Implementation | Lines | Scales to large n? | Uses parallel hardware? |
|---|---|---|---|
| Hard-coded sum | n lines | &cross; | &cross; |
| For loop | ~3 lines | &check; | &cross; |
| `np.dot` (vectorized) | 1 line | &check; | &check; |

For small `n` you won't feel the difference. For `n = 10,000` or `n = 1,000,000` (or running gradient descent over millions of examples), the speedup is often **10×–100×** or more.

### Behind the Scenes — Sequential vs Parallel Timeline

To make the speedup concrete, picture the two implementations as a timeline. Imagine `n = 16` so `j` ranges from 0 to 15.

**For loop — sequential.** Each iteration happens at a distinct timestep:

```
t0:  compute w[0]  * x[0]
t1:  compute w[1]  * x[1]
t2:  compute w[2]  * x[2]
...
t15: compute w[15] * x[15]
then: sum them up one addition at a time
```

16 multiplications happen one after another. 16 additions happen one after another. ~32 sequential operations.

**Vectorized — parallel.** A single timestep does the entire computation:

```
t0:  compute  w[0]*x[0], w[1]*x[1], ..., w[15]*x[15]   ALL AT ONCE (parallel multiply)
t1:  sum all 16 products  with specialized add hardware
```

Two steps instead of ~32. The computer's parallel-multiply hardware does the 16 multiplies in one instruction, then specialized "sum-reduction" hardware adds them efficiently rather than carrying out 15 separate additions in sequence.

This is why vectorization scales so well: doubling `n` barely doubles the wall-clock time (modern hardware processes lanes in fixed-width batches), whereas the for loop scales strictly linearly *and* pays Python interpreter overhead per iteration.

### Concrete Example — Vectorizing a Gradient Descent Update

A direct application: updating all `n` parameters in one gradient step.

Say you have **16 features** and you've already computed the 16 derivative terms `d_1, ..., d_16`. Stored as NumPy arrays:

```python
w = np.array([w_1, w_2, ..., w_16])    # shape (16,)
d = np.array([d_1, d_2, ..., d_16])    # shape (16,)
```

The update rule is:

```
w_j := w_j − α · d_j        for j = 1, 2, ..., 16
```

**Without vectorization — hard-coded:**

```python
w[0]  = w[0]  - 0.1 * d[0]
w[1]  = w[1]  - 0.1 * d[1]
# ... 14 more lines ...
w[15] = w[15] - 0.1 * d[15]
```

**Without vectorization — for loop:**

```python
for j in range(0, 16):
    w[j] = w[j] - 0.1 * d[j]
```

**With vectorization — one line:**

```python
w = w - 0.1 * d
```

That single line does **all 16 subtractions in parallel** and assigns the result back to `w` in one step. NumPy broadcasts the scalar `0.1` across the vector `d`, computes the elementwise product, then computes the elementwise subtraction — all using parallel hardware.

> For 16 features, the speedup is mild. For thousands of features over millions of training examples (typical in ML), this is the difference between **a couple of minutes and many hours**.

### Connection to Week 1's Refactor

This is exactly the trick used to rewrite `compute_cost` and `compute_gradient` in [gradient-descent-for-linear-regression.py](../../introduction-to-machinelearning/week1/gradient-descent-for-linear-regression.py):

- `cost = cost + (f_wb - y[i])**2` → `np.dot(err, err)`
- `dj_dw += (f_wb - y[i]) * x[i]` → `np.dot(err, x)`

The parameter update `w = w - alpha * dj_dw` in the existing `gradient_descent` function is *already* the vectorized-update pattern from this lesson — even though `w` was a single scalar in week 1. Once `w` becomes a vector (multiple features, next lesson), the **exact same line** keeps working with no code change. That's the power of vectorized code: it generalizes from `n = 1` to `n = 1,000,000` without rewriting the update rule.

## Gradient Descent for Multiple Linear Regression

Putting it together: **multiple linear regression + vector notation + vectorization** → gradient descent that scales to `n` features with almost no code change from the one-feature version.

### Quick Recap — The Model in Vector Form

| Piece | One feature (week 1) | `n` features (now) |
|---|---|---|
| Parameters | `w` (scalar), `b` (scalar) | `w = [w_1, ..., w_n]` (vector), `b` (scalar) |
| Model | `f_w,b(x) = w·x + b` | `f_w,b(x) = w · x + b` (dot product) |
| Cost | `J(w, b)` | `J(w, b)` — same name, but `w` is now a vector |

`J` still returns a single number; it just takes a vector input now.

### The Update Rule — Side by Side

![Gradient descent update rules compared. Left ("One feature"): repeat { w = w − α · (1/m) Σ (f(x⁽ⁱ⁾) − y⁽ⁱ⁾)·x⁽ⁱ⁾ ; b = b − α · (1/m) Σ (f(x⁽ⁱ⁾) − y⁽ⁱ⁾) } simultaneously update w, b. Right ("n features, n ≥ 2"): repeat { w_1 = w_1 − α · (1/m) Σ (f(x⁽ⁱ⁾) − y⁽ⁱ⁾)·x_1⁽ⁱ⁾ ; ... ; w_n = w_n − α · (1/m) Σ (f(x⁽ⁱ⁾) − y⁽ⁱ⁾)·x_n⁽ⁱ⁾ ; b = b − α · (1/m) Σ (f(x⁽ⁱ⁾) − y⁽ⁱ⁾) } simultaneously update w_j (for j = 1..n) and b. The summation portions are highlighted on both sides to show they are the partial derivatives of J with respect to w (left) and w_1 (right).](images/gradient-descent-comparison.png)

**One feature** (week 1):

```
repeat {
    w := w − α · (1/m) Σ_{i=1..m} (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾) · x⁽ⁱ⁾
    b := b − α · (1/m) Σ_{i=1..m} (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾)
}
simultaneously update w, b
```

**`n` features** (now) — one update per parameter:

```
repeat {
    w_1 := w_1 − α · (1/m) Σ_{i=1..m} (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾) · x⁽ⁱ⁾_1
    w_2 := w_2 − α · (1/m) Σ_{i=1..m} (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾) · x⁽ⁱ⁾_2
    ...
    w_n := w_n − α · (1/m) Σ_{i=1..m} (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾) · x⁽ⁱ⁾_n
    b   := b   − α · (1/m) Σ_{i=1..m} (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾)
}
simultaneously update w_j (for j = 1..n) and b
```

### What Actually Changed?

Almost nothing — and that's the point.

| Piece | One feature | `n` features |
|---|---|---|
| Error term `(f(x⁽ⁱ⁾) − y⁽ⁱ⁾)` | identical | identical |
| Multiplied by | `x⁽ⁱ⁾` (only one feature exists) | `x⁽ⁱ⁾_j` (the j-th feature of example i) |
| Number of `w` updates per step | 1 | `n` |
| `b` update | unchanged | unchanged |

The derivative `∂J/∂w_j` for any j has the exact same shape as the single-feature derivative — it just picks out feature `j` instead of "the only feature."

> **Simultaneous update reminder.** Compute *all* the new values from the *current* `w_1, ..., w_n, b` first, then assign. Mixing old and new values mid-step corrupts the gradient. Same rule as week 1; just more parameters now.

### Why Vectorization Makes This Trivial to Code

Written out as `n` separate lines, the update looks tedious. With vectors and `np.dot`, it collapses to two lines that work for any `n`:

```python
err  = f - y                        # shape (m,) — one error per example
dj_dw = (1/m) * X.T @ err           # shape (n,) — all n partial derivatives at once
dj_db = (1/m) * err.sum()           # scalar
w = w - alpha * dj_dw               # same line as week 1; now updates all n w_j in parallel
b = b - alpha * dj_db
```

The update `w = w - alpha * dj_dw` is **exactly** the line that already works in [gradient-descent-for-linear-regression.py](../../introduction-to-machinelearning/week1/gradient-descent-for-linear-regression.py) — it just operates on a vector now instead of a scalar. No code change, scales from `n = 1` to `n = 1,000,000`.

### Aside — The Normal Equation

There's an alternative to gradient descent that works **only for linear regression**:

```
w, b = (X^T X)^(-1) X^T y           ← solved analytically, no iterations
```

| Pros | Cons |
|---|---|
| One-shot solution — no learning rate, no iterations | Doesn't generalize to logistic regression, neural networks, or any other algorithm |
| | Very slow when `n` is large (matrix inversion is O(n³)) |
| | You should almost never implement it yourself |

Mature ML libraries (e.g. `sklearn.linear_model.LinearRegression`) may use it under the hood for small problems. **Job-interview vocab:** if someone says "normal equation," this is what they mean. Otherwise, prefer gradient descent — it's the one method that carries over to every other algorithm in the course.

### Takeaway

Multiple linear regression with gradient descent is essentially the week 1 algorithm with two changes:

1. The model uses a dot product: `f(x) = w · x + b`.
2. There are now `n` weight updates per iteration instead of 1.

Vectorization makes both changes invisible in code — the update rule is the same line as before.

## Feature Scaling — Why It Matters

When features have **very different ranges**, gradient descent gets slow. Feature scaling rescales the features so they all sit in comparable ranges, which makes gradient descent run much faster.

This lesson is the **motivation** — *why* scaling helps. The *how* (the actual rescaling formulas) comes in the next lesson.

### Feature Range vs Parameter Size — A Concrete Example

Predict house price using two features:

| Feature | Typical range |
|---|---|
| `x_1` = size in sqft | **300 – 2000** (large numbers) |
| `x_2` = # bedrooms | **0 – 5** (small numbers) |

Take a specific house: `x_1 = 2000`, `x_2 = 5`, true price ≈ **$500k**.

**Bad choice of parameters** — `w_1 = 50`, `w_2 = 0.1`, `b = 50`:

```
price = 50 · 2000 + 0.1 · 5 + 50
      = 100,000   + 0.5     + 50
      ≈ $100,000k   (~ $100 million — way off)
```

**Good choice of parameters** — flip them: `w_1 = 0.1`, `w_2 = 50`, `b = 50`:

```
price = 0.1 · 2000 + 50 · 5 + 50
      = 200        + 250    + 50
      = $500k   ✓
```

**The pattern:**

| Feature range | Reasonable parameter |
|---|---|
| **Large** (e.g. `x_1` up to 2000) | **Small** (e.g. `w_1 ≈ 0.1`) |
| **Small** (e.g. `x_2` up to 5) | **Large** (e.g. `w_2 ≈ 50`) |

Intuition: the product `w_j · x_j` has to land in a sensible range (the price). If `x_j` is already huge, `w_j` must be tiny to compensate, and vice versa.

### Why This Hurts Gradient Descent

The mismatch between parameter scales distorts the cost function's geometry.

![Four-panel comparison of feature size and gradient descent. Top-left ("Features" scatterplot, raw): x_1 (size in feet²) on x-axis from 300 to 2000, x_2 (# bedrooms) on y-axis from 0 to 5 — data forms a thin horizontal strip. Top-right ("Parameters" contour plot, raw): cost J(w, b) plotted over w_1 (narrow, 0 to 1) and w_2 (wide, 10 to 100) — contours are tall, skinny ellipses with red gradient-descent arrows zig-zagging back and forth across the narrow axis before reaching the center. Bottom-left ("Features" scatterplot, rescaled): both axes now span 0 to 1, data forms a roughly square cloud. Bottom-right ("Parameters" contour plot, rescaled): cost contours are now roughly concentric circles, with red gradient-descent arrows pointing in a direct path to the minimum.](images/feature-size-and-gradient-descent.png)

**Scatter plot of the raw data** (`x_1` vs `x_2`, top-left):
- Horizontal axis (`x_1`, sqft) stretches from ~300 to ~2000
- Vertical axis (`x_2`, bedrooms) only spans 0 to 5

So the data is stretched wide and short.

**Contour plot of `J(w_1, w_2)`** (top-right) has the **opposite** stretch — tall and skinny:
- `w_1`-axis is narrow (e.g. 0 to 1) — because `w_1` multiplies a huge feature, a tiny change to `w_1` swings the prediction (and the cost) a lot
- `w_2`-axis is wide (e.g. 10 to 100) — because `w_2` multiplies a small feature, you need a much bigger change to `w_2` to move the cost

**Gradient descent on tall-skinny contours** bounces back and forth across the narrow axis instead of heading straight to the minimum — it takes many iterations to converge.

### What Scaling Fixes

Rescale so `x_1` and `x_2` both fall in, say, **0 to 1** (bottom-left). Now:

- Both axes of the data scatter plot are on comparable scales
- The cost contours become **roughly circular** instead of tall-skinny ellipses (bottom-right)
- Gradient descent walks in a much more **direct path** to the global minimum

Same data, same algorithm — just rescaled axes. Gradient descent converges **much faster**.

### Takeaway

| Situation | Cost contours | Gradient descent |
|---|---|---|
| Features on very different scales | Tall, skinny ellipses | Bounces around — slow |
| Features rescaled to comparable ranges | Roughly circular | Direct path — fast |

> The model itself isn't broken without scaling — given enough iterations, gradient descent will still find good parameters. Scaling just gets you there **much faster** by reshaping the cost surface into something easier to descend. Next lesson: the actual rescaling formulas.

## Feature Scaling — How to Do It

Three common methods, all working on the same example (`x_1` = sqft in [300, 2000], `x_2` = bedrooms in [0, 5]).

### Method 1 — Divide by the Max

![Feature scaling by dividing by the max. Top-left: scatter plot of raw data with x_1 (size in feet²) ranging 300 to 2000 and x_2 (# bedrooms) ranging 0 to 5 — the data forms a thin horizontal strip. Right side shows the formulas: x_1,scaled = x_1 / 2000 (max), x_2,scaled = x_2 / 5 (max), with resulting ranges 0.15 ≤ x_1,scaled ≤ 1 and 0 ≤ x_2,scaled ≤ 1. Bottom-left: rescaled scatter plot where both axes now span comparable 0-to-1 ranges, producing a roughly square cloud of points.](images/feature-scaling-max.png)

```
x_1,scaled = x_1 / max(x_1)        → 0.15 ≤ x_1,scaled ≤ 1
x_2,scaled = x_2 / max(x_2)        → 0    ≤ x_2,scaled ≤ 1
```

Cheapest method. Output is always in `[min/max, 1]` — always non-negative.

### Method 2 — Mean Normalization

Center the features around **zero** so they take on both positive and negative values, roughly in `[−1, +1]`.

![Mean normalization. Top-left: raw data scatter, with annotations μ_1 = 600 (average of x_1) and μ_2 = 2.3 (average of x_2). Right side shows the formulas: x_1 = (x_1 − μ_1) / (2000 − 300) and x_2 = (x_2 − μ_2) / (5 − 0), where the denominator is max − min. Resulting ranges: −0.18 ≤ x_1 ≤ 0.82 and −0.46 ≤ x_2 ≤ 0.54. Bottom-left: normalized scatter plot now centered around the origin, with both axes spanning roughly −1 to 1.](images/mean-normalization.png)

```
x_j,normalized = (x_j − μ_j) / (max_j − min_j)
```

where `μ_j` is the **mean** (average) of feature `j` across the training set.

For the housing example with `μ_1 = 600`, `μ_2 = 2.3`:

```
x_1,normalized = (x_1 − 600) / (2000 − 300)    → −0.18 ≤ x_1,norm ≤ 0.82
x_2,normalized = (x_2 − 2.3) / (5    − 0)      → −0.46 ≤ x_2,norm ≤ 0.54
```

Cloud of data points is now centered around the origin.

### Method 3 — Z-Score Normalization

Same idea as mean normalization, but the denominator is the **standard deviation** `σ` instead of `max − min`.

![Z-score normalization. Top-left: raw data scatter with annotations μ_1 = 600, μ_2 = 2.3, σ_1 = 450, σ_2 = 1.4, plus a small bell-curve diagram illustrating that σ is the standard deviation of a normal distribution. Right side shows the formulas: x_1 = (x_1 − μ_1) / σ_1 and x_2 = (x_2 − μ_2) / σ_2. Resulting ranges: −0.67 ≤ x_1 ≤ 3.1 and −1.6 ≤ x_2 ≤ 1.9. Bottom-left: normalized scatter plot with both axes spanning roughly −3 to 3, centered at the origin.](images/z-score-normalization.png)

```
x_j,normalized = (x_j − μ_j) / σ_j
```

For the housing example with `μ_1 = 600, σ_1 = 450` and `μ_2 = 2.3, σ_2 = 1.4`:

```
x_1,normalized = (x_1 − 600) / 450    → −0.67 ≤ x_1,norm ≤ 3.1
x_2,normalized = (x_2 − 2.3) / 1.4    → −1.6  ≤ x_2,norm ≤ 1.9
```

> **What's `σ`?** The **standard deviation** — a measure of how spread-out the values are around the mean. If you know the normal (bell-curve / Gaussian) distribution, this is its width parameter. If not, you don't need to know the formula for this course — `np.std(x)` computes it.

### Quick Comparison

| Method | Formula | Typical output range | Centered at 0? |
|---|---|---|---|
| Divide by max | `x / max` | `[min/max, 1]` | &cross; |
| Mean normalization | `(x − μ) / (max − min)` | roughly `[−1, +1]` | &check; |
| Z-score normalization | `(x − μ) / σ` | roughly `[−3, +3]` | &check; |

All three shrink the range and equalize feature scales. Z-score is the most common in practice.

### Rule of Thumb — When to Scale

Aim for each feature to land roughly in `[−1, +1]`. The bounds are **loose** — close is good enough.

| Feature range | Need to scale? |
|---|---|
| `−1 to +1`, `−3 to +3`, `−0.3 to +0.3` | Fine as-is |
| `0 to 3`, `−2 to +0.5` | Fine — but no harm rescaling |
| `−100 to +100` (e.g. large raw values) | **Yes** — rescale toward `[−1, +1]` |
| `−0.001 to +0.001` (tiny values) | **Yes** — rescale up |
| `98.6 to 105` (e.g. body temp in °F — values cluster around 100) | **Yes** — gradient descent will be slow otherwise |

> There's almost never any harm in rescaling. **When in doubt, do it.** The cost is a few lines of code; the benefit can be 10× faster training.

### NumPy Sketch

```python
# Z-score (most common)
mu    = X.mean(axis=0)        # mean of each column → shape (n,)
sigma = X.std(axis=0)         # std of each column  → shape (n,)
X_norm = (X - mu) / sigma     # broadcasts: (m, n) − (n,) / (n,) → (m, n)

# Mean normalization
X_norm = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Divide by max
X_norm = X / X.max(axis=0)
```

> **Important:** save `μ` and `σ` (or whatever statistics you used) from the **training set** and reuse them to scale any new data at prediction time. Don't recompute them on the test set or on a single new example — that would change the meaning of the scaled values.
