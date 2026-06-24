# Week 3: Classification

> Week 2 predicted a **number** (linear regression). Week 3 predicts a **category** — the output `y` can only be one of a small, finite set of values. We first see *why linear regression is a poor fit for classification*, then build the algorithm we actually use: **logistic regression**.

## What Is Classification?

In **classification**, the output `y` takes on only one of a **small handful** of possible values — not any number in an infinite range.

When there are only **two** possible outputs, it's called **binary classification** ("binary" = two possible classes/categories).

> "Class" and "category" mean the same thing here and are used interchangeably.

### Examples of Binary Classification Problems

| Question (input → output) | Output values |
|---|---|
| Is this email **spam**? | no / yes |
| Is this online financial transaction **fraudulent**? (e.g. stolen credit card) | no / yes |
| Is this tumor **malignant**? | no / yes |

In each case the answer is one of **two** values: **no** or **yes**.

### Naming the Two Classes

The same two classes can be written several equivalent ways:

| "No" class | "Yes" class |
|---|---|
| no | yes |
| false | true |
| **0** | **1** |
| **negative class** | **positive class** |

- The course mostly uses the numbers **0 and 1** for `y`, because that fits most cleanly with the learning algorithms we implement. In conversation we still say "no/yes" or "false/true."
- `0` / `false` is the **negative class**; `1` / `true` is the **positive class** (the convention `0 = false`, `1 = true` comes from computer science).

> **Negative ≠ bad, positive ≠ good.** They convey **absence (0)** vs **presence (1)** of the thing you're looking for — the absence/presence of spam-ness, of fraud, of malignancy. A spam email is a *positive* example (spam is present); a normal email is a *negative* example.

### Which Class Is 0 and Which Is 1 Is Arbitrary

Either assignment can work. One engineer might make "spam present" the positive class; another might flip it and make "a good/real email" the positive class. The math doesn't care — pick a convention and stay consistent.

## Why Not Just Use Linear Regression?

It's natural to ask: we already know linear regression — can't we just apply it to a classification problem? Let's try, and watch it break.

### Setup — The Tumor Example

Training set: classify a tumor as **malignant (1)** or **benign (0)** from its **size**.

- Horizontal axis: tumor size (the feature `x`)
- Vertical axis: the label `y`, now plotted as **0 or 1**

> In Week 1 we visualized classification on a 1-D number line. Same data — now we put the two classes at heights `0` and `1` on the vertical axis.

```
y
1 |                 ●   ●   ●   ●     ← malignant examples
  |
  |
0 | ●   ●   ●                         ← benign examples
  +--------------------------------► tumor size (x)
```

### Step 1 — Fit a Straight Line

Apply linear regression and fit `f(x) = wx + b` to this data:

```
y
1 |                 ●   ●  ●   ● 
  |              ___/
0.5·············/···················  ← threshold
  |         ___/
0 | ●  ● ●_/
  +--------------------------------► tumor size (x)
```

Linear regression doesn't just output 0 and 1 — it outputs **all numbers in between** (and even below 0 or above 1). But we need a **category**, so we add a decision rule.

### Step 2 — Pick a Threshold (0.5)

Convert the continuous output into a 0/1 prediction with a cutoff at **0.5**:

```
if f(x) <  0.5   →   predict ŷ = 0   (benign / not malignant)
if f(x) ≥  0.5   →   predict ŷ = 1   (malignant)
```

The threshold `0.5` crosses the fitted line at one point. Drop a vertical line there — everything **left** predicts `0`, everything **right** predicts `1`. That vertical dividing line is called the **decision boundary**.

```
y
1 |                 ●  ●  ●   ●
0.5·············●·····················  ← threshold 0.5
  |        ___/  ⋮
0 | ● ● ●_/      ⋮ ← decision boundary
  +--------------┼-----------------► tumor size (x)
     predict 0   │   predict 1
```

For *this* dataset, linear regression looks reasonable. The trouble shows up when the data changes slightly.

### Step 3 — Add One More Example (the Outlier Problem)

Add a single new malignant example **far to the right** (a very large tumor) and extend the axis. Intuitively this **shouldn't change** how we classify anything — large tumors are obviously malignant, and the sensible dividing line is unchanged.

But linear regression minimizes squared error, so that far-right point **drags the best-fit line over to the right**:

```
y
1 |        ●  ●  ●   ●                       ●  ← new far-right example
  |                       ________________/
0.5·····················●······················  ← threshold 0.5
  |       _______/      ⋮
0 | ● ●  ●              ⋮ ← boundary shifted RIGHT
  +--------------------┼-------------------------► tumor size (x)
        predict 0      │      predict 1
                       ✗ some malignant points now fall on the "predict 0" side
```

The decision boundary slides to the **right**, and points that were correctly called malignant are now misclassified as benign. **Adding one example that should change nothing has made the classifier worse.**

> The squared-error cost treats the distant point as a large residual it "wants" to reduce, so it tilts the whole line — even though for classification that point carries no new information. This is the core reason linear regression is unreliable for classification.

### Optional Lab (motivations)

An optional lab gives an **interactive plot** that tries to classify between two categories using linear regression. The point is to *see for yourself* that it often works poorly — which motivates needing a different model for classification.

### Takeaway — Why Linear Regression Fails

| | Behavior |
|---|---|
| Output range | Unbounded — produces values `< 0` and `> 1`, not just 0/1 |
| Sensitivity to far points | A single distant example **shifts the line and the decision boundary**, changing predictions that shouldn't change |
| Result | Classifications can flip for reasons that make no sense for the problem |

> "Sometimes you get lucky and it works, but often it won't." We want a model whose output is **always between 0 and 1** and that doesn't get dragged around by a single far-away point. That model is **logistic regression**.

## Logistic Regression

**Logistic regression** is probably the single most widely used classification algorithm in the world. Instead of a straight line, it fits a smooth **S-shaped curve** to the data.

> **Name warning:** despite the word "regression" in it, **logistic regression is used for classification**, not for predicting numbers. The name is historical.

### The S-Shaped Curve

![Two-panel slide. Left panel: tumor classification with an S-shaped (sigmoid) curve. Horizontal axis is tumor size x (diameter in cm); vertical axis takes only the values 0 ('no', benign) and 1 ('yes', malignant). Red X's for benign tumors cluster at y=0 on the left (small sizes); malignant X's sit at y=1 on the right (large sizes). A red S-curve rises smoothly from 0, through the middle, up toward 1. A horizontal purple line at 0.7 (labeled 'threshold — malignant?') meets the curve and a vertical line drops to a mid-range tumor size, illustrating a model output of 0.7. Right panel: the sigmoid / logistic function g(z) plotted against z (not x). Horizontal axis z ranges from −3 to +3 and takes both negative and positive values; vertical axis from 0 to 1 with a dashed asymptote at 1. Blue handwritten annotations: at z = −100, g(z) ≈ 1/big ≈ 0; at z = 0, g(z) = 1/(1+1) = 0.5 (purple dot at 0.5); at z = 100, g(z) ≈ 1. Formula g(z) = 1/(1+e^−z), with 0 < g(z) < 1. Labeled 'sigmoid function / logistic function — outputs between 0 and 1.'](images/sigmoid-function.jpg)

On the same tumor dataset, logistic regression fits a curve like the left panel above. If a patient comes in with a mid-sized tumor, the model might output **0.7**.

> The output `0.7` is **not** the label. The label `y` is only ever **0 or 1** — `0.7` is what the model produces, and we'll interpret it as a probability below.

### The Sigmoid (Logistic) Function

The building block is the **sigmoid function** (a.k.a. **logistic function**), which squashes any real number into the range `(0, 1)`:

```
g(z) = 1 / (1 + e^−z)            with   0 < g(z) < 1
```

where `e` is the mathematical constant ≈ 2.7, and `e^−z` is `e` raised to the power `−z`.

> **Watch the axes.** In the tumor plot the horizontal axis is the *feature* `x` (tumor size, always positive). In the sigmoid plot the horizontal axis is `z`, which ranges over **both negative and positive** values (the figure shows −3 to +3). They are *different axes* — `z` is computed from `x`, not equal to it.

**Why it has the S shape** — plug in extreme values:

| `z` | `e^−z` | `g(z) = 1/(1+e^−z)` | Result |
|---|---|---|---|
| Large positive (e.g. `100`) | `e^−100` ≈ tiny | `1 / (1 + tiny)` ≈ `1/1` | **≈ 1** |
| `0` | `e^0 = 1` | `1 / (1 + 1)` | **= 0.5** |
| Large negative (e.g. `−100`) | `e^100` ≈ huge | `1 / (1 + huge)` | **≈ 0** |

So the curve starts near `0` for very negative `z`, passes through exactly `0.5` at `z = 0`, and approaches `1` for very positive `z`.

### Building the Model — Two Steps

![Building the logistic regression model in two steps. Left: the sigmoid curve again — z-axis from −3 to 3, dashed asymptote at 1, value 0.5 marked at z=0 — with the formula g(z) = 1/(1+e^−z), 0 < g(z) < 1, labeled 'sigmoid / logistic function, outputs between 0 and 1.' Right: a box defining f_w,b(x) from two stacked steps — first a blue box z = w·x + b, an arrow down to z, then a red box g(z) = 1/(1+e^−z). Below, the two steps combine into f_w,b(x) = g(w·x + b) = 1/(1 + e^−(w·x + b)), with (w·x + b) under-braced as z. Captioned 'logistic regression.'](images/logistic-regression-model.jpg)

Logistic regression is just linear regression's output **fed through the sigmoid**:

**Step 1** — compute the familiar linear combination and call it `z`:

```
z = w · x + b
```

**Step 2** — pass `z` through the sigmoid:

```
g(z) = 1 / (1 + e^−z)
```

Put them together to get the **logistic regression model**:

```
f_w,b(x) = g(w · x + b) = 1 / (1 + e^−(w · x + b))
```

It takes a feature (or feature vector) `x` and outputs a number strictly between `0` and `1`.

> This is the same `z` from the sigmoid plot — the linear part `w·x + b` is what slides along the `z`-axis, and the sigmoid maps it into `(0, 1)`.

### Interpreting the Output as a Probability

![Slide titled 'Interpretation of logistic regression output.' Left: the model f_w,b(x) = 1/(1+e^−(w·x+b)), described as the 'probability' that the class is 1. Example: x is 'tumor size'; y is 0 (not malignant) or 1 (malignant); f_w,b(x) = 0.7 means a 70% chance that y is 1. Right: f_w,b(x) = P(y=1 | x; w, b) — the probability that y is 1 given input x and parameters w, b — and the identity P(y=0) + P(y=1) = 1.](images/logistic-output-interpretation.png)

Think of the output as the **probability that `y = 1`** for the given input `x`.

Example: a patient's tumor has size `x`, and the model outputs `f(x) = 0.7`. That means the model thinks there's a **70% chance** the tumor is malignant (`y = 1`).

Because `y` must be either `0` or `1`, the two probabilities sum to 1:

```
P(y = 0) + P(y = 1) = 1
```

So if `P(y = 1) = 0.7` (70%), then `P(y = 0) = 0.3` (30% chance benign).

#### Notation You Might See Elsewhere

Research papers and blog posts often write the output as:

```
f_w,b(x) = P(y = 1 | x ; w, b)
```

- The vertical bar `|` reads "given" — probability that `y = 1` **given** the input `x`.
- The **semicolon** `;` separates the inputs from the **parameters** `w, b`: it just signals that `w` and `b` are parameters that *affect* this probability, not random variables being conditioned on.

> You don't need to memorize this notation for the course — it's shown only so you recognize it elsewhere.

### Aside — This Algorithm Ran the Internet's Ads

For a long time, much of internet advertising was driven by a slight variation of logistic regression — it decided which ad to show to which user on some very large websites. Hugely lucrative, and a good example of how far this "simple" algorithm reaches.

### Optional Lab (sigmoid)

The optional lab after this lesson shows the **sigmoid function implemented in code** and plots a logistic-regression fit that does better on the classification task from the previous lab. The code is provided — just run it and read along to get familiar.

### Takeaway

| Idea | Detail |
|---|---|
| Logistic regression fits an **S-curve** | Smooth rise from 0 to 1, not a straight line |
| **Sigmoid** function | `g(z) = 1/(1+e^−z)`, always in `(0, 1)`; `g(0) = 0.5`, `g(+∞)→1`, `g(−∞)→0` |
| Two-step model | `z = w·x + b`, then `f(x) = g(z) = 1/(1+e^−(w·x+b))` |
| Output = **probability** | `f(x) = P(y=1 \| x; w, b)`; complement gives `P(y=0)` |
| Label vs output | `f(x)` can be `0.7`; the label `y` is still only `0` or `1` |
| "Regression" in the name | Misleading — it's a **classification** algorithm |

## Decision Boundary

The **decision boundary** separates the region where the model predicts `0` from the region where it predicts `1`. Looking at it gives a much better feel for *how logistic regression actually computes its predictions*.

### Recap — From Probability to a 0/1 Prediction

![Slide recapping how logistic regression turns its output into a hard prediction. Top-left: the sigmoid curve g(z) plotted against z (axis from roughly −3 to 3), with a dashed line at 1, the value 0.5 marked at z = 0, and handwritten notes showing g heads toward 0 as z→−∞ and toward 1 as z→+∞. Bottom-left: a black box stacking the two model steps — a blue box z = w·x + b, an arrow down labelled z, then a red box g(z) = 1/(1+e^−z). Right: f_w,b(x) = g(w·x + b) = 1/(1 + e^−(w·x+b)) = P(y=1 | x; w, b), with example outputs 0.7 and 0.3. Below, '0 or 1?' and 'threshold': Is f_w,b(x) ≥ 0.5? Yes → ŷ = 1, No → ŷ = 0. Then the chain 'When is f_w,b(x) ≥ 0.5? ⟺ g(z) ≥ 0.5 ⟺ z ≥ 0 ⟺ w·x + b ≥ 0 → ŷ = 1', alongside 'w·x + b < 0 → ŷ = 0'.](images/decision-boundary-threshold.jpg)

The model still computes its output in **two steps**:

```
z      = w · x + b
f(x)   = g(z) = 1 / (1 + e^−z) = 1 / (1 + e^−(w·x + b))
```

and we interpret `f(x)` as `P(y = 1 | x; w, b)` — a probability like `0.7` or `0.3`.

To force a concrete **0 or 1** answer, pick a **threshold** (commonly `0.5`):

```
if f(x) ≥ 0.5   →   ŷ = 1
if f(x) <  0.5   →   ŷ = 0
```

### When Does the Model Predict 1?

Trace the threshold back through both steps. Predicting `1` requires `f(x) ≥ 0.5`, and since `f(x) = g(z)`:

| Condition | ⟺ Equivalent to | Why |
|---|---|---|
| `f(x) ≥ 0.5` | `g(z) ≥ 0.5` | because `f(x) = g(z)` |
| `g(z) ≥ 0.5` | `z ≥ 0` | the sigmoid crosses `0.5` exactly at `z = 0` (right half of the curve) |
| `z ≥ 0` | `w · x + b ≥ 0` | because `z = w · x + b` |

So the whole chain collapses to a simple rule:

```
w · x + b ≥ 0   →   predict ŷ = 1
w · x + b <  0   →   predict ŷ = 0
```

> The messy sigmoid drops out entirely. Whether the model predicts 0 or 1 depends **only on the sign of `w · x + b`** — the same linear quantity from linear regression.

### The Decision Boundary

The interesting line to look at is where the model is **exactly neutral** — neither leaning toward `0` nor `1`:

```
w · x + b = 0      (equivalently z = 0)
```

This line is the **decision boundary**. On one side `w·x + b > 0` (predict 1); on the other `w·x + b < 0` (predict 0).

### Linear Decision Boundary — A Two-Feature Example

![Slide titled 'Decision boundary'. Model f_w,b(x) = g(z) = g(w1·x1 + w2·x2 + b) with parameters labelled w1 = 1, w2 = 1, b = −3. The decision boundary is derived: z = w·x + b = 0 → x1 + x2 − 3 = 0 → x1 + x2 = 3. Below, a 2-D plot with axes x1 (horizontal) and x2 (vertical), gridlines at 1, 2, 3. Blue circles (y = 0) cluster in the lower-left near the origin; red X's (y = 1) cluster in the upper-right. A straight purple line runs from (0,3) down to (3,0) — the boundary x1 + x2 = 3 — with the negative region (circled 'y = 0') to its lower-left and the positive region (circled 'y = 1') to its upper-right.](images/decision-boundary-linear.jpg)

Move from one feature to **two features** `x1, x2`. Red crosses are positive examples (`y = 1`), blue circles are negative (`y = 0`). The model is:

```
z = w1·x1 + w2·x2 + b
f(x) = g(z)
```

Take parameters `w1 = 1`, `w2 = 1`, `b = −3`. The decision boundary is where `z = 0`:

```
x1 + x2 − 3 = 0      →      x1 + x2 = 3
```

That's a straight line through `(3, 0)` and `(0, 3)`:

```
x2
3 ●\           ×  ×
  | \        ×  ×  ×
2 ○  \     ×   ×            right of line:  x1 + x2 ≥ 3
  | ○ ○\  ×                  →  w·x+b ≥ 0  →  predict 1
1 ○ ○ ○ \ ×
  | ○ ○ ○ \                 left of line:   x1 + x2 < 3
0 +--------\----------► x1    →  w·x+b < 0  →  predict 0
  0   1   2 3
```

- **Right of the line** → `w·x + b ≥ 0` → predict `ŷ = 1`.
- **Left of the line** → `w·x + b < 0` → predict `ŷ = 0`.

> A different choice of `w1, w2, b` just gives a **different straight line** — same idea, shifted/tilted.

### Non-Linear Decision Boundaries — Polynomial Features

![Slide titled 'Non-linear decision boundaries'. Model f_w,b(x) = g(z) = g(w1·x1² + w2·x2² + b) with w1 = 1, w2 = 1, b = −1. Decision boundary: z = x1² + x2² − 1 = 0 → x1² + x2² = 1. A 2-D plot with axes x1 and x2 and a purple circle of radius 1 centered at the origin. Blue circles (y = 0) sit inside the circle; red X's (y = 1) sit outside it. Annotations: 'x1² + x2² ≥ 1 → ŷ = 1' (outside the circle) and 'x1² + x2² < 1 → ŷ = 0' (inside the circle).](images/decision-boundary-nonlinear.jpg)

Just as with linear regression, you can feed **polynomial features** into logistic regression. Set:

```
z = w1·x1² + w2·x2² + b
```

With `w1 = 1`, `w2 = 1`, `b = −1`, the boundary `z = 0` becomes:

```
x1² + x2² − 1 = 0      →      x1² + x2² = 1
```

which is a **circle** of radius 1 centred at the origin:

```
        x2
         |    ×          outside the circle:
      ×  | ○ ○    ×        x1² + x2² ≥ 1
    ──○──○──○──○────► x1   →  predict ŷ = 1
      ×  | ○ ○    ×
         |    ×          inside the circle:
              ×           x1² + x2² < 1
                          →  predict ŷ = 0
```

- **Outside** the circle (`x1² + x2² ≥ 1`) → predict `1`.
- **Inside** the circle (`x1² + x2² < 1`) → predict `0`.

### Even More Complex Boundaries

Adding **higher-order polynomial terms** lets the boundary take far more elaborate shapes. For example:

```
z = w1·x1 + w2·x2 + w3·x1² + w4·x1·x2 + w5·x2²
```

can carve out **ellipses** or other curves, and still more terms can produce boundaries that look like arbitrary wiggly closed shapes. The model predicts `1` inside the shape and `0` outside (or vice-versa, depending on the parameters).

> **The key contrast:** if you use **only the raw features** `x1, x2, x3, …` (no higher-order terms), the decision boundary is **always linear** — a straight line (or flat plane). Curved boundaries come *entirely* from polynomial features. So logistic regression can fit anything from a simple line to very complex shapes, depending on the features you give it.

### Optional Lab (decision boundary)

The optional lab shows the **code implementation** of the decision boundary. The example uses **two features**, so the boundary appears as a **line** you can see plotted directly.

### Takeaway

| Idea | Detail |
|---|---|
| Prediction rule | Threshold the probability: `f(x) ≥ 0.5 → ŷ = 1`, else `ŷ = 0` |
| Simplifies to a sign test | `f(x) ≥ 0.5 ⟺ g(z) ≥ 0.5 ⟺ z ≥ 0 ⟺ w·x + b ≥ 0`; the sigmoid drops out |
| Decision boundary | The set where `w·x + b = 0` (z = 0) — the model is neutral there |
| Linear boundary | With raw features only, the boundary is a **straight line / plane** (e.g. `x1 + x2 = 3`) |
| Non-linear boundary | **Polynomial features** bend the boundary into circles, ellipses, and complex shapes (e.g. `x1² + x2² = 1`) |
| Flexibility | Logistic regression can fit anything from a simple line to very complex regions, set by the features you supply |

## Cost Function for Logistic Regression

The **cost function** measures how well a particular choice of parameters `w, b` fits the training data — and thereby gives us a way to **choose better parameters**. Here we see *why the squared-error cost is a poor fit for logistic regression*, and define a new cost function that works.

### The Training Set

Each row is a training example — e.g. a patient's visit ending in a diagnosis.

- `m` = number of training examples.
- Each example has `n` features `x1 … xn` (tumor size, age, …).
- Binary classification, so the target `y` is only ever **0 or 1**.
- The model is the logistic regression model `f_w,b(x) = 1 / (1 + e^−(w·x + b))`.

The question: given this training set, how do we choose `w` and `b`?

### Why Squared Error Fails — Non-Convexity

![Slide titled 'Squared error cost'. Top: the squared-error cost J(w,b) = (1/m) Σ_{i=1}^m [ ½ (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾)² ], with the ½ moved inside the summation (boxed). Bottom-left: 'linear regression', f_w,b(x) = w·x + b — a smooth bowl-shaped (convex) J(w,b) curve with blue gradient-descent arrows sliding down to the single global minimum. Bottom-right: 'logistic regression', f_w,b(x) = 1/(1+e^−(w·x+b)) — a wiggly 'non-convex' J(w,b) curve with many dips, and a gradient-descent step getting stuck in a shallow local minimum rather than reaching the global one.](images/squared-error-cost-nonconvex.jpg)

For **linear regression**, the squared-error cost is:

```
J(w,b) = (1/m) Σ_{i=1}^{m}  ½ (f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾)²
```

> Note the `½` is moved **inside** the summation here (vs Week 1's `1/2m` outside). It's purely cosmetic — it makes later math cleaner.

With `f(x) = w·x + b`, this `J` is **convex** — a single bowl — so gradient descent slides straight to the **global minimum**.

But plug in the logistic `f(x) = 1/(1+e^−(w·x+b))` and the same squared-error `J` becomes **non-convex** — a wiggly surface with **many local minima**. Gradient descent can get **stuck** in one of them and never reach the global minimum. So squared error is **not a good choice** for logistic regression.

### Loss vs Cost

We split the cost into a per-example **loss**. Inside the summation, call the term for a single example the **loss** `L`, a function of the prediction `f_w,b(x⁽ⁱ⁾)` and the true label `y⁽ⁱ⁾`:

```
J(w,b) = (1/m) Σ_{i=1}^{m}  L( f_w,b(x⁽ⁱ⁾), y⁽ⁱ⁾ )
```

- **Loss** `L` = how well you're doing on **one** example.
- **Cost** `J` = the average loss over the **whole** training set.

Choosing the right *form* for `L` is what restores convexity.

### The Logistic Loss Function

The loss used for logistic regression is defined piecewise on the label:

```
L( f_w,b(x⁽ⁱ⁾), y⁽ⁱ⁾ ) =  − log( f_w,b(x⁽ⁱ⁾) )        if y⁽ⁱ⁾ = 1
                          − log( 1 − f_w,b(x⁽ⁱ⁾) )    if y⁽ⁱ⁾ = 0
```

#### Case `y = 1` — loss is `−log(f)`

![Slide titled 'Logistic loss function', case y=1. Shows the piecewise loss definition. Left: a plot of L vs f_w,b(x) for y=1 over the valid range f ∈ (0,1) — the curve −log(f) starts very high (→∞) as f→0 and decreases to 0 at f=1, with marked points: f≈0.1 → high loss, f=0.5 → moderate loss, f→1 → loss≈0. Right: the full curves log(f) (purple, rising) and −log(f) (blue, falling) crossing the f-axis at f=1; a pink box highlights the relevant region 0<f<1. Annotations: 'As f→1 then loss→0' (good), 'As f→0 then loss→∞' (bad). Caption: 'Loss is lowest when f predicts close to the true label y.'](images/logistic-loss-y1.jpg)

`f` is always in `(0, 1)`, so only that part of `−log(f)` matters. When the true label is `1`:

| Prediction `f` | Loss `−log(f)` | Meaning |
|---|---|---|
| ≈ 1 | ≈ 0 | predicted ≈ 1, label is 1 → nearly right, tiny loss |
| 0.5 | moderate | hedging → moderate penalty |
| 0.1 | large | predicted 10% malignant but it *is* malignant → big penalty |

So when `y = 1`, the loss **pushes the model to predict close to 1** (loss is lowest there).

#### Case `y = 0` — loss is `−log(1 − f)`

![Slide titled 'Logistic loss function', case y=0. Shows the piecewise loss with the y=0 branch highlighted. Left: a plot of L vs f_w,b(x) for y=0 over f ∈ (0,1) — the curve −log(1−f) is ≈0 at f=0 and rises steeply toward ∞ as f→1, with f=0 marked 'not malignant' (loss≈0) and f near 1 (e.g. 99.9%) marked with loss→∞. Right: the curve −log(1−f) shown rising to infinity as f approaches 1, with a pink box on the relevant 0<f<1 region. Annotations: 'As f→0 then loss→0' (good), 'As f→1 then loss→∞' (bad). Caption: 'The further the prediction f is from target y, the higher the loss.'](images/logistic-loss-y0.jpg)

When the true label is `0`:

| Prediction `f` | Loss `−log(1 − f)` | Meaning |
|---|---|---|
| ≈ 0 | ≈ 0 | predicted ≈ 0, label is 0 → nearly right, tiny loss |
| larger | grows | further from 0 → bigger penalty |
| → 1 | → ∞ | "99.9% malignant" but it's benign → enormous penalty |

In both cases the rule is the same: **the further `f` is from the true `y`, the higher the loss.** This shape is what makes the overall cost convex.

### The Cost Function — Convex Again

![Slide titled 'Cost'. The cost J(w,b) = (1/m) Σ_{i=1}^m L(f_w,b(x⁽ⁱ⁾), y⁽ⁱ⁾), with the L(...) term under-braced as 'loss'. An arrow expands the loss into its piecewise definition: −log(f_w,b(x⁽ⁱ⁾)) if y=1, and −log(1−f_w,b(x⁽ⁱ⁾)) if y=0. Handwritten note: 'convex → can reach a global minimum'. Bottom: 'find w, b that minimize cost J'.](images/logistic-cost-function.jpg)

Putting it together, the **cost** is the average logistic loss over all examples:

```
J(w,b) = (1/m) Σ_{i=1}^{m}  L( f_w,b(x⁽ⁱ⁾), y⁽ⁱ⁾ )
```

With this choice of loss, `J(w,b)` is **convex** — so gradient descent can reliably reach the **global minimum**. (Proving convexity is beyond the course's scope.) Training then means: **find the `w, b` that minimize `J`.**

### Optional Lab (logistic loss)

The optional lab contrasts the two cost surfaces: the **squared-error** surface for classification is **wiggly with many local minima**, while the **logistic-loss** surface is **smooth and convex** with a single global minimum. Run the code and compare the plots.

### Takeaway

| Idea | Detail |
|---|---|
| Squared error on logistic `f` | **Non-convex** — many local minima, gradient descent gets stuck |
| Loss vs cost | **Loss** `L` = error on one example; **cost** `J` = average loss over all `m` examples |
| Logistic loss (`y=1`) | `−log(f)`: → 0 as `f`→1, → ∞ as `f`→0 |
| Logistic loss (`y=0`) | `−log(1−f)`: → 0 as `f`→0, → ∞ as `f`→1 |
| General principle | The further `f` is from the true `y`, the higher the loss |
| Result | The cost `J` is **convex** → gradient descent reaches the global minimum; train by minimizing `J` |

## Simplified Cost Function for Logistic Regression

Because this is **binary** classification, `y` is *always* `0` or `1` — never anything else. That single fact lets us collapse the two-case loss into **one line**, which makes the gradient-descent implementation simpler.

### The Simplified Loss — One Equation

Recall the piecewise loss from the previous section:

```
L( f_w,b(x⁽ⁱ⁾), y⁽ⁱ⁾ ) =  − log( f_w,b(x⁽ⁱ⁾) )        if y⁽ⁱ⁾ = 1
                          − log( 1 − f_w,b(x⁽ⁱ⁾) )    if y⁽ⁱ⁾ = 0
```

Since `y` can only be `0` or `1`, the two cases can be written as a **single expression** (drop the `(i)` superscripts for readability):

```
L( f_w,b(x), y ) =  − y · log( f_w,b(x) )  −  (1 − y) · log( 1 − f_w,b(x) )
```

This one line is **completely equivalent** to the two-case version above.

### Why It's Equivalent — Plug In Each Label

The trick: one of the two terms always **switches off**, because `y` multiplies the first term and `(1 − y)` multiplies the second.

| If `y =` | First term `−y·log(f)` | Second term `−(1−y)·log(1−f)` | Loss reduces to |
|---|---|---|---|
| **1** | `−1·log(f)` = `−log(f)` | `−(1−1)·log(1−f)` = `−0·… = 0` | **`−log(f)`** ✓ |
| **0** | `−0·log(f)` = `0` | `−(1−0)·log(1−f)` = `−log(1−f)` | **`−log(1−f)`** ✓ |

> When `y = 1`, the `(1 − y) = 0` factor kills the second term. When `y = 0`, the `y = 0` factor kills the first term. Either way, you get back exactly the matching case from the original piecewise definition — so the single line loses nothing.

### The Simplified Cost Function

The cost `J` is still just the **average loss** over all `m` examples:

```
J(w,b) = (1/m) Σ_{i=1}^{m}  L( f_w,b(x⁽ⁱ⁾), y⁽ⁱ⁾ )
```

Substitute the one-line loss and pull the two negative signs outside the sum:

```
J(w,b) = − (1/m) Σ_{i=1}^{m} [ y⁽ⁱ⁾ · log( f_w,b(x⁽ⁱ⁾) )  +  (1 − y⁽ⁱ⁾) · log( 1 − f_w,b(x⁽ⁱ⁾) ) ]
```

This is the cost function that **essentially everyone** uses to train logistic regression.

### Why This Particular Cost?

Of all the cost functions we could have invented, why this one?

- It's derived from statistics using a principle called **maximum likelihood estimation (MLE)** — a standard way to efficiently find parameters for many models. (You don't need the details for this course.)
- It has the nice property of being **convex**, so gradient descent can reliably reach the **global minimum**.

### Optional Lab (logistic cost)

An optional lab shows the logistic cost function **implemented in code**, and lets you see how **two different choices of parameters** produce different cost values — the better-fitting (blue) decision boundary has a **lower cost** than the worse (magenta) one. Worth a look, since you implement this in the practice lab at the end of the week.

### Takeaway

| Idea | Detail |
|---|---|
| Simplified loss | `L = − y·log(f) − (1−y)·log(1−f)` — one line, equivalent to the two-case form |
| Why it works | `y` and `(1−y)` switch each term on/off, since `y` is only ever `0` or `1` |
| Simplified cost | `J(w,b) = −(1/m) Σ [ y⁽ⁱ⁾ log(f⁽ⁱ⁾) + (1−y⁽ⁱ⁾) log(1−f⁽ⁱ⁾) ]` |
| Where it comes from | **Maximum likelihood estimation**; the result is **convex** |

## Gradient Descent for Logistic Regression

To fit a logistic regression model we **find the `w, b` that minimize the cost `J(w,b)`** — and the tool, as before, is **gradient descent**. Once trained, feeding the model a new input `x` (e.g. a new patient's tumor size and age) yields `P(y = 1)`, an estimate of the probability the label is 1.

### The Gradient Descent Algorithm

Same template as linear regression — repeatedly nudge each parameter downhill by the learning rate `α` times the partial derivative:

```
repeat until convergence {
    w_j = w_j − α · ∂J(w,b)/∂w_j        (for j = 1 … n)
    b   = b   − α · ∂J(w,b)/∂b
}
```

### The Derivatives

Applying calculus to the logistic cost (derivation beyond scope) gives, for feature `j`:

```
∂J(w,b)/∂w_j = (1/m) Σ_{i=1}^{m} ( f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x_j⁽ⁱ⁾
∂J(w,b)/∂b   = (1/m) Σ_{i=1}^{m} ( f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

where `x_j⁽ⁱ⁾` is the **`j`-th feature of training example `i`**. The two are identical except the `b` derivative drops the trailing `· x_j⁽ⁱ⁾`. The term `( f − y )` is the **error** — prediction minus true label.

Plugging these in gives the full update rules (the `j` loop runs over all `n` features):

```
w_j = w_j − α · (1/m) Σ_{i=1}^{m} ( f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾ ) · x_j⁽ⁱ⁾
b   = b   − α · (1/m) Σ_{i=1}^{m} ( f_w,b(x⁽ⁱ⁾) − y⁽ⁱ⁾ )
```

### Simultaneous Update

As with linear regression, **update all parameters simultaneously**: compute every right-hand side first (using the *old* values), then overwrite all the `w_j` and `b` at once.

### "Wait — Isn't This Identical to Linear Regression?"

The two algorithms look **letter-for-letter the same**. They are *not* the same algorithm — the difference is hidden inside the definition of `f`:

| | `f_w,b(x)` is… |
|---|---|
| **Linear regression** | `f(x) = w · x + b` (a straight line) |
| **Logistic regression** | `f(x) = 1 / (1 + e^−(w · x + b))` (the sigmoid) |

> Same update *equation*, completely different *model*. Because `f` is the sigmoid here, this is a genuinely different algorithm despite the identical-looking math.

### Same Practical Techniques Carry Over

Everything you learned for linear-regression gradient descent applies here too:

| Technique | How it applies to logistic regression |
|---|---|
| **Monitoring convergence** | Plot the learning curve and check `J` decreases each iteration — same method as before |
| **Vectorization** | Vectorized implementation makes gradient descent run much faster (details in the optional labs) |
| **Feature scaling** | Scaling features to similar ranges (e.g. `−1` to `+1`) speeds up convergence here just as it did for linear regression |

### Optional Labs (gradient descent + scikit-learn)

- **Gradient descent lab** — shows the logistic gradient computed in code, with **animated plots** as gradient descent runs: the sigmoid, the **contour plot** of the cost, the **3-D surface** of the cost, and the **learning curve** all evolving together. Useful prep for the end-of-week practice lab.
- **Scikit-learn lab** — short but valuable: how to train a logistic regression model with the popular **scikit-learn** library (`LogisticRegression`). Many practitioners use scikit-learn daily, so it's worth seeing how it's called.

### Takeaway

| Idea | Detail |
|---|---|
| Goal | Find `w, b` minimizing the (convex) logistic cost `J(w,b)` via gradient descent |
| `w_j` update | `w_j = w_j − α·(1/m) Σ ( f⁽ⁱ⁾ − y⁽ⁱ⁾ ) x_j⁽ⁱ⁾` |
| `b` update | `b = b − α·(1/m) Σ ( f⁽ⁱ⁾ − y⁽ⁱ⁾ )` |
| Looks like linear regression | …but it isn't — `f` is the **sigmoid**, not `w·x + b` |
| Update style | **Simultaneous** update of all `w_j` and `b` |
| Carries over | Convergence monitoring, **vectorization**, and **feature scaling** all work the same way |

## The Problem of Overfitting

Linear and logistic regression work well for many tasks, but they can run into **overfitting**, which makes them perform poorly. There's also an almost-opposite problem, **underfitting**. The goal of this section is to *recognize* both; the next sections cover how to fix overfitting (especially **regularization**).

### Three Fits on the Housing Example (Regression)

Predict house **price** `y` from **size** `x`. With the same 5 training points, three models behave very differently:

```
  UNDERFIT (high bias)        JUST RIGHT                 OVERFIT (high variance)
  f = w·x + b                 f = w1·x + w2·x² + b        f = w1·x + … + w4·x⁴ + b
  (straight line)             (quadratic)                 (4th-order polynomial)

price                        price                       price
 |        ___---             |        ___----            |     /\    /‾
 |   __---     ●             |     _--‾  ● ●             |   ●/  \  /  ●
 | ●/  ● ●                   |   ●/ ●                    | ●/    \●/
 |/●                         |●/                         |/  ●
 +----------- size           +----------- size           +----------- size
 fits poorly                 fits well, generalizes      passes through ALL points,
                                                          wiggly, generalizes badly
```

| Model | Fit | Diagnosis | Other name |
|---|---|---|---|
| **Straight line** | Misses the obvious curve (prices flatten as size grows) | **Underfitting** | **High bias** |
| **Quadratic** (`x`, `x²`) | Fits training set well *and* would predict a new house well | **Just right** (good **generalization**) | — |
| **4th-order polynomial** | Passes through all 5 points exactly (cost can be **0**), but wildly wiggly | **Overfitting** | **High variance** |

### Key Vocabulary

| Term | Meaning |
|---|---|
| **Underfitting / high bias** | Model is too simple to capture the clear pattern — it can't even fit the *training* set well. Like a strong **preconception** (e.g. "price *must* be linear in size") that overrides the data. |
| **Generalization** | Making good predictions on **brand-new examples never seen in training** — the actual goal. |
| **Overfitting / high variance** | Model fits the training data *too* well (even noise), so it fails to generalize. |
| **Just right** | Neither underfit nor overfit — neither high bias nor high variance. |

> **"Bias" has two meanings.** Checking algorithms for **bias against genders/ethnicities** is critical and real — but that's *not* the meaning here. In this context, **bias = underfitting**: the model has a strong preconception and can't fit the data.

### Why "High Variance"?

An overfit model is trying *so* hard to pass through every training point that the fit is **fragile**: change the training set even slightly (one house priced a bit differently) and the curve could come out **completely different**. Two engineers fitting the same high-order model to slightly different data would get highly **variable** predictions — hence *high variance*.

> **Goldilocks intuition:** too cold (underfit), too hot (overfit), and one that's *just right* in the middle. The aim of machine learning is the "just right" model — **neither high bias nor high variance**.

### Overfitting in Classification

![Slide titled 'Classification'. Three panels of the same 2-D data — x1 (e.g. tumor size) horizontal, x2 (e.g. patient age) vertical — with red X's (malignant, y=1) and blue circles (benign, y=0), each panel showing a logistic regression decision boundary (where z = 0, g = sigmoid). Left: z = w1·x1 + w2·x2 + b gives a straight line that splits the classes only roughly — labelled 'underfit / high bias'. Middle: adding quadratic terms z = w1·x1 + w2·x2 + w3·x1² + w4·x2² + w5·x1·x2 + b gives a smooth ellipse-like boundary that fits well without classifying every point perfectly — labelled 'just right'. Right: a very high-order polynomial (many higher-order terms, +…+ b) produces a contorted, twisted boundary that wraps around individual points to classify the training set perfectly — labelled 'overfit'.](images/overfitting-classification.jpg)

The same spectrum applies to logistic regression (here `x1` = tumor size, `x2` = patient age; `g` is the sigmoid, with the boundary at `z = 0`):

| Features in `z` | Decision boundary | Diagnosis |
|---|---|---|
| `z = w1·x1 + w2·x2 + b` | **Straight line** — roughly separates, but a poor fit | **Underfit / high bias** |
| add `x1²`, `x2²`, `x1·x2` | **Ellipse-like curve** — fits well, doesn't classify every point perfectly, generalizes | **Just right** |
| many high-order polynomial terms | **Contorted, twisted boundary** wrapping individual points — perfect on training set | **Overfit / high variance** |

### Takeaway

| Idea | Detail |
|---|---|
| **Underfit = high bias** | Too few features / too simple → can't fit even the training data; strong preconception |
| **Overfit = high variance** | Too many features / too complex → fits training data (even noise) almost perfectly, fails on new data, fragile to small data changes |
| **Generalization** | The real goal: do well on **unseen** examples, not just the training set |
| **Just right** | The middle ground — neither high bias nor high variance |
| Applies to both | Same spectrum holds for **regression** (curve fit) and **classification** (decision boundary) |
| What's next | Techniques to **address overfitting** — notably **regularization** |

---

*Prior weeks: [Week 1 — Introduction](../../introduction-to-machinelearning/week1/week1.md) · [Week 2 — Regression with Multiple Input Variables](../../regression-with-multiple-input-variables/week2/week2.md)*
