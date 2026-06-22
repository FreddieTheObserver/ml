# Week 3: Classification

> Week 2 predicted a **number** (linear regression). Week 3 predicts a **category** — the output `y` can only be one of a small, finite set of values. This first lesson shows *why linear regression is a poor fit for classification*, which motivates the algorithm we'll use instead: **logistic regression**.

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

### Summary of the Failure

| | Behavior |
|---|---|
| Output range | Unbounded — produces values `< 0` and `> 1`, not just 0/1 |
| Sensitivity to far points | A single distant example **shifts the line and the decision boundary**, changing predictions that shouldn't change |
| Result | Classifications can flip for reasons that make no sense for the problem |

> "Sometimes you get lucky and it works, but often it won't" — which is why we don't use linear regression for classification.

## Decision Boundary

The **decision boundary** is the dividing line (here, where the model output crosses the threshold) that separates the region predicting `0` from the region predicting `1`. In the example above it was the vertical line where the fitted line hit `0.5`. We'll define and explore it more carefully in the next lesson.

## What We Want Instead → Logistic Regression

The next algorithm, **logistic regression**, fixes these problems:

- Its output is **always between 0 and 1** — never below 0, never above 1.
- It is far more robust: a single far-away example doesn't drag the boundary around the way linear regression's straight line does.

> **Name warning:** despite the word "regression" in it, **logistic regression is used for classification**, not for predicting numbers. The name is historical. It solves **binary classification** problems where `y` is `0` or `1`.

## Optional Lab Note

The upcoming optional lab gives an **interactive plot** that tries to classify between two categories using linear regression. The point is to *see for yourself* that it often works poorly — which motivates needing a different model (logistic regression) for classification.

## Takeaway

| Idea | Detail |
|---|---|
| Classification predicts a **category** | Output `y` is one of a small finite set |
| **Binary** classification | Exactly two classes: `0`/negative/no vs `1`/positive/yes |
| Negative/positive = **absence/presence** | Not bad/good; which class is `0` vs `1` is arbitrary but fixed |
| Linear regression **misfits** classification | Output isn't bounded to 0/1; threshold gives a decision boundary |
| The fatal flaw | One distant example **shifts the line and boundary**, flipping predictions that shouldn't change |
| **Logistic regression** is the fix | Output always in `[0, 1]`, robust — used for classification despite its name |

> **Next:** logistic regression and the **sigmoid function** — how to produce an output between 0 and 1, and how the decision boundary really works.

---

*Prior weeks: [Week 1 — Introduction](../../introduction-to-machinelearning/week1/week1.md) · [Week 2 — Regression with Multiple Input Variables](../../regression-with-multiple-input-variables/week2/week2.md)*
