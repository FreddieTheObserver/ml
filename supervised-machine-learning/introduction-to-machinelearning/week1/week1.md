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
