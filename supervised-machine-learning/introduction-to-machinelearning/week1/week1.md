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