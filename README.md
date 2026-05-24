# Machine Learning Specialization

A learning-tracking repository for my study of the [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) by DeepLearning.AI (Andrew Ng) on Coursera.

This repo holds my course notes, week-by-week summaries, and from-scratch Python implementations of the algorithms taught in the lectures. The goal is to reinforce the material by re-deriving the math in writing and re-implementing each algorithm with NumPy instead of only running the provided lab notebooks.

## Repository Structure

Code is organized by course → week, mirroring the structure of the specialization.

```
supervised-machine-learning/
  introduction-to-machinelearning/
    week1/
      week1.md                    # notes (cost function, gradient descent, etc.)
      univariate-lr-implementation/
      gradient-descent-from-scratch/
      images/                     # diagrams referenced from notes
  regression-with-multiple-input-variables/
    week2/
      week2.md                    # notes (vectorization, feature scaling, etc.)
      multiple-variable-lr-implementation/
      images/
    feature-scaling-and-learning-rate.py
```

Each `weekN.md` is a self-contained writeup of that week's lectures. Sibling folders contain runnable Python implementations of the algorithms covered.

## Courses

- [x] Course 1 — Supervised Machine Learning: Regression and Classification *(in progress)*
- [ ] Course 2 — Advanced Learning Algorithms
- [ ] Course 3 — Unsupervised Learning, Recommenders, Reinforcement Learning

## Tech

- Python 3.13
- NumPy
- Matplotlib
- Managed with [uv](https://github.com/astral-sh/uv) (see `pyproject.toml` / `uv.lock`)

## Running the Code

```powershell
uv sync
uv run python supervised-machine-learning/regression-with-multiple-input-variables/feature-scaling-and-learning-rate.py
```

## Disclaimer

This is a personal study repo. Notes are my own paraphrasing of the lectures and may contain mistakes. Nothing here is an official course resource — please refer to the [original course](https://www.deeplearning.ai/courses/machine-learning-specialization/) for authoritative material.
