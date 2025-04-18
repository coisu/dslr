
# 🔍 Logistic Regression vs Linear Regression & One-vs-All Strategy

## 1. Logistic Regression vs Linear Regression

| Feature              | Linear Regression                      | Logistic Regression                                |
|----------------------|----------------------------------------|----------------------------------------------------|
| **Purpose**          | Predicts numeric values (e.g., price)  | Predicts probability for classification (e.g., yes/no) |
| **Output Range**     | Continuous (e.g., -∞ to +∞)            | Bounded between 0 and 1 (probability)              |
| **Equation**         | `y = θ₀ + θ₁x₁ + θ₂x₂ + ...`           | `h(x) = 1 / (1 + e^(-θᵀx))` (sigmoid function)     |
| **Use Case Example** | Predicting exam score                 | Predicting likelihood of passing an exam           |

> Linear Regression is used for regression tasks.  
> Logistic Regression is used for classification tasks using probabilities.

---

## 2. What is One-vs-All (OvA)?

When you have more than 2 classes (e.g., Hogwarts Houses), Logistic Regression alone isn't enough.  
**One-vs-All (or One-vs-Rest)** allows us to handle multi-class classification using multiple binary classifiers.

For example, with 4 classes (Gryffindor, Slytherin, Ravenclaw, Hufflepuff):

- Train 4 separate logistic regression models:
  - Model 1: Gryffindor vs All
  - Model 2: Slytherin vs All
  - Model 3: Ravenclaw vs All
  - Model 4: Hufflepuff vs All

Each model outputs the **probability** that the input belongs to its specific class.

---

## 3. How do they work together?

1. Train one logistic regression model for **each class** (OvA)  
2. For prediction:
   - Feed the input into all models  
   - Each model gives a probability  
   - Choose the class with the **highest probability**

---

## 4. Implicit Bias vs Explicit Bias

In logistic regression, the **bias term** (also called **intercept**) allows the model to shift the decision boundary.

| Type              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Implicit Bias** | Add a column of 1s to the input features (e.g., `X_with_bias = [1, x₁, x₂, ...]`) and learn `θ₀` as part of `θ`. |
| **Explicit Bias** | Keep bias as a separate variable `b` and compute prediction as `z = Xwᵀ + b`. |

### Differences:
- **Implicit**: Bias is handled as part of the weight vector and requires special care to **exclude it from regularization** (`reg_term[0] = 0`).
- **Explicit**: Bias is handled separately and is **not included in L2 regularization** by default, making the code more readable and less error-prone.

---

## 5. Gradient Descent Variants

Gradient Descent is the optimization algorithm we use to **learn the weights** (`θ`) in logistic regression.

### Currently Used: **Batch Gradient Descent**
```python
for _ in range(num_iters):
    gradient = (1/m) * (X.T @ (sigmoid(X @ theta) - y))
    theta -= alpha * gradient
```

- Computes gradient using **all samples** in every iteration.
- Very stable, but slow for large datasets.

### Other Common Variants

| Type                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Stochastic Gradient Descent (SGD)** | Updates weights using **1 sample at a time** → faster, but noisier updates. |
| **Mini-batch Gradient Descent**      | Compromise between batch and SGD: update weights using a **small subset (batch)** of samples. |
| **Gradient Descent with Momentum**   | Adds a velocity term to smooth updates, helps escape local minima. |
| **Adam Optimizer**                   | Adaptive learning rate optimizer, combining momentum and RMSProp techniques (very popular in deep learning). |

> ✅ We are currently using **Batch Gradient Descent** with optional **L2 regularization** (`lambda_`), which is a solid, interpretable baseline for logistic regression.

---

## 6. What is `pickle` used for in this project?

- In this project, we use `pickle` to save and load the `StandardScaler` object from `scikit-learn`.
- pickle allows us to save and restore Python objects exactly as they were.

### Why do we use `pickle`?
- The model must apply the **same standardization rule** during both training and prediction.
- `StandardScaler` learns the **mean and standard deviation** of each feature during training.
- We save that trained scaler using `pickle`, so we can apply the same transformation to the test data during prediction.

### What's inside `scaler.pkl`?

| Attribute              | Meaning                                         |
|------------------------|--------------------------------------------------|
| `scaler.mean_`         | Mean of each feature from training data          |
| `scaler.scale_`        | Standard deviation of each feature               |
| `scaler.var_`          | Variance of each feature                         |
| `scaler.n_features_in_`| Number of features seen during fitting           |

### Load and use `.pkl` files:

```python
with open("trained/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)
```

---
