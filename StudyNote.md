
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

Gradient Descent is a fundamental algorithm for optimizing model parameters in machine learning and deep learning. Below is a summary of the main types and the most commonly used methods in practice.

> We are currently using **Batch Gradient Descent** to **learn the weights** (`θ`) in logistic regression.


### Currently Used: **Batch Gradient Descent**
```python
for _ in range(num_iters):
    gradient = (1/m) * (X.T @ (sigmoid(X @ theta) - y))
    theta -= alpha * gradient
```

- Computes gradient using **all samples** in every iteration.
- Very stable, but slow for large datasets.

---

### Basic Types of Gradient Descent

| Type                        | Description |
|-----------------------------|-------------|
| **Batch Gradient Descent**  | Uses the entire dataset to compute the gradient at each step. Most stable and simple but slow. |
| **Stochastic Gradient Descent (SGD)** | Uses **one sample at a time** to compute the gradient. Fast but very noisy. |
| **Mini-batch Gradient Descent** | Splits the data into **small batches** to compute the gradient. Balances speed and stability. |

#### 🔍 Detailed Comparison

| Feature           | Batch GD               | SGD                     | Mini-batch GD            |
|-------------------|------------------------|--------------------------|---------------------------|
| Data Used         | All data               | One sample               | Small batch (e.g., 32, 128) |
| Speed             | Slow                   | Fast                     | Moderate                  |
| Memory Efficiency | Low                    | Very high                | High                      |
| Stability         | Very stable            | Very unstable (high variance) | Stable                |
| Real-world Usage  | Less efficient        |  Used in special cases  |  **Most widely used**    |


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
