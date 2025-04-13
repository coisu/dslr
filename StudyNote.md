# 🔍 Logistic Regression vs Linear Regression & One-vs-All Strategy

## 1. Logistic Regression vs Linear Regression

| Feature              | Linear Regression                      | Logistic Regression                                |
|----------------------|----------------------------------------|----------------------------------------------------|
| **Purpose**          | Predicts numeric values (e.g., price)  | Predicts probability for classification (e.g., yes/no) |
| **Output Range**     | Continuous (e.g., -∞ to +∞)            | Bounded between 0 and 1 (probability)              |
| **Equation**         | `y = θ₀ + θ₁x₁ + θ₂x₂ + ...`           | `h(x) = 1 / (1 + e^(-θᵀx))` (sigmoid function)     |
| **Use Case Example** | Predicting exam score                 | Predicting likelihood of passing an exam           |

> ✅ Linear Regression is used for regression tasks.  
> ✅ Logistic Regression is used for classification tasks using probabilities.

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

## ✅ TL;DR

> Linear Regression is for **predicting numbers**.  
> Logistic Regression is for **predicting probabilities** for binary classification.  
> To classify multiple classes, we use **One-vs-All**, training one logistic model per class.

