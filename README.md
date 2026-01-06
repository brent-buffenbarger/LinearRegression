# 📈 Linear Regression From Scratch (Gradient Descent)

This repository contains a Jupyter notebook that implements **linear regression from first principles** using Python, NumPy, and Plotly. Instead of relying on high-level machine learning APIs, the notebook manually builds and visualizes every core component of linear regression to develop strong intuition.

The focus is on **understanding how gradient descent works**, how loss behaves, and how model parameters converge.

---

## 🧠 What This Notebook Covers

The notebook walks step-by-step through the complete lifecycle of a simple linear regression model:

1. Loading and exploring a real-world dataset
2. Visualizing feature–target relationships
3. Defining Mean Squared Error (MSE)
4. Deriving and implementing analytical gradients
5. Running gradient descent until convergence
6. Monitoring loss over time
7. Visualizing the loss surface in parameter space
8. Evaluating the final model with multiple metrics

Each step is implemented explicitly to emphasize *how* learning happens.

---

## 📊 Dataset

The notebook uses the **Diabetes dataset** from `scikit-learn`:

- Loaded via `load_diabetes(as_frame=True).frame`
- Features are **standardized** (mean-centered and scaled)
- The target is a quantitative measure of diabetes disease progression
- For clarity and visualization, the model uses a **single feature**:
  - `bmi` (Body Mass Index)

Using one feature makes it possible to visualize the loss surface directly in 3D.

---

## 📐 Model Definition

The regression model is defined as:

\[
\hat{y} = mx + b
\]

Where:
- `m` is the slope
- `b` is the intercept
- `x` is the BMI feature
- `\hat{y}` is the predicted target value

---

## 📉 Loss Function

The model minimizes **Mean Squared Error (MSE)**:

\[
\text{MSE} = \frac{1}{N} \sum (y - \hat{y})^2
\]

MSE is:
- Differentiable
- Convex for linear regression
- Ideal for gradient-based optimization

This is the exact objective function used during training.

---

## 🔁 Gradient Descent Optimization

Gradient descent is implemented manually using analytical gradients.

### Gradients

\[
\frac{\partial \text{MSE}}{\partial m} = \frac{2}{N} \sum (mx + b - y)x
\]

\[
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{N} \sum (mx + b - y)
\]

### Update rules

\[
m \leftarrow m - \alpha \cdot \frac{\partial \text{MSE}}{\partial m}
\]

\[
b \leftarrow b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
\]

Where:
- `α` is the learning rate
- `epsilon` is the convergence tolerance
- `max_iterations` is a safety cap

Training stops when:
- the change in loss falls below `epsilon`, or
- `max_iterations` is reached

A divergence check is included to guard against unstable learning rates:

- If loss becomes `NaN` or grows extremely large, training stops and suggests reducing `alpha`.

---

## 📈 Evaluation Metrics

After training, the model is evaluated using three complementary metrics:

### Mean Squared Error (MSE)
- The loss function optimized during training
- Penalizes large errors more heavily (because errors are squared)

### Root Mean Squared Error (RMSE)
- Square root of MSE
- Interpretable in the same units as the target
- Represents a typical prediction error magnitude

### R² (Coefficient of Determination)
- Measures how much variance in the target is explained by the model
- Compared against a baseline that predicts the mean
- Scale-free and useful for comparing models

---

## 📊 Visualizations

The notebook includes interactive Plotly visualizations designed to build geometric intuition.

### 1. Data + Regression Line
- Scatter plot of BMI vs target
- Overlaid best-fit regression line learned via gradient descent

### 2. Loss Curve
- MSE plotted over training iterations
- Shows convergence behavior and stability of training

### 3. Loss Surface (3D)
- MSE computed over a grid of `(m, b)` values
- Plotted as a 3D surface to show the convex “bowl” shape
- Helps explain why linear regression has a single global minimum

---

## 🧠 Key Takeaways

- Linear regression has a **single global minimum** (no local minima)
- Feature scaling dramatically affects gradient descent behavior and stability
- Learning rate controls speed vs. overshooting/divergence
- The loss curve is a 1D view of movement across a higher-dimensional surface
- Gradient descent follows the steepest downhill direction on the loss surface

---