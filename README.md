# symbolic-law-discovery
Automated discovery of interpretable mathematical laws from data using Symbolic Regression. Instead of fitting black-box models with millions of hidden parameters, this project leverages PySR’s evolutionary search engine to identify the simplest equations that best explain observed data, enabling transparent and scientifically meaningful modeling.

## 🌟 Core Concepts

### 1️⃣ Parsimony-Driven Search (Occam’s Razor)

The algorithm minimizes prediction error while penalizing equation complexity.

This encourages:

- Simpler models  
- Reduced overfitting  
- Higher interpretability  

Formally, we optimize a multi-objective trade-off between:

- **Loss (prediction error)**
- **Complexity (expression size)**

---

### 2️⃣ Pareto-Optimal Model Selection

PySR generates a **Hall of Fame** — a set of candidate equations forming a Pareto frontier.

Each equation represents an optimal trade-off between:

- Predictive accuracy  
- Structural simplicity  

The selected equation lies on this frontier, balancing performance and interpretability.

---

## 📊 Example: Iris Dataset

In this demonstration, symbolic regression is applied to the Iris dataset.

### 🔎 Discovered Equation

y = (x₂ × x₃) / (1.948 × x₁)

Where:

- **x₁** = Sepal Width  
- **x₂** = Petal Length  
- **x₃** = Petal Width  

This suggests species separation is strongly influenced by petal dimensions normalized by sepal width.

> ⚠ Note: This experiment applies regression to categorical labels for demonstration purposes. Future extensions may include proper classification thresholds or symbolic logistic models.

---

## 📈 Performance

| Metric | Value |
|--------|--------|
| Mean Absolute Error (Test Set) | 0.1846 |
| Mean Absolute Error (Subset Validation) | 0.0443 |
| Interpretability | High (Explicit Closed-Form Equation) |

---

## 🛠️ Tech Stack

- **Python** – Data preprocessing and evaluation  
- **PySR** – Symbolic regression engine  
- **Julia** – High-performance evolutionary backend  
- **SymPy** – Symbolic equation formatting  
- **scikit-learn** – Dataset handling and metrics  
## ⚠️ Prototype & Interpretation Notes

This repository is a **proof-of-concept prototype** demonstrating symbolic regression for interpretable modeling.

The discovered equation should **not** be interpreted as a definitive biological law. In this example, regression is applied to categorical labels for demonstration purposes, and the resulting formula is an approximation of decision boundaries rather than a mechanistic model of species formation.

### Why This Still Matters

Even when the formula is not physically "correct," symbolic regression provides valuable insight into:

- Which variables dominate predictions  
- How features interact  
- The structural form of the learned relationship  

In this sense, the equation acts as a **diagnostic window into model behavior**.

---

## 🧠 Symbolic Regression as a Diagnostic Tool

Interpretable equations can be used to evaluate model generalization:

- If we *know* the true governing law of a system and the AI produces a structurally different equation, this may indicate:
  - Overfitting  
  - Poor generalization  
  - Missing variables  
  - Biased training data  

- If the discovered structure closely resembles known theory, this increases confidence in:
  - Model validity  
  - Stability  
  - Scientific consistency  

Thus, symbolic regression can function as a **model auditing mechanism**, not just a predictive tool.

---
## 🔎 Interpretability & Diagnostic Value

One of the key motivations behind symbolic regression is not merely prediction accuracy, but **structural insight**.

Unlike high-parameter black-box models, symbolic regression exposes the mathematical form of the learned relationship. This makes it useful as a diagnostic tool for evaluating model behavior and generalization.

---

### 1️⃣ Detecting Spurious Correlations ("Clever Hans" Effects)

Machine learning models can achieve high accuracy by exploiting unintended signals in the data (data leakage or spurious correlations).

Symbolic regression can help reveal this.

For example, if a discovered equation includes irrelevant variables such as:
y = 0.0001 * timestamp + target

this immediately signals that the model is using dataset artifacts rather than meaningful features.

Such structural transparency is difficult to obtain from large neural networks with millions of parameters.

---

### 2️⃣ Structural Consistency vs Numerical Accuracy

In scientific modeling, structural plausibility often matters more than marginal numerical improvements.

For example:

- A discovered law resembling
- F = G * m1 * m2 / r^2
- is structurally meaningful.

- A lower-error but structurally arbitrary expression such as
- F = sin(m1) + exp(m2)
- may indicate overfitting or poor generalization.

Symbolic regression allows us to evaluate whether the model’s structure aligns with domain knowledge.

---

### 3️⃣ Transparent Feature Interactions

Symbolic models explicitly reveal interactions between variables.

In this project, the discovered relationship:
y = (x₂ × x₃) / (1.948 × x₁)

makes it clear that feature interactions (multiplication and normalization) play a role in prediction.

Such structural relationships are often hidden in black-box models and cannot be fully inferred from feature importance scores alone.

---

## 🎯 Key Takeaway

The goal of this project is not merely predictive performance.

It is to demonstrate how symbolic regression can provide:

- Interpretability  
- Structural transparency  
- Model auditing capability  
- Insight into feature interactions  

In scientific and engineering contexts, this structural insight can be as valuable as raw accuracy.

This project demonstrates that even approximate symbolic models:

- Provide interpretability  
- Reveal feature interactions  
- Help assess generalization behavior  

The goal is not merely prediction, but **structural insight**.
