# 💰 Employee Salary Prediction using Lasso Regression

## 📌 Project Overview

This project focuses on predicting **employee monthly salary** using Machine Learning techniques.
The goal is to build a robust regression model while addressing common challenges such as:

* Overfitting
* Multicollinearity
* Model complexity

To achieve this, we implemented and compared:

✅ Linear Regression (Baseline)
✅ Ridge Regression (L2 Regularization)
✅ Lasso Regression (L1 Regularization)

The final model was deployed as an **interactive Streamlit web application**.

---

## 🎯 Problem Statement

Accurate salary prediction is valuable for:

* Compensation planning
* Budget forecasting
* HR analytics
* Fair pay analysis

This project uses employee attributes like experience, education, and job role to predict salary.

---

## 📊 Dataset Description

The dataset represents structured HR analytics data containing employee-related features:

| Feature           | Description           |
| ----------------- | --------------------- |
| Age               | Employee age          |
| Gender            | Male / Female         |
| Department        | Functional department |
| JobRole           | Employee designation  |
| EducationLevel    | Qualification         |
| YearsExperience   | Total work experience |
| PerformanceRating | Performance score     |
| WorkHoursPerWeek  | Weekly work hours     |
| MonthlySalary     | **Target Variable**   |

---

## 🧠 Machine Learning Approach

### **1️⃣ Linear Regression**

Linear Regression serves as the **baseline model** for comparison.

The objective is to minimize:

[
Loss = RSS = \sum (y_i - \hat{y}_i)^2
]

Where:

* ( y_i ) → Actual salary
* ( \hat{y}_i ) → Predicted salary
* **RSS** → Residual Sum of Squares

---

### **2️⃣ Ridge Regression (L2 Regularization)**

Ridge Regression adds an **L2 penalty** to reduce model complexity and handle multicollinearity.

[
Loss = RSS + λ \sum w_j^2
]

Where:

* **RSS** → Residual Sum of Squares
* **λ (lambda)** → Regularization parameter
* ( w_j^2 ) → Squared model coefficients

**Effect of Ridge Regression:**

✔ Shrinks coefficients toward zero
✔ Reduces variance
✔ Improves stability
✔ Retains all features

---

### **3️⃣ Lasso Regression (L1 Regularization)**

Lasso Regression applies an **L1 penalty**, enabling coefficient shrinkage and feature selection.

[
Loss = RSS + λ \sum |w_j|
]

Where:

* **RSS** → Residual Sum of Squares
* **λ (lambda)** → Regularization parameter
* ( |w_j| ) → Absolute coefficient values

**Effect of Lasso Regression:**

✔ Shrinks coefficients
✔ Forces some coefficients = 0
✔ Performs automatic feature selection
✔ Improves interpretability

---

## 🎯 Role of Regularization Parameter (λ)

The parameter **λ (lambda)** controls the strength of regularization:

* **λ = 0** → Equivalent to Linear Regression
* **Small λ** → Mild shrinkage
* **Large λ** → Strong shrinkage

Trade-off:

✔ Higher λ → Less overfitting
❌ Too large λ → Underfitting

---

✅ Proper symbols
✅ Academic formatting
✅ Portfolio-grade

---

If you want next, I can add:

📊 Coefficient shrinkage intuition
📉 Ridge vs Lasso geometry explanation
✨ Math + visual combo section

Just tell me 😄🔥

---

## ⚖️ Ridge vs Lasso

| Aspect                | Ridge    | Lasso   |
| --------------------- | -------- | ------- |
| Regularization        | L2       | L1      |
| Feature Selection     | ❌ No     | ✅ Yes   |
| Coefficient Shrinkage | ✅ Yes    | ✅ Yes   |
| Model Complexity      | Moderate | Simpler |

---

## 🛠️ Project Workflow

1️⃣ Data Preprocessing
2️⃣ Exploratory Data Analysis (EDA)
3️⃣ Feature Encoding
4️⃣ Feature Scaling (StandardScaler)
5️⃣ Model Training
6️⃣ Hyperparameter Tuning (GridSearchCV)
7️⃣ Model Evaluation
8️⃣ Final Model Selection
9️⃣ Deployment (Streamlit)

---

## 📉 Model Evaluation Metrics

Models were evaluated using:

* **RMSE (Root Mean Squared Error)**
* **R² Score**

---

## 📊 Results Summary

| Model                | RMSE        | R² Score     |
| -------------------- | ----------- | ------------ |
| Linear Regression    | 5199.74     | 0.9555       |
| Ridge Regression     | 5193.94     | 0.9556       |
| **Lasso Regression** | **5058.03** | **0.9579** ✅ |

---

## 🏆 Final Model Selection

**Lasso Regression** was selected because:

✔ Lowest RMSE (better accuracy)
✔ Highest R² Score
✔ Automatic Feature Selection
✔ Improved Interpretability

---

## 🔍 Key Insights

✅ Years of experience strongly influences salary
✅ JobRole significantly impacts compensation
✅ Performance rating positively affects salary
✅ Lasso eliminated weak predictors

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Streamlit
* Joblib

---

## 🎓 Learning Outcomes

Through this project, I learned:

✅ Data preprocessing techniques
✅ Exploratory Data Analysis (EDA)
✅ Regression modeling
✅ Ridge vs Lasso regularization
✅ Hyperparameter tuning
✅ Model evaluation metrics
✅ ML model deployment

---

## ⭐ Key Takeaway

> Regularization techniques like Ridge and Lasso improve model generalization, stability, and interpretability by controlling coefficient magnitudes and complexity.

Just tell me 😄🔥
