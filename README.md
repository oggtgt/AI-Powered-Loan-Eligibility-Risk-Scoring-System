![Ashprogrammer29 Profile Banner](https://s3.ap-south-1.amazonaws.com/d2c-cdn-mumbai/uploads/user-project-files/6884f818c661f_aswin_other.png)

# 🚀 AI-Powered Loan Eligibility & Risk Scoring System

**Empowering smarter, faster, and fairer loan decisions with Machine Learning & FastAPI!** 💸🤖

---

## Table of Contents

- [🎯 Project Objective](#project-objective)
- [🛤️ Step-by-Step Guide](#step-by-step-guide)
- [🗂️ Repository Structure](#repository-structure)
- [⚡ Installation](#installation)
- [🔌 API Usage](#api-usage)
- [📊 Model Details](#model-details)
- [📝 Results Interpretation](#results-interpretation)
- [🤝 Contributing](#contributing)
- [📄 License](#license)

---

## 🎯 Project Objective

Build a robust Loan Risk Model using multiple ML algorithms. The project focuses on handling imbalanced defaulter datasets, extensive feature engineering, and delivering instant risk predictions via a FastAPI backend.

---

## 🛤️ Step-by-Step Guide

1. **Data Preprocessing**
   - 📥 Load dataset (CSV or other source).
   - 🔍 Check for null values (none found, so skip imputation).
   - 🎯 Define target variable (e.g., `isDefault`).
   - 🧬 Identify variable types (`object`, `float`, `int`).
   - 📈 Analyze feature relationships via correlation matrix.

2. **Feature Engineering**
   - 🛠️ Create new extract features (~15 engineered features for deeper insights).
   - 🔢 Encode categorical features numerically.
   - 📏 Scale features using `StandardScaler`.

3. **Data Preparation for Modeling**
   - ✂️ Split data into train/test sets.
   - ⚖️ Handle class imbalance (7.61:1 ratio) with class weights.

4. **Feature Selection**
   - ⭐ Map and select important features.

5. **Model Training**
   - 🤖 Train: LightGBM, XGBoost, Random Forest, Logistic Regression.
   - 🏆 Evaluate with AUC and Precision-Recall (Logistic Regression performed best).

6. **Model Ensemble**
   - 🧩 Stacking Ensemble: LightGBM + XGBoost + Random Forest (Meta-learner: Logistic Regression).

7. **Model Evaluation**
   - 📋 Generate classification report.
   - 🟦 Visualize results with confusion matrix.

8. **FastAPI Backend**
   - 🦸 Deploy best model via FastAPI app.
   - 🛡️ Use Pydantic for type checking and input validation.
   - 🔮 Serve predictions at `/predict` endpoint.

9. **Prediction**
   - 🚦 Predict loan risk for new applicants in real time.

---

## 🗂️ Repository Structure

```
.
├── data/                 # 📊 Dataset files
├── feature_engineering/  # 🛠️ Feature engineering scripts/notebooks
├── modeling/             # 🤖 ML model training scripts
├── api/                  # 🔌 FastAPI app & Pydantic models
├── requirements.txt      # 📦 Python dependencies
├── README.md             # 📄 Project documentation
└── LICENSE               # 📜 License info
```

---

## ⚡ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Ashprogrammer29/AI-Powered-Loan-Eligibility-Risk-Scoring-System.git
    cd AI-Powered-Loan-Eligibility-Risk-Scoring-System
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔌 API Usage

1. **Start the FastAPI server:**
    ```bash
    uvicorn api.main:app --reload
    ```
2. **Send a POST request to `/predict` with applicant data:**
    ```json
    {
      "feature_1": value,
      "feature_2": value,
      ...
    }
    ```
3. **Get real-time loan risk prediction in the response!**

---

## 📊 Model Details

- **Features:** Engineered features + original variables.
- **Models:** Logistic Regression (best performer), LightGBM, XGBoost, Random Forest (in ensemble).
- **Metrics:** Area Under Curve (AUC), Precision-Recall, Classification Report, Confusion Matrix.

---

## 📝 Results Interpretation

- **Classification Report:** Precision, recall, f1-score for each class.
- **Confusion Matrix:** Visualizes correct vs. incorrect predictions.
- **API Response:** Instant loan risk prediction for each applicant.

---

## 🤝 Contributing

Contributions are welcome!  
Open an issue or submit a pull request to make this project even better. 🌟

---

## 📄 License

This project is licensed under the **Mozilla Public License Version 2.0**.  
See the [`LICENSE`](./LICENSE) file for details.

---

> _Made with ❤️ by [Ashprogrammer29](https://github.com/Ashprogrammer29) — Revolutionizing lending with AI & open source!_
