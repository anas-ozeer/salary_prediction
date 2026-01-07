# Salary Prediction

## 📊 Overview

An interactive **Streamlit web application** that analyzes Data Science job postings to understand salary drivers and predict salaries using machine learning. This app explores what factors most influence a data scientist's salary — including company size, revenue, location, skills (like Python), and ownership type.

### ✨ Features

- **📈 Data Visualization**: Interactive dashboard with comprehensive job market insights
- **🤖 ML Modeling**: XGBoost regression model for salary prediction with 85% test-train split
- **🔍 AI Explainability**: SHAP (SHapley Additive exPlanations) visualizations to interpret model predictions
- **⚙️ Hyperparameter Tuning**: Automated model optimization using PyCaret
- **📊 Interactive Filters**: Filter job postings by industry, location, and more
- **🎯 Salary Predictor**: Real-time salary predictions based on job features

### 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML/AI**: scikit-learn, XGBoost, PyCaret, SHAP
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, Looker Studio integration
- **MLOps**: MLflow, DagsHub

## 🚀 How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

## 📂 Dataset

The app uses `salary_data_cleaned.csv` containing Data Science job posting data with features including:
- Company information (size, revenue, ownership type)
- Job location (US states mapped to proximity tiers)
- Required skills (Python, etc.)
- Average salary (target variable)
