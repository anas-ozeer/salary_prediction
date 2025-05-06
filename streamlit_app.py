import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
from pycaret.regression import *
import streamlit as st
import dagshub
import mlflow
from sklearn import metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('salary_data_cleaned.csv')
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select", ["Introduction", "Data Visualization", "Modeling", "AI Explainability", "Hyperparameter Tuning"])

# Data Exploration Page
if page == "Introduction":
    st.title("📊 Job Postings Data Analysis App")

    st.markdown("""
    👋 **Welcome to the Job Postings Data Analysis App!**

    This interactive tool is built to explore a rich dataset of Data Science job postings — and more importantly, to understand **what drives salaries** in the field.  
    By analyzing patterns across industries, company types, job requirements, and locations, we aim to answer a central question:

    ### 💡 *What factors most influence a data scientist’s salary — and can we predict it accurately based on those features?*

    ---

    ### 🔍 Through this app, we explore:
    - **Which company sizes** (startups, mid-size, large enterprises) are hiring data scientists most actively  
    - **Which sectors and industries** are investing heavily in data science talent  
    - **How skill requirements differ** between startups and established companies  
    - **Which U.S. locations** tend to offer **the highest salaries** for data scientists  

    As you navigate through filters and visualizations, you’ll see how these variables relate to salary.

    ---

    ### 🤖 And at the core of it all:
    We’re building a **machine learning model** that uses these variables — such as average salary, location, company size, company type and more — to **predict the expected salary** for a given job posting.

    In short, this project is about turning job market data into actionable salary insights using data science itself.
    """)
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Summary Statistics (Numerical Columns)")
    st.write(df.describe())

    # st.subheader("Missing Values Per Column")
    # missing = df.isnull().sum()
    # st.write(missing[missing > 0].sort_values(ascending=False))

    st.subheader("Unique Values in Categorical Columns")
    cols = df.select_dtypes(include='object').columns
    for col in cols:
        st.write(f"**{col}**: {df[col].nunique()} unique values")

    st.subheader("Top 10 Job Titles")
    st.write(df['Job Title'].value_counts().head(10))

    st.subheader("Top 10 Hiring Companies")
    st.write(df['Company Name'].value_counts().head(10))

    st.subheader("Top 10 Locations")
    st.write(df['Location'].value_counts().head(10))

    st.subheader("Top 10 Industries")
    st.write(df['Industry'].value_counts().head(10))

    if 'salary' in df.columns:
        st.subheader("Salary Statistics")
        st.write(df['salary'].describe())

    # Optional filtering
    # Filters inside the page
    with st.expander("🔍 Filter Options"):
        col1, col2 = st.columns(2)

        with col1:
            selected_industries = st.multiselect(
                "Select Industry:",
                options=df['Industry'].dropna().unique(),
                key="industry_filter"
            )

        with col2:
            selected_locations = st.multiselect(
                "Select Location:",
                options=df['Location'].dropna().unique(),
                key="location_filter"
            )


    filtered_df = df.copy()

    if selected_industries:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]

    if selected_locations:
        filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations)]

    # Show filtered results
    st.subheader("Filtered Job Postings")
    st.write(f"Total jobs found: {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head(20))



# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    import streamlit.components.v1 as components

    # Your Looker Studio dashboard URL
    looker_dashboard_url = "https://lookerstudio.google.com/embed/reporting/4836b76b-fc1b-4563-a492-3ba2a915ecd8/page/bpRIF"

    # Embed Looker dashboard
    components.iframe(src=looker_dashboard_url, height=600, width=1000)



# Modeling Page
elif page == "Modeling":
    st.title("Modeling")
    st.write("This page is under construction. 🤖")

# AI Explainability Page
elif page == "AI Explainability":
    st.title("AI Explainability")
    st.write("This page is under construction. 🤖")

# Hyperparameter Tuning Page        
elif page == "Hyperparameter Tuning":
    st.title("🤖 Hyperparameter Tuning with PyCaret + MLflow via DAGsHub")

    st.markdown("""
    In this section, we will perform **hyperparameter tuning** using **PyCaret** for regression models. 
    We'll fine-tune the selected models' hyperparameters and log the results into **MLflow** for tracking and reproducibility.
    """)

    # Load the dataset
    df = load_data()  # Use the existing function to load the dataset

    # Preprocess dataset (drop missing values or any preprocessing as needed)
    df_hp = df.dropna(subset=["avg_salary"])  # Drop rows with missing target variable

    # Split dataset into features (X) and target variable (y)
    X = df_hp.drop("avg_salary", axis=1)
    y = df_hp["avg_salary"]

    # Split into train/test sets
    salary_train, salary_test = train_test_split(df_hp, test_size=0.2, random_state=42)

    # Initialize PyCaret setup
    reg1 = setup(data=salary_train, target='avg_salary', session_id=42, verbose=False)

    # Dropdown for model selection
    model_choice = st.selectbox(
        "Select Model for Tuning",
        options=["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "Ridge", "Lasso"]
    )

    # Model tuning based on selected choice
    if model_choice:
        if model_choice == "Linear Regression":
            model = create_model('lr')
        elif model_choice == "Random Forest":
            model = create_model('rf')
        elif model_choice == "XGBoost":
            model = create_model('xgboost')
        elif model_choice == "LightGBM":
            model = create_model('lightgbm')
        elif model_choice == "Ridge":
            model = create_model('ridge')
        elif model_choice == "Lasso":
            model = create_model('lasso')

        # Tuning the model
        tuned_model = tune_model(model)

        # Display tuned model summary
        st.subheader(f"Tuned {model_choice} Model Summary")
        st.write(tuned_model)

        # Log the tuned model to MLflow
        with mlflow.start_run(run_name=f"Tuned {model_choice}"):
            # Log model
            mlflow.sklearn.log_model(tuned_model, f"{model_choice}_tuned_model")

            # Log parameters
            params = tuned_model.get_params()
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Predict using tuned model
            X_test = salary_test.drop("avg_salary", axis=1)
            y_test = salary_test["avg_salary"]
            y_pred = tuned_model.predict(X_test)

            # Calculate regression metrics
            rmse = sk_metrics.mean_squared_error(y_test, y_pred, squared=False)
            mae = sk_metrics.mean_absolute_error(y_test, y_pred)
            r2 = sk_metrics.r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # End the MLflow run
            mlflow.end_run()

        # Display the results of hyperparameter tuning
        st.subheader("Hyperparameter Tuning Results")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAE: {mae}")
        st.write(f"R2: {r2}")

        st.write("Tuned Model Parameters:")
        st.write(params)

    st.write("🔧 Adjust the model's hyperparameters and tune the model by selecting different options.")
