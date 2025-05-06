import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk_metrics

# PyCaret for regression modeling
from pycaret.regression import setup, compare_models

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# DAGsHub integration for MLflow logging
import dagshub

from xgboost import XGBRegressor
import shap
from streamlit_shap import st_shap

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

    st.title("🔍 Explainable AI")
    st.markdown("Understand model behavior using SHAP (SHapley Additive Explanations).")

    # Make sure model and dfnew are already defined earlier in your app
    # Load the dataset
    df = load_data()
    dfnew = df.select_dtypes(include=[np.number])
    # Define features (X) and target (y)
    X = dfnew.drop("avg_salary", axis=1)
    y = dfnew["avg_salary"]

    # If train/test split hasn't been done yet, do it here
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

    # Retrain the model just to ensure it exists (or reuse a previously trained model)
    from xgboost import XGBRegressor
    xgb_model = XGBRegressor(n_estimators=19, learning_rate=0.81, max_depth=6)
    xgb_model.fit(X_train, y_train)

    # Create SHAP explainer and compute values
    explainer = shap.Explainer(xgb_model, X)
    shap_values = explainer(X)

    # Feature selection dropdown for summary plot
    st.subheader("Select features for SHAP")
    selected_features = st.multiselect("Choose features to display in SHAP summary plot", X.columns.tolist(), default=X.columns.tolist())

    # Prepare filtered SHAP values
    filtered_X = X[selected_features]
    filtered_shap_values = shap.Explanation(
        values=shap_values.values[:, [X.columns.get_loc(col) for col in selected_features]],
        base_values=shap_values.base_values,
        data=filtered_X.values,
        feature_names=selected_features
    )

    # Display SHAP Summary Plot
    st.subheader("SHAP Summary Plot")
    st_shap(shap.plots.beeswarm(filtered_shap_values), height=500)

    # SHAP Dependence Plot
    st.markdown("---")
    st.subheader("SHAP Dependence Plot")

    dependence_feature = st.selectbox("Select a feature for dependence plot", selected_features)
    st.markdown(f"*SHAP Dependence Plot for ⁠ {dependence_feature} ⁠*")
    st_shap(shap.plots.scatter(filtered_shap_values[:, dependence_feature], color=filtered_shap_values), height=500)



# Hyperparameter Tuning Page        
elif page == "Hyperparameter Tuning":
    st.title("🤖 Hyperparameter Tuning with PyCaret + MLflow via DAGsHub")

    st.markdown("""
    In this section, we will:
    - Run **PyCaret's model comparison** to select the best 3 models
    - **Log each model** to **DAGsHub MLflow** with hyperparameters and performance metrics  
    - Tune and evaluate models only when the button is clicked
    """)

    # Load and clean the dataset
    df = load_data()
    df_hp = df.dropna(subset=["avg_salary"])
    df_hp = df_hp.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Train-test split
    salary_train, salary_test = train_test_split(df_hp, test_size=0.2, random_state=42)

    # DAGsHub MLflow Integration
    dagshub.init(repo_owner='anas-ozeer', repo_name='salary_prediction', mlflow=True)

    # Button to trigger tuning
    if st.button("🚀 Run Hyperparameter Tuning & Log to MLflow"):
        with st.spinner("Training and logging top models..."):
            # PyCaret setup
            reg1 = setup(data=salary_train, target='avg_salary', session_id=42, verbose=False)

            # Select top 3 models
            top3 = compare_models(n_select=3)

            # Evaluate and log each model
            for i, model in enumerate(top3, 1):
                with mlflow.start_run(run_name=f"Regressor {i}: {model.__class__.__name__}"):
                    model_name = f"regressor_model_{i}"

                    # Log model
                    mlflow.sklearn.log_model(model, model_name)

                    # Log parameters
                    params = model.get_params()
                    for key, value in params.items():
                        mlflow.log_param(key, value)

                    # Predict and evaluate
                    y_test = salary_test["avg_salary"]
                    X_test = salary_test.drop("avg_salary", axis=1)
                    y_pred = model.predict(X_test)

                    # Calculate regression metrics
                    rmse = sk_metrics.root_mean_squared_error(y_test, y_pred, squared=False)
                    mae = sk_metrics.mean_absolute_error(y_test, y_pred)
                    r2 = sk_metrics.r2_score(y_test, y_pred)

                    # Log metrics
                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("R2", r2)

                    st.success(f"✅ Logged Regressor {i}: {model.__class__.__name__}")
                    st.write(f"**RMSE:** {rmse:.2f} | **MAE:** {mae:.2f} | **R2:** {r2:.2f}")

                mlflow.end_run()