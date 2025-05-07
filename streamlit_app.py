import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk_metrics
from pycaret.regression import setup, tune_model, compare_models
import mlflow
import mlflow.sklearn
import dagshub
from xgboost import XGBRegressor
import shap
from streamlit_shap import st_shap
import os

os.environ["DAGSHUB_TOKEN"] = "a2dd5cc1b8858cf2430c40a71d57f1814389d5fa"

@st.cache_data
def load_data():
    return pd.read_csv('salary_data_cleaned.csv')

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select", ["Introduction", "Data Visualization", "Modeling", "AI Explainability", "Hyperparameter Tuning"])

df = load_data()
dfnew = df[['python_yn','Size','Revenue','job_state','Type of ownership','avg_salary']]

size_map = {"1 to 50 employees": 25, "51 to 200 employees": 125, "201 to 500 employees": 350, "501 to 1000 employees": 750,
            "1001 to 5000 employees": 3000, "5001 to 10000 employees": 7500, "10000+ employees": 15000}
rev_map = {"Less than $1 million (USD)": 5e5, "$1 to $5 million (USD)": 2.5e6, "$5 to $10 million (USD)": 7.5e6,
            "$10 to $25 million (USD)": 1.75e7, "$25 to $50 million (USD)": 3.75e7, "$50 to $100 million (USD)": 7.5e7,
            "$100 to $500 million (USD)": 2.5e8, "$500 million to $1 billion (USD)": 7.5e8, "$1 to $2 billion (USD)": 1.5e9,
            "$2 to $5 billion (USD)": 3.5e9, "$5 to $10 billion (USD)": 7.5e9, "$10+ billion (USD)": 1.5e10}
state_map = {' KS': 1, ' NE': 1, ' OK': 1, ' MO': 1, ' IA': 1, ' AR': 1, ' IL': 2, ' IN': 2, ' KY': 2, ' CO': 2, 
                ' SD': 2, ' MN': 2, ' TX': 2, ' TN': 2, ' WI': 3, ' MI': 3, ' OH': 3, ' MS': 3, ' ND': 3, ' NM': 3, 
                ' WY': 3, ' GA': 4, ' AL': 4, ' PA': 4, ' NC': 4, ' SC': 4, ' LA': 4, ' MT': 4, ' AZ': 4, ' WV': 4,
                ' NY': 5, ' NJ': 5, ' CA': 5, ' FL': 5, ' WA': 5, ' OR': 5, ' MA': 5, ' CT': 5, ' RI': 5, ' NH': 5,
                ' ME': 5, ' DE': 5, ' MD': 5, ' VT': 5, ' NV': 5, ' UT': 5, ' ID': 5, ' AK': 5}
owner_map = {"Company - Private": 2, "Company - Public": 1, "Nonprofit Organization": 0,
                "Subsidiary or Business Segment": 2, "Government": 1, "Hospital": 2,
                "College / University": 1, "Other Organization": 1, "School / School District": 1}


dfnew['Size'] = dfnew['Size'].map(size_map)
dfnew['Revenue'] = dfnew['Revenue'].map(rev_map).fillna(0)
dfnew['job_state'] = dfnew['job_state'].map(state_map).fillna(0)
dfnew['Type of ownership'] = dfnew['Type of ownership'].map(owner_map)
dfnew.dropna(inplace=True)

# INTRODUCTION
if page == "Introduction":
    st.title("📊 Job Postings Data Analysis App")

    st.markdown("""
    👋 **Welcome to the Job Postings Data Analysis App!**

    This tool explores a dataset of Data Science job postings to understand **what drives salaries**.

    ### 💡 *What factors most influence a data scientist’s salary — and can we predict it accurately?*

    ---

    ### 🔍 We explore:
    - Which **company sizes**, **industries**, **skills**, and **locations** affect salary
    - Building an ML model to **predict salary** from job features

    --- 
    """)

    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Summary Statistics (Numerical Columns)")
    st.write(df.describe())

    st.subheader("Unique Values in Categorical Columns")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
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

    # Filters
    with st.expander("🔍 Filter Options"):
        col1, col2 = st.columns(2)
        selected_industries = col1.multiselect("Select Industry", df['Industry'].dropna().unique())
        selected_locations = col2.multiselect("Select Location", df['Location'].dropna().unique())

        filtered_df = df.copy()
        if selected_industries:
            filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
        if selected_locations:
            filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations)]

        st.subheader("Filtered Job Postings")
        st.write(f"Total jobs found: {filtered_df.shape[0]}")
        st.dataframe(filtered_df.head(20))

# DATA VISUALIZATION
elif page == "Data Visualization":
    st.title("Data Visualization")
    import streamlit.components.v1 as components
    components.iframe(src="https://lookerstudio.google.com/embed/reporting/4836b76b-fc1b-4563-a492-3ba2a915ecd8/page/bpRIF", height=600, width=1000)

# MODELING
elif page == "Modeling":
    st.title("Modeling")

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    # Define features and target
    X = dfnew.drop('avg_salary', axis=1)
    y = dfnew['avg_salary']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and train model
    linregmodel = LinearRegression()
    linregmodel.fit(X_train, y_train)

    # Define features and target
    X = dfnew.drop('avg_salary', axis=1)
    y = dfnew['avg_salary']

    
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=8)
    
      # Initialize and train XGBoost model
    xgb_model = XGBRegressor(n_estimators=19, learning_rate=0.81, max_depth=6,)
    xgb_model.fit(X_train, y_train)
    
    size = st.slider("Company Size", min_value=1, max_value=15000, value=7500)
    revenue = st.slider("Revenue", min_value=500000, max_value=10000000000, value=1000000)

    python_yn = st.selectbox("Uses Python?", options=["No", "Yes"])
    python_yn = 1 if python_yn == "Yes" else 0  # Convert to numeric if needed
    
    # 2. US State selection
    us_states = [
        " AL", " AK", " AZ", " AR", " CA", " CO", " CT", " DE", " FL", " GA",
        " HI", " ID", " IL", " IN", " IA", " KS", " KY", " LA", " ME", " MD",
        " MA", " MI", " MN", " MS", " MO", " MT", " NE", " NV", " NH", " NJ",
        " NM", " NY", " NC", " ND", " OH", " OK", " OR", " PA", " RI", " SC",
        " SD", " TN", " TX", " UT", " VT", " VA", " WA", " WV", " WI", " WY"
    ]
    job_state = st.selectbox("Job State", options=us_states)
    
    # 3. Type of ownership
    ownership_types = ["Company - Private", "Company - Public", "Government", "Nonprofit Organization"]
    ownership = st.selectbox("Company Ownership Type", options=ownership_types)
    
    # Display the inputs
    st.write("Selected values:")
    st.write("Python:", "Yes" if python_yn else "No")
    st.write("Job State:", state_map[job_state])
    st.write("Ownership:", owner_map[ownership])
    
    inputdatapoint = pd.DataFrame([{
        'python_yn': python_yn,
        'Size': size,
        'Revenue': revenue,
        'job_state': state_map[job_state],
        'Type of ownership': owner_map[ownership]
    }])

    st.write("Linear Regression Prediction: ") 
    st.write(linregmodel.predict(inputdatapoint)*1000)
    
    st.write("XGBoost Prediction: ")
    st.write(xgb_model.predict(inputdatapoint)*1000)


# AI EXPLAINABILITY
elif page == "AI Explainability":
    st.title("🔍 AI Explainability")
    st.write("Explaining model predictions using SHAP values.")


    X = dfnew.drop("avg_salary", axis=1)
    y = dfnew["avg_salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.3)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    selected_features = st.multiselect("Choose SHAP features", X.columns.tolist(), default=X.columns.tolist())
    if selected_features:
        idxs = [X.columns.get_loc(f) for f in selected_features]
        filtered_shap = shap.Explanation(
            values=shap_values.values[:, idxs],
            base_values=shap_values.base_values,
            data=X[selected_features].values,
            feature_names=selected_features
        )

        st.subheader("SHAP Summary Plot")
        st_shap(shap.plots.beeswarm(filtered_shap), height=500)

        feature = st.selectbox("Dependence plot feature", selected_features)
        st.subheader(f"SHAP Dependence Plot: {feature}")
        st_shap(shap.plots.scatter(filtered_shap[:, feature], color=filtered_shap), height=500)

# HYPERPARAMETER TUNING
elif page == "Hyperparameter Tuning":
    st.title("🤖 Hyperparameter Tuning with PyCaret + MLflow via DAGsHub")
    
    # Train-test split
    salary_train, salary_test = train_test_split(dfnew, test_size=0.2, random_state=42)

    # DAGsHub MLflow Integration
    dagshub.init(repo_owner='anas-ozeer', repo_name='salary_prediction', mlflow=True)

    if st.button("🚀 Run Hyperparameter Tuning & Log to MLflow"):
        with st.spinner("Training and logging top models..."):
            from pycaret.regression import setup, compare_models, tune_model

            # PyCaret setup
            reg1 = setup(data=salary_train, target='avg_salary', session_id=42, verbose=False)

            # Select top 3 models
            top3_models = compare_models(n_select=3)

            st.subheader("📊 Top 3 Models (Before Tuning):")
            for i, model in enumerate(top3_models, 1):
                with mlflow.start_run(run_name=f"Top Model {i}: {model.__class__.__name__}"):
                    model_name = f"top_model_{i}"
                    
                    y_test = salary_test["avg_salary"]
                    X_test = salary_test.drop("avg_salary", axis=1)
                    y_pred = model.predict(X_test)

                    rmse = sk_metrics.mean_squared_error(y_test, y_pred, squared=False)
                    mae = sk_metrics.mean_absolute_error(y_test, y_pred)
                    r2 = sk_metrics.r2_score(y_test, y_pred)

                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("R2", r2)

                    st.write(f"**Model {i}: {model.__class__.__name__}**")
                    st.write(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.2f}")

                mlflow.end_run()

            # Pick the best of the top 3
            # Find model with highest R2 on test set
            best_model = max(top3_models, key=lambda model: sk_metrics.r2_score(
                salary_test["avg_salary"], model.predict(salary_test.drop("avg_salary", axis=1))
            ))

            st.subheader("🎯 Hyperparameter Tuning on Best Model")


            # Perform hyperparameter tuning
            tuned_model = tune_model(
                best_model,
                n_iter=10,  # Limit number of iterations
                early_stopping=True,  # Enable early stopping
                early_stopping_max_iters=5,  # Set a limit on the number of iterations
                search_library='scikit-learn',  # Use the default library
                search_algorithm='random',  # Use random search (faster)
            )

            with mlflow.start_run(run_name=f"Tuned Model: {tuned_model.__class__.__name__}"):
                model_name = "tuned_regressor_model"

                mlflow.sklearn.log_model(tuned_model, model_name)
                params = tuned_model.get_params()
                for key, value in params.items():
                    mlflow.log_param(key, value)

                y_pred = tuned_model.predict(X_test)

                rmse = sk_metrics.mean_squared_error(y_test, y_pred, squared=False)
                mae = sk_metrics.mean_absolute_error(y_test, y_pred)
                r2 = sk_metrics.r2_score(y_test, y_pred)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)

                st.success(f"✅ Logged Tuned Model: {tuned_model.__class__.__name__}")
                st.write(f"**Tuned Model - RMSE:** {rmse:.2f} | **MAE:** {mae:.2f} | **R2:** {r2:.2f}")

            mlflow.end_run()
