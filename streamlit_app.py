import streamlit as st
import pandas as pd
import os
import mlflow
from pycaret.regression import setup, compare_models, tune_model, pull
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('salary_data_cleaned.csv')
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select", ["Introduction", "Data Visualization", "Modeling", "AI Explainability", "Hypperparameter Tuning"])

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
elif page == "Hypperparameter Tuning":
    st.title("🤖 Hyperparameter Tuning with PyCaret + MLflow via DAGsHub")

    from pycaret.regression import setup, compare_models, tune_model, pull
    import dagshub
    import mlflow

    # Initialize DAGsHub logging
    dagshub.init(repo_owner='anas-ozeer', repo_name='salary_prediction', mlflow=True)

    st.markdown("We’re running multiple regression models and tuning the best one. Everything is logged to DAGsHub with MLflow.")

    # Prepare dataset
    df_model = df.dropna()  # Make sure there are no missing values
    target_column = 'avg_salary'

    if st.button("🚀 Run PyCaret Pipeline"):
        with st.spinner("Training and tuning models..."):
            # Setup PyCaret
            reg_setup = setup(
                data=df_model,
                target=target_column,
                session_id=42,
                log_experiment=True,
                experiment_name="salary_regression",
                silent=True,
                verbose=False
            )

            # Compare models
            best_model = compare_models()
            st.success("✅ Best Model Found!")

            # Show leaderboard
            leaderboard_df = pull()
            st.dataframe(leaderboard_df)

            # Tune best model
            tuned_model = tune_model(best_model)
            st.success("✅ Tuned Best Model")

            # Pull tuned model metrics
            metrics_df = pull()
            st.subheader("📊 Tuned Model Performance")
            st.dataframe(metrics_df)

            st.markdown("🔗 [View Full Logs on DAGsHub](https://dagshub.com/anas-ozeer/salary_prediction/experiments)")
    else:
        st.info("Click the button to run model comparison and hyperparameter tuning.")
