import streamlit as st
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('salary_data_cleaned.csv')
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Data Visualization", "Modeling"])

# Data Exploration Page
if page == "Introduction":
    # Title and Introduction
    st.title("📊 Job Postings Data Analysis App")

    st.markdown("""
    Welcome to the **Job Postings Data Analysis App**!  
    This application allows you to explore and analyze a dataset of job postings to gain insights into hiring trends, job roles, industries, and locations.

    With this tool, you can:
    - Examine the distribution of job postings by industry, job function, and location.
    - Identify popular job titles and companies.
    - Visualize key metrics and trends interactively.

    Upload your own dataset or use the default dataset to get started.
    """)

    st.subheader("Dataset Overview")
    st.dataframe(df.head())


    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # Optional filtering
    # Filters inside the page
    with st.expander("🔍 Filter Options"):
        col1, col2 = st.columns(2)

        with col1:
            selected_industries = st.multiselect(
                "Select Industry:",
                options=df['industry'].dropna().unique(),
                key="industry_filter"
            )

        with col2:
            selected_locations = st.multiselect(
                "Select Location:",
                options=df['location'].dropna().unique(),
                key="location_filter"
            )


    filtered_df = df.copy()

    if selected_industries:
        filtered_df = filtered_df[filtered_df['industry'].isin(selected_industries)]

    if selected_locations:
        filtered_df = filtered_df[filtered_df['location'].isin(selected_industries)]

    # Show filtered results
    st.subheader("Filtered Job Postings")
    st.write(f"Total jobs found: {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head(20))


    st.subheader("Basic Info")
    df.info()

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Unique Values in Categorical Columns")
    for col in df.columns:
        if df[col].dtype == 'object':
            st.write(f"**{col}**: {df[col].nunique()} unique values")

    if 'salary' in df.columns:
        st.subheader("Salary Distribution")
        st.write(df['salary'].describe())

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
