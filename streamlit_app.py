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
page = st.sidebar.radio("Go to", ["Data Exploration", "Data Visualization", "Modeling"])

# Data Exploration Page
if page == "Data Exploration":
    st.title("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

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
    st.write("This page is under construction. 📊")

# Modeling Page
elif page == "Modeling":
    st.title("Modeling")
    st.write("This page is under construction. 🤖")
