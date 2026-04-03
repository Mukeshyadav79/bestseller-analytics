import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_and_preprocess_data():

    df = pd.read_csv("bestsellers with categories.csv")

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Feature Engineering
    # 1. Log transform Reviews (to handle right skewness)
    df['Reviews_Log'] = np.log1p(df['Reviews'])

    # 2. Author Popularity (appearances in the list)
    author_counts = df['Author'].value_counts()
    df['Author_Popularity'] = df['Author'].map(author_counts)

    # 3. Age of Book
    current_year = pd.Timestamp.now().year
    df['Age_of_Book'] = current_year - df['Year']

    return df
