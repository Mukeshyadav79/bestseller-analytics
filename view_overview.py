import streamlit as st

def render_overview(df):
    st.header("1. Dataset Overview")
    st.markdown("A comprehensive look at the raw and pre-processed Amazon Bestselling Books dataset (2009-2019).")
    
    # --- Top Level Metrics ---
    st.subheader("At a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Books (Unique Entries)", value=f"{len(df):,}")
    with col2:
        st.metric(label="Average Price", value=f"${df['Price'].mean():.2f}")
    with col3:
        st.metric(label="Average Rating", value=f"{df['User Rating'].mean():.2f}/5.0")
    with col4:
        st.metric(label="Total Reviews Processed", value=f"{df['Reviews'].sum():,}")
        
    st.divider()

    # --- Data Presentation ---
    st.subheader("Data Sample")
    st.dataframe(df.head(100), use_container_width=True)
        
    st.divider()
    
    # --- Statistical Description & Extremes ---
    col_s1, col_s2 = st.columns([1.5, 1])
    with col_s1:
        st.subheader("Statistical Summary")
        # Transpose the describe matrix so features are rows, easier to read
        st.dataframe(df.describe().T, use_container_width=True) 
        
    with col_s2:
        st.subheader("Dataset Highlights")
        
        # Most Reviewed
        most_reviewed = df.loc[df['Reviews'].idxmax()]
        st.info(f"**Most Reviewed Book:**\n\n*{most_reviewed['Name']}* by {most_reviewed['Author']} ({most_reviewed['Reviews']:,} reviews)")
        
        # Most Expensive
        most_expensive = df.loc[df['Price'].idxmax()]
        st.error(f"**Most Expensive Book:**\n\n*{most_expensive['Name']}* by {most_expensive['Author']} (${most_expensive['Price']})")
        
        # Perfect 5.0 Rating with Most Reviews
        perfect_books = df[df['User Rating'] == 5.0]
        if not perfect_books.empty:
            best_perfect = perfect_books.loc[perfect_books['Reviews'].idxmax()]
            st.success(f"**Top Perfect 5.0 Rating:**\n\n*{best_perfect['Name']}* by {best_perfect['Author']} ({best_perfect['Reviews']:,} reviews)")
