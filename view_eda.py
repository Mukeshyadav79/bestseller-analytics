import streamlit as st
import plotly.express as px

def render_eda(df):
    st.header("2. Exploratory Data Analysis (EDA)")
    st.markdown("Explore deep insights through various analytical dimensions.")
    
    eda_tabs = st.tabs([
        "📊 Univariate Analysis", 
        "📈 Bivariate Analysis", 
        "⏳ Time Series Trends", 
        "🧩 Correlations & Insights"
    ])
    
    with eda_tabs[0]:
        st.subheader("Distributions of Key Metrics")
        col1, col2 = st.columns(2)
        with col1:
            fig_price = px.histogram(df, x='Price', nbins=50, marginal='box', 
                                     color_discrete_sequence=['#1f77b4'], 
                                     title="Price Distribution")
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            fig_rating = px.histogram(df, x='User Rating', nbins=20, marginal='violin', 
                                      color_discrete_sequence=['#ff7f0e'], 
                                      title="User Rating Distribution")
            st.plotly_chart(fig_rating, use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            fig_reviews = px.histogram(df, x='Reviews', nbins=50, marginal='box', log_x=True,
                                     color_discrete_sequence=['#2ca02c'], 
                                     title="Reviews Distribution (Log Scale)")
            st.plotly_chart(fig_reviews, use_container_width=True)
        with col4:
            fig_genre = px.pie(df, names='Genre', title="Proportion of Fiction vs Non-Fiction", hole=0.4, 
                               color_discrete_map={'Fiction':'#9467bd', 'Non Fiction':'#8c564b'})
            fig_genre.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_genre, use_container_width=True)

    with eda_tabs[1]:
        st.subheader("Relationships Between Variables")
        
        # Scatter: Reviews vs Price
        fig_scatter = px.scatter(df, x='Reviews', y='Price', color='Genre', opacity=0.6, 
                                 hover_data=['Name', 'Author'], log_x=True,
                                 title="Reviews vs Price Segmented by Genre")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Boxplots: Genre vs variables
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_box_price = px.box(df, x='Genre', y='Price', color='Genre', 
                                   title="Price Variation across Genres")
            st.plotly_chart(fig_box_price, use_container_width=True)
        with col_b2:
            fig_box_rating = px.box(df, x='Genre', y='User Rating', color='Genre', 
                                    title="Rating Variation across Genres")
            st.plotly_chart(fig_box_rating, use_container_width=True)
            
        # Top Authors
        st.subheader("Top 10 Authors by Bestseller Count")
        top_authors = df['Author'].value_counts().reset_index().head(10)
        top_authors.columns = ['Author', 'Books Count']
        fig_authors = px.bar(top_authors, x='Books Count', y='Author', orientation='h', 
                             color='Books Count', color_continuous_scale='Blues',
                             title="Top 10 Bestselling Authors")
        fig_authors.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_authors, use_container_width=True)

    with eda_tabs[2]:
        st.subheader("Temporal Trends (2009 - 2019)")
        
        # Yearly aggregation
        yearly_trends = df.groupby(['Year', 'Genre']).agg({
            'Price': 'mean',
            'Reviews': 'mean',
            'User Rating': 'mean'
        }).reset_index()

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_trend_price = px.line(yearly_trends, x='Year', y='Price', color='Genre', markers=True,
                                      title="Average Price Trend over Time")
            st.plotly_chart(fig_trend_price, use_container_width=True)
        with col_t2:
            fig_trend_reviews = px.line(yearly_trends, x='Year', y='Reviews', color='Genre', markers=True,
                                        title="Average Reviews Trend over Time")
            st.plotly_chart(fig_trend_reviews, use_container_width=True)
            
        # Overall User Rating Trend
        fig_trend_rating = px.line(yearly_trends, x='Year', y='User Rating', color='Genre', markers=True,
                                   title="Average User Rating Trend over Time")
        st.plotly_chart(fig_trend_rating, use_container_width=True)

    with eda_tabs[3]:
        st.subheader("Correlation Heatmap & Multivariate Analysis")
        
        col_c1, col_c2 = st.columns([1.5, 1])
        with col_c1:
            # Correlation Heatmap
            numerical_cols = ['User Rating', 'Reviews_Log', 'Price', 'Age_of_Book', 'Author_Popularity']
            corr = df[numerical_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', 
                                 title="Correlation Matrix (Numerical Features)")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col_c2:
            # Insights
            st.markdown("### Automated Insights")
            fiction_avg = df[df['Genre'] == 'Fiction']['Price'].mean()
            nonfiction_avg = df[df['Genre'] == 'Non Fiction']['Price'].mean()
            
            st.info(f"💡 **Pricing:** Non-Fiction books average **${nonfiction_avg:.2f}**, whereas Fiction average **${fiction_avg:.2f}**.")
            
            high_review_rating = df[df['Reviews'] > df['Reviews'].median()]['User Rating'].mean()
            low_review_rating = df[df['Reviews'] <= df['Reviews'].median()]['User Rating'].mean()
            
            st.success(f"📈 **Engagement:** Books with above-average reviews have an average rating of **{high_review_rating:.2f}**, vs **{low_review_rating:.2f}** for below-average.")
            
            # Author insight
            max_author = df['Author'].value_counts().idxmax()
            st.warning(f"🏆 **Top Author:** **{max_author}** has the most appearances on the bestseller list, driving up our 'Author Popularity' feature.")
