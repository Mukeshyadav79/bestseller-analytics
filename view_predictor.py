import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_predictor(df, model, encoder, rmse, mae, r2, feat_imp_df):
    st.header("3. Machine Learning Price Predictor Engine")
    st.markdown("Leverage the trained Random Forest model to estimate the optimal price of a book and analyze strategic pricing scenarios.")

    # Model Performance Metrics
    st.markdown("### 📊 Model Performance")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    col_metrics1.metric("Model RMSE (Root Mean Squared Error)",
                        f"${rmse:.2f}", help="Lower is better. Measures magnitude of error.")
    col_metrics2.metric("Model MAE (Mean Absolute Error)",
                        f"${mae:.2f}", help="Average prediction error in dollars. Used for confidence bounds.")
    col_metrics3.metric(
        "Model R² Score", f"{r2:.2f}", help="Close to 1.0 means the model explains variance extremely well.")

    st.divider()

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Predict a Book's Price")
        with st.form("prediction_form"):
            in_genre = st.selectbox("Genre", df['Genre'].unique())
            in_reviews = st.number_input(
                "Number of Reviews", min_value=0, max_value=200000, value=5000)
            in_rating = st.slider("User Rating", 1.0, 5.0, 4.5, 0.1)
            in_year = st.selectbox("Publication Year", sorted(
                df['Year'].unique(), reverse=True))
            in_popularity = st.number_input(
                "Author Popularity (Books by Author)", min_value=1, max_value=50, value=1)

            # Optional actual price for comparison strategy
            in_actual_price = st.number_input(
                "Actual Current Price ($) (Optional for Strategy)", min_value=0.0, value=0.0)

            submit = st.form_submit_button("Predict Optimal Price")

        st.subheader("Feature Importance")
        fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Viridis',
                         title="Impact of Characteristics on Price")
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        if submit:
            st.subheader("📈 Prediction Results & Strategy")

            # Prepare Input
            input_data = pd.DataFrame([{
                'Genre': in_genre,
                'Reviews_Log': np.log1p(in_reviews),
                'User Rating': in_rating,
                'Age_of_Book': pd.Timestamp.now().year - in_year,
                'Author_Popularity': in_popularity
            }])

            # Encode
            input_encoded = encoder.transform(input_data)

            # Predict
            pred_price = model.predict(input_encoded)[0]

            # Add a realistic pricing range using MAE safely bounded at 0
            lower_bound = max(0, pred_price - mae)
            upper_bound = pred_price + mae

            st.success(f"### Predicted Optimal Price: **${pred_price:.2f}**")
            st.markdown(
                f"**Recommended Pricing Range:** `${lower_bound:.2f} - ${upper_bound:.2f}` *(Based on Model MAE confidence bounds)*")

            # Strategy Engine Gauge
            if in_actual_price > 0:
                diff = pred_price - in_actual_price

                # Gauge chart for Price Status
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=in_actual_price,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={
                        'text': "Market Pricing Alignment<br><span style='font-size:0.8em;color:gray'>Current Price vs Expected</span>", 'font': {'size': 20}},
                    delta={'reference': pred_price, 'increasing': {
                        'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, max(in_actual_price, pred_price) * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            # Underpriced zone (Reddish)
                            {'range': [0, pred_price - mae],
                                'color': 'rgba(255, 99, 71, 0.4)'},
                            # Sweet spot (Greenish)
                            {'range': [pred_price - mae, pred_price + mae],
                                'color': 'rgba(144, 238, 144, 0.4)'},
                            # Overpriced zone (Orange)
                            {'range': [
                                pred_price + mae, max(in_actual_price, pred_price) * 1.5], 'color': 'rgba(255, 165, 0, 0.4)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': pred_price
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Adjusted logic to account for the MAE bound
                if in_actual_price < (pred_price - mae):
                    st.warning(
                        f"🚨 **Strategy Insight:** Book is significantly underpriced outside the model threshold. **Increase price by ~${diff:.2f} to maximize margin.**")
                elif in_actual_price > (pred_price + mae):
                    st.error(
                        f"🚨 **Strategy Insight:** Book is overpriced outside the model threshold. **Consider a markdown of ~${abs(diff):.2f} to accelerate volume.**")
                else:
                    st.success(
                        "✅ **Strategy Insight:** Book is competitively priced within the optimal market sweet spot.")

            # What-If Simulator
            st.markdown("---")
            st.subheader("What-If Analysis (Sensitivity Tuning)")
            st.markdown(
                "Observe how changing the **User Rating** influences the predicted optimal price, holding other factors constant (Conceptual Partial Dependence).")

            # Simulate ratings from 3.0 to 5.0
            simulated_ratings = np.linspace(3.0, 5.0, 21)
            simulated_prices = []

            for sim_rating in simulated_ratings:
                sim_data = input_data.copy()
                sim_data['User Rating'] = sim_rating
                sim_encoded = encoder.transform(sim_data)
                sim_price = model.predict(sim_encoded)[0]
                simulated_prices.append(sim_price)

            sim_df = pd.DataFrame({
                'User Rating': simulated_ratings,
                'Estimated Price': simulated_prices
            })

            fig_sim = px.line(sim_df, x='User Rating', y='Estimated Price', markers=True,
                              title="Simulated Price Elasticity vs. User Rating",
                              labels={"Estimated Price": "Predicted Price ($)", "User Rating": "Simulated Rating"})

            # Highlight current user rating
            fig_sim.add_vline(x=in_rating, line_dash="dash", line_color="red",
                              annotation_text="Current Input Rating", annotation_position="bottom right")
            st.plotly_chart(fig_sim, use_container_width=True)

        else:
            st.info("👈 Enter the book details on the left and click **Predict Optimal Price** to generate advanced analytics, a strategy gauge, and what-if simulation.")
