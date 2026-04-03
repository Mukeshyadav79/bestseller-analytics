import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce


@st.cache_resource
def train_model(df):
    # Features & Target
    features = ['Genre', 'Reviews_Log', 'User Rating',
                'Age_of_Book', 'Author_Popularity']
    target = 'Price'

    X = df[features].copy()
    y = df[target].copy()

    # Target Encoding for categorical variables
    encoder = ce.TargetEncoder(cols=['Genre'])
    X_encoded = encoder.fit_transform(X, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances}).sort_values(
        'Importance', ascending=False)

    return model, encoder, rmse, mae, r2, feat_imp_df, X_encoded.columns
