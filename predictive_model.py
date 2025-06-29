import streamlit as st
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, pd.Index, LabelEncoder]:
    """Trains a RandomForest model to predict Price Tier."""
    features = [
        'Release Year', 'RAM (GB)', 'Battery Capacity', 'Screen Diagonal (inches)',
        'Rear Cameras', 'Pixels Per Inch', 'Weight (grams)', 'Brand'
    ]
    target = 'Price Tier'
    
    model_df = df[features + [target]].dropna()
    
    if model_df.empty:
        raise ValueError("No valid data available for model training after removing rows with missing values. "
                         "This often means one or more key columns in the source CSV are completely empty.")

    X = pd.get_dummies(model_df[features], columns=['Brand'], prefix='Brand')
    le = LabelEncoder()
    y_encoded = le.fit_transform(model_df[target])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X, y_encoded)
    
    st.session_state['oob_score'] = model.oob_score_
    
    return model, X.columns, le