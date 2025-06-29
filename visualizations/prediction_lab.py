import streamlit as st
import pandas as pd
import plotly.express as px

def render(df, model, model_columns, label_encoder, min_year, max_year):
    st.header("🔮 Can We Predict a Phone's Price Tier?")
    st.markdown("Use the controls below to design a hypothetical smartphone, and our trained machine learning model will predict whether it's a **Budget**, **Mid-range**, or **Premium** device.")
    st.info("The model was trained on historical data from 2004-2020. Its accuracy is not guaranteed but provides an educated guess based on past trends.", icon="🤖")

    st.sidebar.divider()
    st.sidebar.header("Prediction Inputs")

    pred_year = st.sidebar.slider("Release Year (for prediction)", min_year, max_year, max_year)
    pred_ram = st.sidebar.select_slider("RAM (GB)", options=[1, 2, 3, 4, 6, 8, 12, 16], value=8)
    pred_battery = st.sidebar.slider("Battery (mAh)", 1000, 7000, 4500, 100)
    pred_screen = st.sidebar.slider("Screen Diagonal (inches)", 4.0, 7.5, 6.5, 0.1)
    pred_cameras = st.sidebar.slider("Number of Rear Cameras", 1, 5, 3)
    pred_ppi = st.sidebar.slider("Pixels Per Inch (PPI)", 100, 600, 400)
    pred_weight = st.sidebar.slider("Weight (grams)", 100, 250, 180)
    pred_brand = st.sidebar.selectbox("Brand", options=sorted(df['Brand'].unique()))

    input_data = pd.DataFrame([{'Release Year': pred_year, 'RAM (GB)': pred_ram, 'Battery Capacity': pred_battery,
                                'Screen Diagonal (inches)': pred_screen, 'Rear Cameras': pred_cameras,
                                'Pixels Per Inch': pred_ppi, 'Weight (grams)': pred_weight, 'Brand': pred_brand}])
    
    input_encoded = pd.get_dummies(input_data, columns=['Brand'], prefix='Brand')
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction_encoded = model.predict(input_aligned)
    prediction_proba = model.predict_proba(input_aligned)
    predicted_tier = label_encoder.inverse_transform(prediction_encoded)[0]

    st.subheader("Prediction Results")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Predicted Price Tier", predicted_tier.upper())
        st.write(f"Model Out-of-Bag Score: {st.session_state.get('oob_score', 0):.2%}")
        st.caption("This score estimates the model's accuracy on unseen data.")
    with col2:
        proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
        st.markdown("**Prediction Confidence**")
        st.bar_chart(proba_df.T)
        
    st.divider()
    st.subheader("Which Features Matter Most to the Model?")
    feature_importances = pd.DataFrame({'feature': model_columns, 'importance': model.feature_importances_})
    fig_importance = px.bar(feature_importances.sort_values('importance', ascending=False).head(15),
                             x='importance', y='feature', orientation='h',
                             title='Top 15 Most Important Features for Price Tier Prediction')
    st.plotly_chart(fig_importance, use_container_width=True)