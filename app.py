import streamlit as st

# Import your modules
from data_processing import load_data
from predictive_model import train_model
from utils import calculate_hhi
from visualizations import insights, market_dynamics, manufacturing, hardware_trends, prediction_lab

def main():
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Smartphone Industry Analysis",
        page_icon="📱",
        layout="wide"
    )

    st.title("📱 Smartphone Industry: An Interactive Analysis")
    st.markdown("An interactive dashboard to explore trends, competition, and technological evolution in the smartphone market, now with predictive capabilities.")

    # --- DATA LOADING ---
    try:
        df = load_data('./smartphonedata.csv')
    except FileNotFoundError:
        st.error("Error: `smartphonedata.csv` not found. Please place it in the same directory as this app.py file.")
        st.stop()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔬 Global Filters")
    min_year, max_year = int(df['Release Year'].min()), int(df['Release Year'].max())
    selected_years = st.sidebar.slider("Analysis Year Range", min_year, max_year, (2010, 2020))
    analysis_df = df[df['Release Year'].between(selected_years[0], selected_years[1])]
    
    st.sidebar.header("Highlight Brands")
    all_brands = sorted(analysis_df['Brand'].dropna().unique())
    highlighted_brands = st.sidebar.multiselect("Select brands to highlight in charts:", options=all_brands, default=['Samsung', 'Apple', 'Huawei', 'Xiaomi', 'Oppo', 'Nokia'])

    # --- TAB LAYOUT ---
    tabs = st.tabs([
        "📊 Meaningful Market Insights",
        "📈 Market Dynamics & Competition",
        "🌍 Global Manufacturing Landscape",
        "⚙️ Hardware & Feature Trends",
        "🔮 Prediction Lab"
    ])

    with tabs[0]:
        insights.render(analysis_df)

    with tabs[1]:
        market_dynamics.render(analysis_df, hhi_calculator=calculate_hhi)
    
    with tabs[2]:
        manufacturing.render(analysis_df)

    with tabs[3]:
        hardware_trends.render(analysis_df, highlighted_brands)

    with tabs[4]:
        try:
            model, model_columns, label_encoder = train_model(df)
            prediction_lab.render(df, model, model_columns, label_encoder, min_year, max_year)
        except ValueError as e:
            st.error(f"⚠️ **Model Training Failed:** {e}")
            st.warning("This error typically means your source CSV file is missing a critical column or has no valid data in a column needed for the model (e.g., 'RAM (GB)', 'Battery Capacity'). Please check your data file.")
        except Exception as e:
            st.error(f"An unexpected error occurred during the prediction process: {e}")


if __name__ == "__main__":
    main()