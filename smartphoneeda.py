import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from typing import Union, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smartphone Industry Analysis",
    page_icon="📱",
    layout="wide"
)

# --- DATA LOADING & PREPARATION ---
@st.cache_data
def load_data(file_path: Union[BytesIO, str]) -> pd.DataFrame:
    """Loads, cleans, and prepares the smartphone data for analysis."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    rename_map = {
        'Display Diagonal (inches)': 'Screen Diagonal (inches)',
        'RAM Memory (GB)': 'RAM (GB)',
        'Weight (g)': 'Weight (grams)'
    }
    df.rename(columns=rename_map, inplace=True)

    yes_no_cols = [
        'Battery Removable', 'Wide Angle Lens', 'Telephoto Lens', 'NFC',
        'HDR Display', 'Fingerprint', 'Wireless Charging', '5G', 'Dual SIM',
        '4K Video Recording', '8K Video Recording', 'Always-on Display',
        'Battery Fast Charge', 'Display Notch'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip().map({'yes': True, 'no': False}).astype('boolean')

    if 'USB type C' in df.columns:
        df['USB type C'] = ((df['USB type C'].astype(str).str.lower().str.strip() != 'no') &
                           (df['USB type C'].astype(str).str.lower().str.strip() != '') &
                           (df['USB type C'].notna())).astype('boolean')

    # --- FIX IS HERE: More robust handling of numeric columns ---
    numeric_cols = [
        'Rear Cameras', 'Battery Capacity', 'Thickness (mm)', 'Weight (grams)',
        'Screen Diagonal (inches)', 'Rear Camera MP', 'RAM (GB)',
        'Pixels Per Inch', 'CPU Frequency (GHz)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            col_median = df[col].median()
            
            # Check if the column was entirely empty, resulting in a NaN median
            if pd.isna(col_median):
                st.warning(f"Warning: Column '{col}' contains no valid numbers. Filling missing values with 0.")
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(col_median, inplace=True)

    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df['Release Year'] = df['Release Year'].fillna(df['Release Date'].dt.year)
    df.dropna(subset=['Release Year', 'Brand', 'Brand Country'], inplace=True)
    df['Release Year'] = df['Release Year'].astype(int)

    # Feature Engineering
    df['Spec Score'] = (df['RAM (GB)'] * 20 + df['Rear Camera MP'] * 2 +
                        df['Pixels Per Inch'] * 1 + df['Battery Capacity'] * 0.05)
    
    aspect_ratio = 18 / 9
    df['Screen Area (sq inches)'] = (df['Screen Diagonal (inches)']**2 / (aspect_ratio**2 + 1)) * aspect_ratio
    df['Phone Density (g/sq inch)'] = df['Weight (grams)'] / df['Screen Area (sq inches)']

    low_bound, high_bound = df['Spec Score'].quantile([0.33, 0.66])
    df['Price Tier'] = pd.cut(df['Spec Score'], bins=[0, low_bound, high_bound, df['Spec Score'].max()],
                              labels=['Budget', 'Mid-range', 'Premium'], include_lowest=True)
    
    yearly_spec_quantile = df.groupby('Release Year')['Spec Score'].transform(lambda x: x.quantile(0.85))
    df['Is Flagship'] = (df['Spec Score'] >= yearly_spec_quantile) & (df['Price Tier'] == 'Premium')

    return df

# --- ML MODEL TRAINING ---
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

# --- UI & APP LOGIC ---
st.title("📱 Smartphone Industry: An Interactive Analysis")
st.markdown("An interactive dashboard to explore trends, competition, and technological evolution in the smartphone market, now with predictive capabilities.")

try:
    df = load_data('./smartphonedata.csv')
except FileNotFoundError:
    st.error("Error: `smartphonedata.csv` not found. Please place it in the same directory.")
    st.stop()

# --- SIDEBAR ---
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

# Code for other tabs remains the same...
with tabs[0]:
    st.header("At a Glance: Key Stories from the Data")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Growth of Screen Real Estate")
        avg_screen_area = analysis_df.groupby('Release Year')['Screen Area (sq inches)'].mean().reset_index()
        fig_area = px.area(avg_screen_area, x='Release Year', y='Screen Area (sq inches)', markers=True, title='Average Smartphone Screen Area Over Time')
        st.plotly_chart(fig_area, use_container_width=True)
    with col2:
        st.subheader("The Flagship Race")
        flagship_counts = analysis_df[analysis_df['Is Flagship']].groupby(['Release Year', 'Brand']).size().unstack().fillna(0)
        top_flagship_brands = flagship_counts.sum().nlargest(5).index
        fig_flagship = px.bar(flagship_counts[top_flagship_brands], title='Number of "Flagship" Models Released by Top Brands')
        st.plotly_chart(fig_flagship, use_container_width=True)
    st.subheader("The Rise and Fall of Market Leaders")
    market_share_df_insight = analysis_df.groupby(['Release Year', 'Brand']).size().unstack(fill_value=0)
    market_share_df_insight = market_share_df_insight.apply(lambda x: x / x.sum(), axis=1)
    top_brands_insight = market_share_df_insight.sum().nlargest(10).index
    ranked_df_insight = market_share_df_insight[top_brands_insight].rank(axis=1, method='min', ascending=False)
    plot_data_insight = ranked_df_insight.reset_index().melt(id_vars='Release Year', value_name='Rank', var_name='Brand')
    fig_bump = px.line(plot_data_insight, x='Release Year', y='Rank', color='Brand', markers=True, title='Market Share Ranking of Top 10 Brands')
    fig_bump.update_yaxes(autorange="reversed", tickvals=list(range(1, 11)))
    st.plotly_chart(fig_bump, use_container_width=True)

with tabs[1]:
    st.header("How Competitive is the Smartphone Market?")
    st.subheader("Measuring Market Concentration (HHI)")
    def calculate_hhi(df, year):
        year_df = df[df['Release Year'] == year]
        if year_df.empty: return np.nan
        market_shares = year_df['Brand'].value_counts(normalize=True) * 100
        return (market_shares ** 2).sum()
    hhi_data = [calculate_hhi(analysis_df, year) for year in range(selected_years[0], selected_years[1] + 1)]
    hhi_df = pd.DataFrame({'Year': range(selected_years[0], selected_years[1] + 1), 'HHI': hhi_data}).dropna()
    fig_hhi = px.line(hhi_df, x='Year', y='HHI', title='Market Concentration (HHI) Over Time', markers=True)
    fig_hhi.add_hline(y=1500, line_dash="dash", line_color="green", annotation_text="Competitive")
    fig_hhi.add_hline(y=2500, line_dash="dash", line_color="red", annotation_text="Highly Concentrated")
    st.plotly_chart(fig_hhi, use_container_width=True)
    st.subheader("Market Share of Top Brands Over Time")
    market_share_df = analysis_df.groupby(['Release Year', 'Brand']).size().unstack(fill_value=0)
    market_share_df = market_share_df.apply(lambda x: x / x.sum(), axis=1)
    top_by_count = market_share_df.sum().nlargest(8).index.tolist()
    must_have_brands = ['Apple', 'Samsung']
    final_brands_to_show = list(dict.fromkeys(top_by_count + must_have_brands))
    final_brands_to_show = [b for b in final_brands_to_show if b in market_share_df.columns]
    fig_market_share = px.area(market_share_df[final_brands_to_show], title='Market Share by Variety of Models Released', labels={'value': 'Market Share', 'Release Year': 'Year'})
    st.plotly_chart(fig_market_share, use_container_width=True)

with tabs[2]:
    st.header("What is the current state of manufacturing in countries?")
    st.subheader("Global Concentration of Smartphone Brands")
    country_counts = analysis_df['Brand Country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_iso_map = px.data.gapminder()[['country', 'iso_alpha']].drop_duplicates()
    map_data = pd.merge(country_counts, country_iso_map, on='country', how='left')
    fig_map = px.choropleth(map_data, locations="iso_alpha", color="count", hover_name="country", color_continuous_scale=px.colors.sequential.Plasma, title="Number of Smartphone Models by Brand Country of Origin")
    st.plotly_chart(fig_map, use_container_width=True)

with tabs[3]:
    st.header("How are Smartphone Specs and Features Evolving?")
    st.subheader("Are All Phones Becoming the Same? (Hardware Convergence)")
    commoditization_metric = st.selectbox("Select a Specification to Analyze:", options=['Battery Capacity', 'Screen Diagonal (inches)', 'Pixels Per Inch', 'Weight (grams)'])
    pivot_df = analysis_df[analysis_df['Brand'].isin(highlighted_brands)].pivot_table(index='Release Year', columns='Brand', values=commoditization_metric, aggfunc='median')
    fig_commoditization = px.line(pivot_df, title=f'Median {commoditization_metric} for Highlighted Brands', labels={'value': commoditization_metric, 'Release Year': 'Year'}, markers=True)
    st.plotly_chart(fig_commoditization, use_container_width=True)

# --- TAB 5: PREDICTION LAB ---
with tabs[4]:
    st.header("🔮 Can We Predict a Phone's Price Tier?")
    st.markdown("Use the controls below to design a hypothetical smartphone, and our trained machine learning model will predict whether it's a **Budget**, **Mid-range**, or **Premium** device.")
    st.info("The model was trained on historical data from 2004-2020. Its accuracy is not guaranteed but provides an educated guess based on past trends.", icon="🤖")

    try:
        model, model_columns, label_encoder = train_model(df)
        
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

        input_data = pd.DataFrame([{'Release Year': pred_year, 'RAM (GB)': pred_ram, 'Battery Capacity': pred_battery, 'Screen Diagonal (inches)': pred_screen, 'Rear Cameras': pred_cameras, 'Pixels Per Inch': pred_ppi, 'Weight (grams)': pred_weight, 'Brand': pred_brand}])
        
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
        st.markdown("This chart shows which specifications were most influential for the model when it was learning to predict the price tier.")
        
        feature_importances = pd.DataFrame({'feature': model_columns, 'importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False).head(15)
        
        fig_importance = px.bar(feature_importances.sort_values('importance', ascending=True), x='importance', y='feature', orientation='h', title='Top 15 Most Important Features for Price Tier Prediction')
        st.plotly_chart(fig_importance, use_container_width=True)

    except ValueError as e:
        st.error(f"⚠️ **Model Training Failed:** {e}")
        st.warning("This error typically means your source `smartphonedata.csv` file is missing a critical column or has no valid data in a column needed for the model (e.g., 'RAM (GB)', 'Battery Capacity'). Please check your data file.")
    except Exception as e:
        st.error(f"An unexpected error occurred during the prediction process: {e}")