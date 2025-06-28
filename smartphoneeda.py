#if yo code chitta bujhena bhaney tala comment gareko code run gara hai upto 363lines
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Smartphone EDA Dashboard", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------------

def load_data(uploaded_file: BytesIO | str):
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.session_state["raw"] = df.copy()
    df.columns = df.columns.str.strip()

    if "Release Date" in df.columns:
        df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
        df["Release Year"] = df["Release Year"].fillna(df["Release Date"].dt.year)

    for col in df.select_dtypes(include="object").columns:
        uniques = df[col].dropna().str.lower().str.strip().unique()
        if set(uniques).issubset({"yes", "no", "y", "n"}):
            df[col] = df[col].str.lower().str.strip().map({"yes": True, "y": True, "no": False, "n": False})

    num_cols = df.select_dtypes(include=["float64", "int64", "object"]).columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    if "High Refresh Rate (Hz)" in df.columns:
        df["High Refresh"] = df["High Refresh Rate (Hz)"].gt(60)

    if "5G" in df.columns:
        df["has_5G"] = df["5G"].astype(bool)

    return df

# Sidebar: Data source --------------------------------------------
st.sidebar.header("📁 Data Source")
uploaded = st.sidebar.file_uploader("Upload smartphone CSV", type=["csv"])
DEFAULT_PATH = "/Users/anushashrestha/Documents/7thsem/DM/project/smartphonedata.csv"

df = load_data(uploaded if uploaded else DEFAULT_PATH)

# Sidebar: Filters -------------------------------------------------
st.sidebar.header("🔍 Filters")

brands = df["Brand"].dropna().sort_values().unique()
selected_brands = st.sidebar.multiselect("Brand", brands, default=list(brands)[:10])

min_year, max_year = int(df["Release Year"].min()), int(df["Release Year"].max())
year_range = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))

bool_filters = {}
for boolean_col in ["has_5G", "Wireless Charging", "Fingerprint", "High Refresh"]:
    if boolean_col in df.columns:
        bool_filters[boolean_col] = st.sidebar.checkbox(boolean_col.replace("_", " "), value=False)

mask = (
    df["Brand"].isin(selected_brands) &
    df["Release Year"].between(year_range[0], year_range[1])
)
for col, flag in bool_filters.items():
    if flag:
        mask &= df[col]

filtered_df = df[mask]

st.sidebar.markdown(f"**Filtered Rows:** {len(filtered_df):,} / {len(df):,}")

# ------------------------------------------------------------------
# 2. MAIN LAYOUT
# ------------------------------------------------------------------

st.title("📱 Smartphone Specs EDA Dashboard")
st.markdown("Interactively explore trends and distributions in smartphone specifications from 2004‑2025.")

Overview, Distributions, Trends, Correlation, Innovation, Data = st.tabs([
    "Overview", "Distributions", "Trends", "Correlation", "Innovation", "Data"])

with Overview:
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", f"{len(filtered_df):,}")
    if "Battery Capacity" in filtered_df.columns:
        c2.metric("Avg Battery (mAh)", f"{filtered_df['Battery Capacity'].mean():.0f}")
    if "RAM Memory (GB)" in filtered_df.columns:
        c3.metric("Avg RAM (GB)", f"{filtered_df['RAM Memory (GB)'].mean():.1f}")
    if "Pixels Per Inch" in filtered_df.columns:
        c4.metric("Avg PPI", f"{filtered_df['Pixels Per Inch'].mean():.0f}")

    st.divider()
    st.subheader("Releases by Year")
    releases = filtered_df.groupby("Release Year").size().reset_index(name="count")
    fig = px.bar(releases, x="Release Year", y="count", title="Number of Smartphone Models Released Per Year")
    st.plotly_chart(fig, use_container_width=True)

with Distributions:
    st.subheader("Numeric Feature Distribution")
    numeric_cols = filtered_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if numeric_cols:
        dist_col = st.selectbox("Select numeric column", numeric_cols, index=numeric_cols.index("Battery Capacity") if "Battery Capacity" in numeric_cols else 0)
        hist = px.histogram(filtered_df, x=dist_col, nbins=40, title=f"Distribution of {dist_col}")
        st.plotly_chart(hist, use_container_width=True)

with Trends:
    st.subheader("Trend Over Time")
    trend_options = {
        "Average Battery Capacity": "Battery Capacity",
        "Average RAM (GB)": "RAM Memory (GB)",
        "Average CPU Frequency (GHz)": "CPU Frequency (GHz)",
    }
    available_trends = {k: v for k, v in trend_options.items() if v in filtered_df.columns}
    trend_label = st.selectbox("Metric", list(available_trends.keys()))
    trend_col = available_trends[trend_label]

    trend_df = filtered_df.groupby("Release Year")[trend_col].mean().reset_index()
    line = px.line(trend_df, x="Release Year", y=trend_col, title=f"{trend_label} by Year")
    st.plotly_chart(line, use_container_width=True)

with Correlation:
    st.subheader("Correlation Heatmap (Numeric Features)")
    num_df = filtered_df.select_dtypes(include=["float64", "int64"]).corr()
    heat = px.imshow(num_df, text_auto=".2f")
    st.plotly_chart(heat, use_container_width=True)

with Innovation:
    st.subheader("Brand Adaptation to New Features")

    trend_features = ["has_5G", "Wireless Charging", "High Refresh", "Fingerprint"]
    recent_df = df[df["Release Year"] >= 2015]

    brand_feature_adoption = (
        recent_df.groupby("Brand")[trend_features]
        .mean()
        .sort_values(by="has_5G", ascending=False)
        .round(2)
    )

    st.dataframe(brand_feature_adoption.style.format("{:.2%}"), use_container_width=True)

    st.markdown("**Top Innovative Brands (2015 and later)**")
    for feature in trend_features:
        top = brand_feature_adoption[feature].nlargest(1)
        st.write(f"📌 **{top.index[0]}** leads in adopting **{feature.replace('_', ' ')}** with **{(top.iloc[0]*100):.1f}%** of models.")

    st.subheader("📈 Earliest Feature Adoption")
    earliest_adopters = {}
    for feature in trend_features:
        adopters = df[df[feature] == 1].sort_values("Release Year")
        if not adopters.empty:
            earliest = adopters.iloc[0]
            earliest_adopters[feature] = (earliest["Brand"], int(earliest["Release Year"]))

    for feat, (brand, year) in earliest_adopters.items():
        st.markdown(f"- **{brand}** first adopted **{feat.replace('_', ' ')}** in **{year}**")

    st.subheader("🏆 Innovation Score (weighted average)")
    weights = {"has_5G": 3, "Wireless Charging": 2, "High Refresh": 2, "Fingerprint": 1}
    score_df = brand_feature_adoption.copy()
    score_df["Innovation Score"] = sum(score_df[col] * weight for col, weight in weights.items())
    score_df = score_df.sort_values("Innovation Score", ascending=False)
    st.dataframe(score_df[["Innovation Score"] + trend_features].style.format("{:.2%}"), use_container_width=True)

with Data:
    st.subheader("Raw / Cleaned Dataset")
    st.dataframe(filtered_df, use_container_width=True, height=600)

    csv = filtered_df.to_csv(index=False).encode()
    st.download_button("Download filtered CSV", csv, "filtered_smartphone_data.csv", "text/csv")

    if "raw" in st.session_state:
        raw_csv = st.session_state["raw"].to_csv(index=False).encode()
        st.download_button("Download original CSV", raw_csv, "raw_smartphone_data.csv", "text/csv", type="primary")

st.markdown("""---
**Usage**: Save this script as `smartphone_eda_app.py` and run `streamlit run smartphone_eda_app.py` in your terminal. Upload a CSV with similar structure or use the default dataset provided. © 2025
""")


# # Smartphone EDA & Interactive Dashboard
# # -------------------------------------------------
# # A Streamlit application for exploring smartphone
# # specification data. The app loads the CSV, performs
# # cleaning & feature engineering, and offers interactive
# # visualizations.
# # -------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from io import BytesIO

# st.set_page_config(page_title="Smartphone EDA Dashboard", layout="wide", initial_sidebar_state="expanded")

# # ------------------------------------------------------------------
# # 1. LOAD & CLEAN DATA
# # ------------------------------------------------------------------

# def load_data(uploaded_file: BytesIO | str):
#     """Load the CSV from an UploadedFile or a filepath string, then clean it."""
#     if isinstance(uploaded_file, str):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_csv(uploaded_file)

#     st.session_state["raw"] = df.copy()  # keep raw for downloads

#     # --------------------------------------------------------------
#     # Basic cleaning
#     # --------------------------------------------------------------
#     df.columns = df.columns.str.strip()

#     # Dates
#     if "Release Date" in df.columns:
#         df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
#         df["Release Year"] = df["Release Year"].fillna(df["Release Date"].dt.year)

#     # Detect yes/no columns -> boolean
#     for col in df.select_dtypes(include="object").columns:
#         uniques = df[col].dropna().str.lower().str.strip().unique()
#         if set(uniques).issubset({"yes", "no", "y", "n"}):
#             df[col] = df[col].str.lower().str.strip().map({"yes": True, "y": True, "no": False, "n": False})

#     # Numeric coercion & median imputation
#     num_cols = df.select_dtypes(include=["float64", "int64", "object"]).columns
#     for col in num_cols:
#         # Attempt numeric conversion where possible
#         df[col] = pd.to_numeric(df[col], errors="ignore")
#     num_cols = df.select_dtypes(include=["float64", "int64"]).columns
#     df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#     # Additional feature engineering
#     if "High Refresh Rate (Hz)" in df.columns:
#         df["High Refresh"] = df["High Refresh Rate (Hz)"].gt(60)

#     if "5G" in df.columns:
#         df["has_5G"] = df["5G"].astype(bool)

#     return df

# # Sidebar: Data source --------------------------------------------
# st.sidebar.header("📁 Data Source")
# uploaded = st.sidebar.file_uploader("Upload smartphone CSV", type=["csv"])
# DEFAULT_PATH = "/Users/anushashrestha/Documents/7thsem/DM/project/smartphonedata.csv"

# df = load_data(uploaded if uploaded else DEFAULT_PATH)

# # Sidebar: Filters -------------------------------------------------
# st.sidebar.header("🔍 Filters")

# brands = df["Brand"].dropna().sort_values().unique()
# selected_brands = st.sidebar.multiselect("Brand", brands, default=list(brands)[:10])

# min_year, max_year = int(df["Release Year"].min()), int(df["Release Year"].max())
# year_range = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))

# # Optional boolean filters if columns exist
# bool_filters = {}
# for boolean_col in ["has_5G", "Wireless Charging", "Fingerprint", "High Refresh"]:
#     if boolean_col in df.columns:
#         bool_filters[boolean_col] = st.sidebar.checkbox(boolean_col.replace("_", " "), value=False)

# # Apply filters ----------------------------------------------------
# mask = (
#     df["Brand"].isin(selected_brands) &
#     df["Release Year"].between(year_range[0], year_range[1])
# )
# for col, flag in bool_filters.items():
#     if flag:
#         mask &= df[col]

# filtered_df = df[mask]

# st.sidebar.markdown(f"**Filtered Rows:** {len(filtered_df):,} / {len(df):,}")

# # ------------------------------------------------------------------
# # 2. MAIN LAYOUT
# # ------------------------------------------------------------------

# st.title("📱 Smartphone Specs EDA Dashboard")
# st.markdown("Interactively explore trends and distributions in smartphone specifications from 2004‑2025.")

# # Tabs for organization
# Overview, Distributions, Trends, Correlation, Data = st.tabs([
#     "Overview", "Distributions", "Trends", "Correlation", "Data"])

# with Overview:
#     st.subheader("Key Metrics")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Models", f"{len(filtered_df):,}")
#     if "Battery Capacity" in filtered_df.columns:
#         c2.metric("Avg Battery (mAh)", f"{filtered_df['Battery Capacity'].mean():.0f}")
#     if "RAM Memory (GB)" in filtered_df.columns:
#         c3.metric("Avg RAM (GB)", f"{filtered_df['RAM Memory (GB)'].mean():.1f}")
#     if "Pixels Per Inch" in filtered_df.columns:
#         c4.metric("Avg PPI", f"{filtered_df['Pixels Per Inch'].mean():.0f}")

#     st.divider()
#     st.subheader("Releases by Year")
#     releases = filtered_df.groupby("Release Year").size().reset_index(name="count")
#     fig = px.bar(releases, x="Release Year", y="count", title="Number of Smartphone Models Released Per Year")
#     st.plotly_chart(fig, use_container_width=True)

# with Distributions:
#     st.subheader("Numeric Feature Distribution")
#     numeric_cols = filtered_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
#     if numeric_cols:
#         dist_col = st.selectbox("Select numeric column", numeric_cols, index=numeric_cols.index("Battery Capacity") if "Battery Capacity" in numeric_cols else 0)
#         hist = px.histogram(filtered_df, x=dist_col, nbins=40, title=f"Distribution of {dist_col}")
#         st.plotly_chart(hist, use_container_width=True)

# with Trends:
#     st.subheader("Trend Over Time")
#     trend_options = {
#         "Average Battery Capacity": "Battery Capacity",
#         "Average RAM (GB)": "RAM Memory (GB)",
#         "Average CPU Frequency (GHz)": "CPU Frequency (GHz)",
#     }
#     available_trends = {k: v for k, v in trend_options.items() if v in filtered_df.columns}
#     trend_label = st.selectbox("Metric", list(available_trends.keys()))
#     trend_col = available_trends[trend_label]

#     trend_df = filtered_df.groupby("Release Year")[trend_col].mean().reset_index()
#     line = px.line(trend_df, x="Release Year", y=trend_col, title=f"{trend_label} by Year")
#     st.plotly_chart(line, use_container_width=True)

# with Correlation:
#     st.subheader("Correlation Heatmap (Numeric Features)")
#     num_df = filtered_df.select_dtypes(include=["float64", "int64"]).corr()
#     heat = px.imshow(num_df, text_auto=".2f")
#     st.plotly_chart(heat, use_container_width=True)

# with Data:
#     st.subheader("Raw / Cleaned Dataset")
#     st.dataframe(filtered_df, use_container_width=True, height=600)

#     # Download buttons
#     csv = filtered_df.to_csv(index=False).encode()
#     st.download_button("Download filtered CSV", csv, "filtered_smartphone_data.csv", "text/csv")

#     if "raw" in st.session_state:
#         raw_csv = st.session_state["raw"].to_csv(index=False).encode()
#         st.download_button("Download original CSV", raw_csv, "raw_smartphone_data.csv", "text/csv", type="primary")

# # ------------------------------------------------------------------
# # 3. FOOTER
# # ------------------------------------------------------------------

# st.markdown("""---
# **Usage**: Save this script as `smartphone_eda_app.py` and run `streamlit run smartphone_eda_app.py` in your terminal. Upload a CSV with similar structure or use the default dataset provided. © 2025
# """)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page config
# st.set_page_config(
#     page_title="Smartphone EDA",
#     page_icon="📱",
#     layout="wide"
# )

# # Load data with caching
# @st.cache_data
# def load_data():
#     # Load your dataset here (replace with your actual file path)
#     df = pd.read_csv("/Users/anushashrestha/Documents/7thsem/DM/project/smartphonedata.csv")
    
#     # Keep only specified columns
#     selected_columns = [
#         'ID', 'Brand', 'Release Year', 'Model', 'Release Date',
#         'CPU Frequency (GHz)', 'Battery Capacity', 'Screen-to-body Ratio',
#         'Weight (g)', 'Thickness (mm)', 'Rear Camera MP', 'RAM Memory (GB)',
#         'USB type C', 'Display Notch', 'HDR Display', 'Dual SIM', 'GPS',
#         'Rear Cameras', 'Telephoto Lens', 'Touchscreen Type', 'Multi-touch',
#         'Battery Removable', 'Fingerprint', 'Battery Fast Charge'
#     ]
#     return df[selected_columns]

# df = load_data()

# # ------------------------
# # DATA CLEANING SECTION
# # ------------------------
# st.sidebar.header("🧹 Data Cleaning")

# # Convert to datetime and extract year if needed
# if 'Release Date' in df.columns:
#     df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
#     df['Release Year'] = df['Release Year'].fillna(df['Release Date'].dt.year)

# # Handle missing values
# cleaning_options = st.sidebar.multiselect(
#     "Handle Missing Values:",
#     options=['Remove rows with missing values', 'Fill numerical with median', 
#              'Fill categorical with mode'],
#     default=['Fill numerical with median', 'Fill categorical with mode']
# )

# # Numerical columns
# num_cols = df.select_dtypes(include=np.number).columns
# for col in num_cols:
#     if 'Fill numerical with median' in cleaning_options:
#         df[col] = df[col].fillna(df[col].median())
#     elif 'Remove rows with missing values' in cleaning_options:
#         df = df.dropna(subset=[col])

# # Categorical columns
# cat_cols = df.select_dtypes(include='object').columns
# for col in cat_cols:
#     if 'Fill categorical with mode' in cleaning_options:
#         df[col] = df[col].fillna(df[col].mode()[0])
#     elif 'Remove rows with missing values' in cleaning_options:
#         df = df.dropna(subset=[col])

# # Convert Yes/No to boolean
# binary_cols = ['USB type C', 'Display Notch', 'HDR Display', 'Dual SIM', 'GPS',
#                'Telephoto Lens', 'Multi-touch', 'Battery Removable', 
#                'Fingerprint', 'Battery Fast Charge']

# for col in binary_cols:
#     if col in df.columns:
#         df[col] = df[col].map({'Yes': True, 'No': False})

# # ------------------------
# # INTERACTIVE FILTERS
# # ------------------------
# st.sidebar.header("🔍 Data Filters")

# # Year filter
# year_min = int(df['Release Year'].min())
# year_max = int(df['Release Year'].max())
# year_range = st.sidebar.slider(
#     "Release Year Range",
#     year_min, year_max, (2010, year_max)
# )

# # Brand filter
# all_brands = df['Brand'].unique()
# selected_brands = st.sidebar.multiselect(
#     "Brands", 
#     all_brands, 
#     default=all_brands[:3] if len(all_brands) > 3 else all_brands
# )

# # Technical specs
# ram_range = st.sidebar.slider(
#     "RAM (GB)", 
#     float(df['RAM Memory (GB)'].min()), 
#     float(df['RAM Memory (GB)'].max()), 
#     (2.0, 8.0)
# )

# battery_range = st.sidebar.slider(
#     "Battery Capacity (mAh)", 
#     float(df['Battery Capacity'].min()), 
#     float(df['Battery Capacity'].max()), 
#     (2000.0, 5000.0)
# )

# # Feature toggles
# st.sidebar.subheader("Feature Filters")
# usb_c = st.sidebar.checkbox("USB Type-C", True)
# fingerprint = st.sidebar.checkbox("Fingerprint Sensor", True)
# fast_charge = st.sidebar.checkbox("Fast Charging", True)

# # Apply filters
# filtered_df = df[
#     (df['Release Year'].between(*year_range)) &
#     (df['Brand'].isin(selected_brands)) &
#     (df['RAM Memory (GB)'].between(*ram_range)) &
#     (df['Battery Capacity'].between(*battery_range)) &
#     (df['USB type C'] == (True if usb_c else df['USB type C'])) &
#     (df['Fingerprint'] == (True if fingerprint else df['Fingerprint'])) &
#     (df['Battery Fast Charge'] == (True if fast_charge else df['Battery Fast Charge']))
# ]

# # ------------------------
# # EDA VISUALIZATIONS
# # ------------------------
# st.title("📱 Smartphone Dataset EDA (2000-2020)")
# st.write(f"**Filtered Devices:** {len(filtered_df)}/{len(df)}")

# # Display data sample
# with st.expander("View Filtered Data"):
#     st.dataframe(filtered_df)

# # Summary stats
# st.subheader("📊 Key Metrics")
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Avg. Battery", f"{filtered_df['Battery Capacity'].mean():.0f} mAh")
# col2.metric("Avg. RAM", f"{filtered_df['RAM Memory (GB)'].mean():.1f} GB")
# col3.metric("Avg. Weight", f"{filtered_df['Weight (g)'].mean():.0f} g")
# col4.metric("Avg. Thickness", f"{filtered_df['Thickness (mm)'].mean():.1f} mm")

# # Visualization tabs
# tab1, tab2, tab3, tab4 = st.tabs([
#     "📈 Distributions", 
#     "📅 Trends Over Time", 
#     "🔍 Feature Analysis", 
#     "🔄 Correlations"
# ])

# with tab1:  # Distributions
#     st.subheader("Feature Distributions")
#     col = st.selectbox("Select feature to visualize", 
#                       ['Battery Capacity', 'RAM Memory (GB)', 
#                        'Thickness (mm)', 'Weight (g)', 
#                        'Screen-to-body Ratio', 'CPU Frequency (GHz)'])
    
#     fig, ax = plt.subplots(figsize=(10, 4))
#     sns.histplot(filtered_df[col], kde=True, ax=ax)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     st.pyplot(fig)
    
#     # Brand distribution
#     st.subheader("Brand Distribution")
#     top_brands = filtered_df['Brand'].value_counts().nlargest(10)
#     fig, ax = plt.subplots(figsize=(10, 4))
#     sns.barplot(x=top_brands.index, y=top_brands.values, ax=ax)
#     plt.xticks(rotation=45)
#     plt.title("Top 10 Brands in Filtered Data")
#     plt.ylabel("Device Count")
#     st.pyplot(fig)

# with tab2:  # Trends over time
#     st.subheader("Technology Trends Over Time")
    
#     # Group by year
#     yearly = filtered_df.groupby('Release Year').agg({
#         'Battery Capacity': 'mean',
#         'RAM Memory (GB)': 'mean',
#         'Screen-to-body Ratio': 'mean',
#         'Thickness (mm)': 'mean'
#     }).reset_index()
    
#     metric = st.selectbox("Select metric to track", 
#                          ['Battery Capacity', 'RAM Memory (GB)',
#                           'Screen-to-body Ratio', 'Thickness (mm)'])
    
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=yearly, x='Release Year', y=metric, marker='o', ax=ax)
#     plt.title(f"Evolution of {metric} Over Time")
#     plt.ylabel(metric)
#     plt.grid(True)
#     st.pyplot(fig)
    
#     # Feature adoption over time
#     st.subheader("Feature Adoption Over Time")
#     feature = st.selectbox("Select feature", 
#                           ['USB type C', 'HDR Display', 'Display Notch', 
#                            'Fingerprint', 'Battery Fast Charge'])
    
#     if feature in binary_cols:
#         adoption = filtered_df.groupby('Release Year')[feature].mean().reset_index()
#         fig, ax = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=adoption, x='Release Year', y=feature, ax=ax)
#         plt.title(f"{feature} Adoption Rate")
#         plt.ylabel("Proportion of Devices")
#         plt.ylim(0, 1)
#         st.pyplot(fig)

# with tab3:  # Feature analysis
#     st.subheader("Feature Relationships")
    
#     x_axis = st.selectbox("X-Axis", num_cols, index=0)
#     y_axis = st.selectbox("Y-Axis", num_cols, index=3)
#     hue = st.selectbox("Color By", ['None'] + list(filtered_df.columns))
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     if hue == 'None':
#         sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
#     else:
#         sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue=hue, ax=ax)
#     plt.title(f"{x_axis} vs {y_axis}")
#     st.pyplot(fig)
    
#     # Feature presence comparison
#     st.subheader("Feature Presence Comparison")
#     features_to_compare = st.multiselect("Select features to compare", 
#                                        binary_cols, 
#                                        default=binary_cols[:3])
    
#     if features_to_compare:
#         feature_presence = filtered_df[features_to_compare].mean().sort_values()
#         fig, ax = plt.subplots(figsize=(10, 4))
#         sns.barplot(x=feature_presence.values, y=feature_presence.index, ax=ax)
#         plt.title("Feature Presence in Filtered Devices")
#         plt.xlabel("Proportion of Devices")
#         st.pyplot(fig)

# with tab4:  # Correlations
#     st.subheader("Feature Correlations")
    
#     # Select numerical columns for correlation
#     num_df = filtered_df.select_dtypes(include=np.number)
#     if not num_df.empty:
#         corr_method = st.radio("Correlation Method", 
#                               ['pearson', 'spearman'], horizontal=True)
        
#         # Compute correlation matrix
#         corr = num_df.corr(method=corr_method)
        
#         # Plot heatmap
#         fig, ax = plt.subplots(figsize=(12, 8))
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
#                     vmin=-1, vmax=1, ax=ax)
#         plt.title("Feature Correlation Matrix")
#         st.pyplot(fig)
#     else:
#         st.warning("No numerical columns available for correlation analysis")

# # ------------------------
# # ADDITIONAL INSIGHTS
# # ------------------------
# st.subheader("🔍 Key Observations")
# st.write("""
# 1. **Battery Evolution**: Battery capacity shows steady increase over time, 
#    with premium devices now exceeding 5000mAh
# 2. **Thickness Trend**: Devices initially got thinner but stabilized around 7-8mm 
#    as battery sizes increased
# 3. **Feature Adoption**: USB-C and fingerprint sensors became standard after 2016
# 4. **RAM Growth**: Average RAM increased from 1-2GB to 6-8GB in flagship devices
# 5. **Screen-to-body**: Ratio increased dramatically with bezel-less designs
# """)

# # Reset filters button
# if st.sidebar.button("Reset All Filters"):
#     st.experimental_rerun()