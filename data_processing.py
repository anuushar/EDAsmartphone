
import streamlit as st
import pandas as pd
import numpy as np
from typing import Union

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

    numeric_cols = [
        'Rear Cameras', 'Battery Capacity', 'Thickness (mm)', 'Weight (grams)',
        'Screen Diagonal (inches)', 'Rear Camera MP', 'RAM (GB)',
        'Pixels Per Inch', 'CPU Frequency (GHz)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            col_median = df[col].median()
            
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