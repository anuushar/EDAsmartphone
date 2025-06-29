import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Callable

def render(analysis_df: pd.DataFrame, hhi_calculator: Callable):
    st.header("How Competitive is the Smartphone Market?")
    st.markdown("Here we analyze the overall market structure and the dominance of leading brands.")
    
    st.subheader("Measuring Market Concentration (HHI)")
    hhi_data = [hhi_calculator(analysis_df, year) for year in analysis_df['Release Year'].unique()]
    hhi_df = pd.DataFrame({'Year': analysis_df['Release Year'].unique(), 'HHI': hhi_data}).dropna().sort_values('Year')
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