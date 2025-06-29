import streamlit as st
import pandas as pd
import plotly.express as px

def render(analysis_df: pd.DataFrame, highlighted_brands: list):
    st.header("How are Smartphone Specs and Features Evolving?")
    
    st.subheader("Are All Phones Becoming the Same? (Hardware Convergence)")
    commoditization_metric = st.selectbox("Select a Specification to Analyze:", 
                                           options=['Battery Capacity', 'Screen Diagonal (inches)', 'Pixels Per Inch', 'Weight (grams)'])
    pivot_df = analysis_df[analysis_df['Brand'].isin(highlighted_brands)].pivot_table(index='Release Year', columns='Brand', 
                                                                                     values=commoditization_metric, aggfunc='median')
    fig_commoditization = px.line(pivot_df, title=f'Median {commoditization_metric} for Highlighted Brands', 
                                   labels={'value': commoditization_metric, 'Release Year': 'Year'}, markers=True)
    st.plotly_chart(fig_commoditization, use_container_width=True)