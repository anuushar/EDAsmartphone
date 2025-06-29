import streamlit as st
import pandas as pd
import plotly.express as px

def render(analysis_df: pd.DataFrame):
    st.header("What is the current state of manufacturing in countries?")
    
    st.subheader("Global Concentration of Smartphone Brands")
    st.markdown("This world map is colored based on the number of smartphone models originating from each country in the selected time period.")
    country_counts = analysis_df['Brand Country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_iso_map = px.data.gapminder()[['country', 'iso_alpha']].drop_duplicates()
    map_data = pd.merge(country_counts, country_iso_map, on='country', how='left')
    fig_map = px.choropleth(map_data, locations="iso_alpha", color="count", hover_name="country",
                           color_continuous_scale=px.colors.sequential.Plasma,
                           title="Number of Smartphone Models by Brand Country of Origin")
    st.plotly_chart(fig_map, use_container_width=True)