import streamlit as st
import pandas as pd
import plotly.express as px

def render(analysis_df: pd.DataFrame):
    st.header("At a Glance: Key Stories from the Data")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Growth of Screen Real Estate")
        avg_screen_area = analysis_df.groupby('Release Year')['Screen Area (sq inches)'].mean().reset_index()
        fig_area = px.area(avg_screen_area, x='Release Year', y='Screen Area (sq inches)', markers=True,
                           title='Average Smartphone Screen Area Over Time')
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