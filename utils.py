import pandas as pd
import numpy as np

def calculate_hhi(df: pd.DataFrame, year: int) -> float:
    """Calculates the Herfindahl-Hirschman Index for market concentration."""
    year_df = df[df['Release Year'] == year]
    if year_df.empty: 
        return np.nan
    market_shares = year_df['Brand'].value_counts(normalize=True) * 100
    return (market_shares ** 2).sum()