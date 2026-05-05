import pandas as pd


class TimeSeriesWrangler:
    """Handle time series resampling and gap filling for daily sales data.

    Parameters
    ----------
    date_col : str
        The name of the date column in the input DataFrame.
    fill_col : str
        The column whose missing values should be filled after resampling.
    freq : str, default 'D'
        The frequency used to reindex the time series.
    fill_method : str, default 'zeros'
        The strategy used to fill missing values. Supported values are 'zeros', 'ffill', and 'bfill'.
    """

    def __init__(self, date_col: str, fill_col: str, freq: str = 'D', fill_method: str = 'zeros'):
        self.date_col = date_col
        self.freq = freq
        self.fill_col = fill_col
        self.fill_method = fill_method

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input DataFrame, resample to the target frequency, and fill missing values."""
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.drop_duplicates(subset=[self.date_col])
        df = df.set_index(self.date_col)

        date_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.freq)
        resampler = df.reindex(date_index)

        if self.fill_method == 'zeros':
            df_clean = resampler[self.fill_col].fillna(0)
        elif self.fill_method == 'ffill':
            df_clean = resampler[self.fill_col].ffill().fillna(0)
        elif self.fill_method == 'bfill':
            df_clean = resampler[self.fill_col].bfill().fillna(0)
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")

        df_clean.index.name = self.date_col
        return df_clean.reset_index()
