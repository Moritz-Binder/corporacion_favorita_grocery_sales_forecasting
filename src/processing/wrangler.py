import pandas as pd

class TimeSeriesWrangler:
    """
    Handles structural data changes like resampling and gap filling.

    Parameters:
    - date_col: str
        The name of the date column in the DataFrame.
    - fill_col: str
        The name of the column to fill when resampling.
    - freq: str, optional (default='D')
        The frequency to resample the data to (e.g., 'D' for daily, 'H' for hourly).
    - fill_method: str, optional (default='zero')
        The method to fill missing values after resampling. Options include:
        'zero' (fill with zeros), 'ffill' (forward fill), 'bfill' (backward fill).
    """
    def __init__(self, date_col: str, fill_col: str, freq: str = 'D', fill_method: str = 'zero'):
        self.date_col = date_col
        self.freq = freq
        self.fill_col = fill_col
        self.fill_method = fill_method

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The main entry point for cleaning. 
        Ensures the dataframe is structurally sound.
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # 1. Handle Duplicates
        df = df.drop_duplicates(subset=[self.date_col])
        
        # 2. Fill Gaps (Resampling)
        df = df.set_index(self.date_col)
        
        resampler = df.resample(self.freq)
        
        if self.fill_strategy == 'zeros':
            df = resampler[self.fill_col].asfreq().fillna(0)
        elif self.fill_strategy == 'ffill':
            df = resampler[self.fill_col].ffill().fillna(0)
        elif self.fill_strategy == 'bfill':
            df = resampler[self.fill_col].bfill().fillna(0)
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")
        
        # 3. Reset index to keep it compatible with downstream tasks
        return df.reset_index()