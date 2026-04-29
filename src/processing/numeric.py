import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Generates lag features for specified numeric columns.
    
    Parameters:
    - columns: list of str
        The numeric columns to create lags for.
    - lags: list of int
        The lag offsets (e.g., [1, 7, 30] for yesterday, last week, last month).
    - drop_na: bool, default=False
        Whether to drop the rows with NaN values created by lagging.
    """
    def __init__(self, column_lag: dict, fill_method: str = 'none'):
        self.column_lag = column_lag
        self.fill_strategy = fill_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        for key, value in self.column_lag.items():
            for lag in value:
                X_out[f"{key}_lag_{lag}"] = X_out[key].shift(lag)
        
        if self.fill_strategy == 'zeros':
            X_out = X_out.fillna(0)
        elif self.fill_strategy == 'ffill':
            X_out = X_out.ffill().fillna(0)
        elif self.fill_strategy == 'bfill':
            X_out = X_out.bfill().fillna(0)
        elif self.fill_strategy == 'none':
            pass  # Keep NaN values for lagged features
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")                            
            
        return X_out

class WindowFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Generates lag features for specified numeric columns.
    
    Parameters:
    - columns: list of str
        The numeric columns to create lags for.
    - lags: list of int
        The lag offsets (e.g., [1, 7, 30] for yesterday, last week, last month).
    - drop_na: bool, default=False
        Whether to drop the rows with NaN values created by lagging.
    """
    def __init__(self, column_lag: dict, fill_method: str = 'none'):
        self.column_lag = column_lag
        self.fill_strategy = fill_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        for key, value in self.column_lag.items():
            for lag in value:
                X_out[f"{key}_rolling_{lag}_mean"] = X_out[key].rolling(window=lag, closed='left', ).mean()
                X_out[f"{key}_rolling_{lag}_std"] = X_out[key].rolling(window=lag, closed='left', ).std()
        
        if self.fill_strategy == 'zeros':
            X_out = X_out.fillna(0)
        elif self.fill_strategy == 'ffill':
            X_out = X_out.ffill().fillna(0)
        elif self.fill_strategy == 'bfill':
            X_out = X_out.bfill().fillna(0)
        elif self.fill_strategy == 'none':
            pass  # Keep NaN values for lagged features
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")                            
            
        return X_out