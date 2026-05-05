import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Generate lag features for one or more numeric columns.

    Parameters
    ----------
    column_lag : dict
        Mapping of column names to lists of lag offsets, e.g. {'unit_sales': [1, 7, 30]}.
    fill_method : str, default 'none'
        Strategy for filling NaN values created by lagging.
        Options are 'none', 'zeros', 'ffill', and 'bfill'.
    """

    def __init__(self, column_lag: dict, fill_method: str = 'none'):
        self.column_lag = column_lag
        self.fill_strategy = fill_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        for column, lags in self.column_lag.items():
            for lag in lags:
                X_out[f"{column}_lag_{lag}"] = X_out[column].shift(lag)

        if self.fill_strategy == 'zeros':
            X_out = X_out.fillna(0)
        elif self.fill_strategy == 'ffill':
            X_out = X_out.ffill().fillna(0)
        elif self.fill_strategy == 'bfill':
            X_out = X_out.bfill().fillna(0)
        elif self.fill_strategy == 'none':
            pass
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_strategy}")

        return X_out


class WindowFeatureTransformer(BaseEstimator, TransformerMixin):
    """Generate rolling-window features for one or more numeric columns.

    Parameters
    ----------
    column_lag : dict
        Mapping of column names to rolling window sizes, e.g. {'unit_sales': [7, 14, 21]}.
    fill_method : str, default 'none'
        Strategy for filling NaN values created during rolling-window calculation.
    """

    def __init__(self, column_lag: dict, fill_method: str = 'none'):
        self.column_lag = column_lag
        self.fill_strategy = fill_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        for column, windows in self.column_lag.items():
            for window in windows:
                X_out[f"{column}_rolling_{window}_mean"] = (
                    X_out[column].rolling(window=window, closed='left').mean()
                )
                X_out[f"{column}_rolling_{window}_std"] = (
                    X_out[column].rolling(window=window, closed='left').std()
                )

        if self.fill_strategy == 'zeros':
            X_out = X_out.fillna(0)
        elif self.fill_strategy == 'ffill':
            X_out = X_out.ffill().fillna(0)
        elif self.fill_strategy == 'bfill':
            X_out = X_out.bfill().fillna(0)
        elif self.fill_strategy == 'none':
            pass
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_strategy}")

        return X_out
