import pandas as pd
import numpy as np
import holidays
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A professional-grade transformer for flexible date feature engineering.

    Parameters:
    - column_name: str
        The name of the date column to transform.
    - features: list of str, optional (default=['year', 'month', 'day_of_week', 'is_weekend'])
        A list of date features to extract. Supported features include:
        'year', 'month', 'day_of_week', 'is_weekend', 'is_payday'.
    - payday_val: int, optional (default=15)
        The day of the month considered as payday for the 'is_payday' feature.
    - country: str, optional (default='EC')
        The country code for holiday calculations (e.g., 'EC', 'US', 'UK', 'DE').
    - subdiv: str, optional (default='P')
        The subdivision code for holiday calculations (e.g., P: Pichincha (Quito), G: Guayas (Guayaquil), A: Azuay (Cuenca), M: Manabí (Manta)).
    """
    def __init__(self, 
                 column_name: str, 
                 features: list = ['year', 'month', 'day_of_week', 'is_weekend', 'is_holiday'],
                 payday_val: int = 15,
                 country: str = 'EC',
                 subdiv: str = None,
                 drop_date_col: bool = True):
        self.column_name = column_name
        self.features = features
        self.payday_val = payday_val
        self.country = country
        self.subdiv = subdiv
        self.holiday_lookup = None
        self.drop_date_col = drop_date_col

    def fit(self, X, y=None):
        # We pre-calculate holidays during fit to ensure consistency 
        # and speed up the transform step.
        date_series = pd.to_datetime(X[self.column_name])
        start_year = date_series.dt.year.min()
        end_year = date_series.dt.year.max()
        
        # Populate holiday dictionary for the range of years in the data
        self.holiday_lookup = holidays.CountryHoliday(
            self.country,
            subdiv=self.subdiv, 
            years=range(start_year, end_year + 1)
        )
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.column_name not in X_out.columns:
            raise ValueError(f"Column '{self.column_name}' not found.")
        
        date_series = pd.to_datetime(X_out[self.column_name])

        feature_map = {
            'year': lambda ds: ds.dt.year,
            'month': lambda ds: ds.dt.month,
            'day_of_week': lambda ds: ds.dt.dayofweek,
            'is_weekend': lambda ds: ds.dt.dayofweek.isin([5, 6]).astype(int),
            'is_holiday': lambda ds: ds.apply(lambda x: 1 if x in self.holiday_lookup else 0),
            'is_payday': lambda ds: ((ds.dt.day == self.payday_val) | 
                                     (ds.dt.day > ds.dt.day.shift(-1))).astype(int)
        }

        for feature in self.features:
            if feature in feature_map:
                X_out[f"{self.column_name}_{feature}"] = feature_map[feature](date_series)
            else:
                print(f"Warning: Feature '{feature}' not recognized.")

        if self.drop_date_col: 
                X_out = X_out.drop(columns=[self.column_name])
        else:
                X_out = X_out
        return X_out