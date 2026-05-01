import pandas as pd
import numpy as np

class RecursiveForecaster:
    """
    Expert-grade wrapper for recursive multi-step forecasting.
    Encapsulates feature engineering and model inference.
    """
    def __init__(self, model, lag_list, rolling_list, target_col, exog_cols):
        self.model = model
        self.lag_list = sorted(lag_list)
        self.rolling_list = sorted(rolling_list)
        self.target_col = target_col
        self.exog_cols = exog_cols
        # Internal attribute to ensure feature order consistency
        self._feature_order = None 

    def _generate_features(self, history_series):
        """Generates a single row of features from the most recent history."""
        feature_dict = {}
        
        # 1. Lags
        for lag in self.lag_list:
            feature_dict[f'lag_{lag}'] = [history_series.iloc[-lag]]

        # 2. Rolling Stats
        for window in self.rolling_list:
            slice_data = history_series.iloc[-window:]
            feature_dict[f'rolling_mean_{window}'] = [slice_data.mean()]
            feature_dict[f'rolling_std_{window}'] = [slice_data.std()]
            
        return pd.DataFrame(feature_dict)

    def predict(self, n_periods, history_df, future_exog_df):
        """
        n_periods: How many steps to forecast
        history_df: DF containing the recent target values (index must be datetime)
        future_exog_df: DF containing exog features for the forecast horizon
        """
        # Ensure we are working with a Series for the target history
        history = history_df[self.target_col].copy()
        predictions = []
        current_date = history.index[-1]

        for i in range(n_periods):
            # Move to next date
            next_date = current_date + pd.Timedelta(days=1)
            
            # Generate Autoregressive features
            features = self._generate_features(history)
            features.index = [next_date]
            
            # Get Exogenous features for this specific date
            # Using .iloc[i:i+1] assumes future_exog_df is already sliced to n_periods
            exog_row = future_exog_df.loc[[next_date], self.exog_cols]
            
            # Combine
            full_row = pd.concat([features, exog_row], axis=1)
            
            # Feature Alignment (Crucial for XGBoost/RF)
            if self._feature_order is None:
                self._feature_order = full_row.columns.tolist()
            full_row = full_row[self._feature_order]

            # Predict
            y_hat = self.model.predict(full_row)[0]
            
            # Update state for next iteration
            predictions.append(y_hat)
            history[next_date] = y_hat
            current_date = next_date

        return np.array(predictions)