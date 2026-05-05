import pandas as pd
import numpy as np


class RecursiveForecaster:
    """Recursive multi-step forecaster for models that require lag and exogenous features."""

    def __init__(self, model, lag_list, rolling_list, target_col, exog_cols):
        self.model = model
        self.lag_list = sorted(lag_list)
        self.rolling_list = sorted(rolling_list)
        self.target_col = target_col
        self.exog_cols = exog_cols
        self._feature_order = None

    def _generate_features(self, history_series):
        """Build a single row of autoregressive and rolling features from history."""
        feature_dict = {}

        for lag in self.lag_list:
            feature_dict[f'lag_{lag}'] = [history_series.iloc[-lag]]

        for window in self.rolling_list:
            slice_data = history_series.iloc[-window:]
            feature_dict[f'rolling_mean_{window}'] = [slice_data.mean()]
            feature_dict[f'rolling_std_{window}'] = [slice_data.std()]

        return pd.DataFrame(feature_dict)

    def predict(self, n_periods, history_df, future_exog_df):
        """Forecast multiple periods ahead using recursive feature generation.

        Parameters
        ----------
        n_periods : int
            Number of future periods to predict.
        history_df : pandas.DataFrame
            Historical data containing the target column and date index.
        future_exog_df : pandas.DataFrame
            Exogenous features for the forecast horizon.

        Returns
        -------
        numpy.ndarray
            Array of predicted values.
        """
        history = history_df[self.target_col].copy()
        predictions = []
        current_date = history.index[-1]

        for _ in range(n_periods):
            next_date = current_date + pd.Timedelta(days=1)
            features = self._generate_features(history)
            features.index = [next_date]

            exog_row = future_exog_df.loc[[next_date], self.exog_cols]
            full_row = pd.concat([features, exog_row], axis=1)

            if self._feature_order is None:
                self._feature_order = full_row.columns.tolist()

            full_row = full_row[self._feature_order]
            y_hat = self.model.predict(full_row)[0]

            predictions.append(y_hat)
            history[next_date] = y_hat
            current_date = next_date

        return np.array(predictions)
