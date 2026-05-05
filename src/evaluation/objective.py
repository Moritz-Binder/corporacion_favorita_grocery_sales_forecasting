import os
import pickle
import numpy as np
import pandas as pd
import mlflow
from hyperopt import fmin, tpe, Trials, STATUS_OK
from darts.metrics import mape, rmse
from sklearn.metrics import mean_absolute_error
import json
import copy

import mlflow.pyfunc
# ---------------------------------------------------------
# 1. THE Darts Helper Function for MLflow Logging
# ---------------------------------------------------------
class DartsTranslator(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """
        This method is called by MLflow when the model is loaded.
        'context.artifacts' is a dictionary mapping your labels to the actual file paths.
        """
        from darts.models import RegressionModel # or your specific Darts class
        # We load the model from the path provided by MLflow's artifact store
        self.model = RegressionModel.load(context.artifacts["darts_checkpoint"])

    def predict(self, context, model_input):
        """
        Darts usually predicts 'n' steps ahead. 
        You can decide how your model_input triggers the forecast.
        """
        # Example: If model_input is an integer, predict that many steps
        if isinstance(model_input, int):
            forecast = self.model.predict(n=model_input)
        else:
            # Or if it's a dataframe, predict based on its length
            forecast = self.model.predict(n=len(model_input))
            
        return forecast.values() # Return as a numpy array/list for compatibility


# ---------------------------------------------------------
# 1. THE HYPEROPT OBJECTIVE
# ---------------------------------------------------------
class DartsObjective:
    """
    Universal Evaluator for Darts models with Feature Selection.
    Supports ARIMA, Prophet, and ExponentialSmoothing.
    """
    def __init__(self, series, model_class, horizon, metric, exog=None, start_ratio=0.7):
        self.series = series
        self.model_class = model_class
        self.horizon = horizon
        self.metric = metric
        self.exog = exog # This is the 'Master' exog series
        self.start_ratio = start_ratio

    def __call__(self, params):
        # Expanded integer list to include seasonal components
        int_params = [
            'p', 'd', 'q', 'P', 'D', 'Q', 's'
        ]
        # 1. Feature Selection Logic
        # Extract the list of chosen features from the space
        # If 'selected_features' isn't in params, it defaults to all features
        selected_features = []
        if self.exog is not None:
            # 2. Check if this specific model actually tuned 'selected_features'
            if 'selected_features' in params:
                selected_features = [f for f in params['selected_features'] if f is not None]
                params.pop('selected_features', None)

                if not selected_features:
                    current_exog = None
                else:
                    current_exog = self.exog[selected_features]
            else:
                # Fallback: The model is given exog data, but didn't tune feature selection.
                current_exog = self.exog
        else:
            current_exog = None
        
        # 3. Parameter Cleaning & Tuple Reconstruction
        clean_params = {}
        for k, v in params.items():
            # Handle the specific seasonal_order tuple for SARIMAX
            if k == 'seasonal_order' and isinstance(v, (list, tuple)):
                clean_params[k] = tuple(int(x) if i < 3 else x for i, x in enumerate(v))
            # Standard integer casting
            elif k in int_params:
                clean_params[k] = int(v)
            else:
                clean_params[k] = v
        # 4. Instantiate Model
        model = self.model_class(**clean_params)
        
        # 5. Determine Covariate Type
        # Some models use 'future_covariates' (Prophet/ARIMA), 
        # some use 'past_covariates', and some (ExpSmoothing) use none.
        kwargs = {}
        if current_exog is not None:
            if model.supports_future_covariates:
                kwargs['future_covariates'] = current_exog
            elif model.supports_past_covariates:
                kwargs['past_covariates'] = current_exog
        else:
            # If no exogenous variables are selected, ensure we don't pass any covariates
            kwargs = {}

        # 6. Backtesting
        try:
            forecasts = model.historical_forecasts(
                self.series,
                start=self.start_ratio,
                forecast_horizon=self.horizon,
                retrain=True,
                last_points_only=True,
                **kwargs
            )
            # Hyperopt minimizes this 'loss' value
            loss = self.metric(self.series, forecasts)
        except Exception as e:
            print(f"Trial failed for {self.model_class.__name__}: {e}")
            loss = 1e9 # High penalty for non-convergence

        # We return the selected features in the result so the Optimizer can log them
        if self.exog is not None:
            return {
                'loss': loss, 
                'status': STATUS_OK, 
                'params': clean_params, 
                'selected_features': selected_features
            }
        else:
            return {
                'loss': loss, 
                'status': STATUS_OK, 
                'params': clean_params
            }


# ---------------------------------------------------------
# 2. THE ORCHESTRATOR
# ---------------------------------------------------------
class TimeSeriesOptimizer:
    """
    Manages batching, MLflow logging, and experiment state.
    """
    def __init__(self, experiment_name: str, trials_dir: str = "../trials"):
        self.experiment_name = experiment_name
        self.trials_dir = trials_dir
        os.makedirs(self.trials_dir, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    def optimize_and_log(self, model_name, model_class, space, series, horizon, metric, exog=None, max_evals=50):
        trials_path = os.path.join(self.trials_dir, f"{model_name}_{str(horizon)}_trials.pkl")
        
        # 1. Load existing trials for batching
        if os.path.exists(trials_path):
            with open(trials_path, "rb") as f:
                trials = pickle.load(f)
            print(f"Resuming {model_name}: Found {len(trials.trials)} previous trials.")
        else:
            trials = Trials()

        # Check if we already hit our evaluation target
        current_evals = len(trials.trials)
        if current_evals >= max_evals:
            print(f"Optimization for {model_name} already completed {max_evals} evals. Skipping search.")
        else:
            # 2. Run Hyperopt
            print(f"Running Hyperopt for {model_name} up to {max_evals} evals...")
            objective = DartsObjective(series, model_class, horizon, metric, exog)
            
            _ = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                show_progressbar=True
            )
            
            # 3. Save the updated trials object immediately
            with open(trials_path, "wb") as f:
                pickle.dump(trials, f)

        # 4. Log ONLY the Champion to MLflow (if target evals are met)
        if len(trials.trials) >= max_evals:
            self._log_champion(model_name, model_class, trials, series, exog, trials_path)

    def _log_champion(self, model_name, model_class, trials, series, exog, trials_path):
        # Extract the best parameters and loss from the trials object
        best_trial = trials.best_trial
        best_params = copy.deepcopy(best_trial['result']['params'])
        best_loss = best_trial['result']['loss']

        # Expanded integer list to include seasonal components
        int_params = [
            'p', 'd', 'q', 'P', 'D', 'Q', 's'
        ]
        if 'selected_features' in best_params:
            selected_features = [f for f in best_params['selected_features'] if f is not None]
        else:
            selected_features = None

        with mlflow.start_run(run_name=f"Champion_{model_name}") as run:
            print(f"Logging Champion {model_name} to MLflow...")

            # Log metrics, params, and artifacts
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_loss", best_loss)
            
            if exog is not None:
                # 2. Check if this specific model actually tuned 'selected_features'
                if 'selected_features' in best_params:
                    selected_features = [f for f in best_params['selected_features'] if f is not None]
                    best_params.pop('selected_features', None)

                    if not selected_features: # More Pythonic check for an empty list
                        current_exog = None
                    else:
                        current_exog = exog[selected_features]
                else:
                    # Fallback: The model is given exog data, but didn't tune feature selection.
                    # We default to passing all available exogenous features.
                    current_exog = exog
            else:
                current_exog = None

            # 3. Parameter Cleaning & Tuple Reconstruction
            clean_params = {}
            for k, v in best_params.items():
                # Handle the specific seasonal_order tuple for SARIMAX
                if k == 'seasonal_order' and isinstance(v, (list, tuple)):
                    clean_params[k] = tuple(int(x) if i < 3 else x for i, x in enumerate(v))
                # Standard integer casting
                elif k in int_params:
                    clean_params[k] = int(v)
                else:
                    clean_params[k] = v
            # Re-train the final model on the ENTIRE dataset for production deployment
            final_model = model_class(**clean_params)
            
            kwargs = {}
            if current_exog is not None:
                if final_model.supports_future_covariates:
                    kwargs['future_covariates'] = current_exog
                elif final_model.supports_past_covariates:
                    kwargs['past_covariates'] = current_exog
            else:
                # If no exogenous variables are selected, ensure we don't pass any covariates
                kwargs = {}
                
            final_model.fit(series, **kwargs)
            
            # Attach the trials history as a file so you don't lose the R&D data
            mlflow.log_artifact(trials_path, artifact_path="hyperopt_history")

            features = {"features": selected_features if selected_features is not None else []}
            with open("features.json", "w") as f:
                json.dump(features, f)

            mlflow.log_artifact("features.json")
            
            # Log the final model
            # Note: Depending on your MLflow version, you might need to use mlflow.pyfunc 
            # or save the model locally via final_model.save() and log it as an artifact.
            try:
                model_path = f"{model_name}_best.pkl"
                final_model.save(model_path)
                artifacts = {
                    "darts_checkpoint": model_path
                }
                mlflow.pyfunc.log_model(
                                name="model",
                                python_model=DartsTranslator(),
                                artifacts=artifacts,
                                # Optional: Add the environment to ensure the right version of Darts is installed later
                                pip_requirements=["darts==0.43.0"] 
                            )
                os.remove(model_path)
            except AttributeError:
                # Fallback if darts flavor is missing in your mlflow version
                model_path = f"{model_name}_best.pkl"
                final_model.save(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
                os.remove(model_path)

# ---------------------------------------------------------
# 3. Forecast Function
# ---------------------------------------------------------
def forecasting_pipeline(model, y_train, x_dates, num_days, test, test_dates, lag_list, rolling_list, selected_features, target):
    """Create recursive forecasts for a model that requires lag and rolling features."""
    import pandas as pd
    import numpy as np

    history = y_train.copy().set_index(x_dates)
    xogen = test.copy().set_index(test_dates)
    xogen = xogen.loc[:, ~xogen.columns.str.contains('lag|rolling', case=False)]
    last_date = history.index[-1]
    future = []

    for i in range(num_days):
        next_date = last_date + pd.Timedelta(days=1)
        feature_dict = {}

        for lag in lag_list:
            feature_dict[f'{target}_lag_{lag}'] = [history[target].iloc[-lag]]

        for window in rolling_list:
            feature_dict[f'{target}_rolling_{window}_mean'] = [history[target].iloc[-window:].mean()]
            feature_dict[f'{target}_rolling_{window}_std'] = [history[target].iloc[-window:].std()]

        row = pd.DataFrame(data=feature_dict, index=[next_date])
        rows = row.merge(xogen, how='left', left_index=True, right_index=True)
        rows = rows[selected_features]

        if rows.isnull().any().any():
            rows = rows.fillna(0)

        y_hat = model.predict(rows)[0]
        future.append(y_hat)

        history = pd.concat([history, pd.DataFrame(data={'unit_sales': y_hat}, index=[next_date])])
        last_date = next_date

    return future

import os
import pickle
import numpy as np
import pandas as pd
import mlflow
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 4. Machine Learning Recursive Objective for Cross-Validation
# ---------------------------------------------------------
class MLRecursiveObjective:
    def __init__(self, df, target_col, date_col, model_class, metric, 
                 lag_list, rolling_list, start_ratio=0.7, step_size=7):
        self.df = df.sort_values(date_col).reset_index(drop=True)
        self.target_col = target_col
        self.date_col = date_col
        self.model_class = model_class
        self.metric = metric
        self.lag_list = lag_list
        self.rolling_list = rolling_list
        self.start_ratio = start_ratio
        self.step_size = step_size
        self.exog_cols = [c for c in df.columns if c not in [target_col, date_col]]

    def __call__(self, params):
        # 1. Parameter Clean-up (Hyperopt specific)
        int_params = [
            'n_estimators', 'max_depth', 'min_samples_split', 'max_iter', 'min_samples_leaf'
        ]
        # 1. Feature Selection Logic
        # Extract the list of chosen features from the space
            # 2. Check if this specific model actually tuned 'selected_features'
        if 'selected_features' in params:
            selected_features = [f for f in params['selected_features'] if f is not None]
            total_columns = [self.date_col, self.target_col] + selected_features
            params.pop('selected_features', None)

            if not selected_features: # More Pythonic check for an empty list
                current_df = self.df[[self.target_col, self.date_col]]
            else:
                current_df = self.df[total_columns]
        else:
            # Fallback: The model is given exog data, but didn't tune feature selection.
            # We default to passing all available exogenous features.
            current_df = self.df
        
        # 3. Parameter Cleaning & Tuple Reconstruction
        clean_params = {}
        for k, v in params.items():
            # Standard integer casting
            if k in int_params:
                clean_params[k] = int(v)
            else:
                clean_params[k] = v

        model = self.model_class(**clean_params)
        
        # 2. Rolling CV
        total_len = len(current_df)
        start_idx = int(total_len * self.start_ratio)
        fold_losses = []
        
        for current_split in range(start_idx, total_len - self.step_size, self.step_size):
            train_df = current_df.iloc[:current_split]
            test_df = current_df.iloc[current_split : current_split + self.step_size]
            
            # Static Fit
            X_train = train_df.drop(columns=[self.target_col, self.date_col])
            y_train = train_df[self.target_col]
            model.fit(X_train, y_train)
            
            # Recursive Predict using the dynamic lists
            preds = forecasting_pipeline(
                model=model,
                y_train=train_df[[self.target_col]],
                x_dates=train_df[self.date_col],
                num_days=self.step_size,
                test=test_df.drop(columns=[self.target_col]),
                test_dates=test_df[self.date_col],
                lag_list=self.lag_list,
                rolling_list=self.rolling_list,
                selected_features=selected_features,
                target=self.target_col
            )
            
            fold_losses.append(self.metric(test_df[self.target_col], preds))

        return {'loss': np.mean(fold_losses), 'status': STATUS_OK, 'params': params}

# ---------------------------------------------------------
# 5. Machine Learning Optimizer for Hyperparameter Tuning with MLflow Logging
# ---------------------------------------------------------

class MLOptimizer:
    def __init__(self, experiment_name: str, trials_dir: str = "../trials"):
        self.experiment_name = experiment_name
        self.trials_dir = trials_dir
        os.makedirs(self.trials_dir, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    def optimize_ml_model(self, model_name, model_class, space, df, target_col, date_col,lag_list,rolling_list, metric, start_ratio=0.7, step_size=7, max_evals=30):
        trials_path = os.path.join(self.trials_dir, f"{model_name}_{str(step_size)}_trials.pkl")
        
        if os.path.exists(trials_path):
            with open(trials_path, "rb") as f:
                trials = pickle.load(f)
        else:
            trials = Trials()

        if len(trials.trials) < max_evals:
            objective = MLRecursiveObjective(
                df=df, 
                target_col=target_col, 
                date_col=date_col, 
                model_class=model_class, 
                metric=metric,
                lag_list=lag_list,
                rolling_list=rolling_list,
                start_ratio=start_ratio,
                step_size=step_size
            )
            
            fmin(fn=objective, space=space, algo=tpe.suggest, 
                 max_evals=max_evals, trials=trials)
            
            with open(trials_path, "wb") as f:
                pickle.dump(trials, f)

        self._log_ml_champion(model_name, model_class, trials, df, target_col, date_col, trials_path)

    def _log_ml_champion(self, model_name, model_class, trials, df, target_col, date_col, trials_path):
        # Extract the best parameters and loss from the trials object
        best_trial = trials.best_trial
        best_params = copy.deepcopy(best_trial['result']['params'])
        best_loss = best_trial['result']['loss']
        int_params = [
            'n_estimators', 'max_depth', 'min_samples_split', 'max_iter', 'min_samples_leaf'
        ]
        if 'selected_features' in best_params:
            selected_features = [f for f in best_params['selected_features'] if f is not None]
        else:
            selected_features = None
        with mlflow.start_run(run_name=f"Champion_{model_name}") as run:
            print(f"Logging Champion {model_name} to MLflow...")

            # Log metrics, params, and artifacts
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_loss", best_loss)
            
            if 'selected_features' in best_params:
                selected_features = [f for f in best_params['selected_features'] if f is not None]
                total_columns = selected_features + [self.target_col, self.date_col]
                best_params.pop('selected_features', None)

                if not selected_features: # More Pythonic check for an empty list
                    current_df = df[[self.target_col, self.date_col]]
                else:
                    current_df = df[total_columns]
            else:
                # Fallback: The model is given exog data, but didn't tune feature selection.
                # We default to passing all available exogenous features.
                current_df = df

            # 3. Parameter Cleaning & Tuple Reconstruction
            clean_params = {}
            for k, v in best_params.items():
                # Standard integer casting
                if k in int_params:
                    clean_params[k] = int(v)
                else:
                    clean_params[k] = v
            
            # Train final model on full dataset
            X = current_df.drop(columns=[target_col, date_col])
            y = current_df[target_col]
            
            final_model = model_class(**clean_params)
            final_model.fit(X, y)
            
            # Save trials history
            mlflow.log_artifact(trials_path, artifact_path="hyperopt_history")

            features = {"features": selected_features if selected_features is not None else []}
            with open("features.json", "w") as f:
                json.dump(features, f)

            mlflow.log_artifact("features.json")
            
            # Log Model based on library
            if "XGB" in str(model_class):
                mlflow.xgboost.log_model(final_model, "model")
            else:
                mlflow.sklearn.log_model(final_model, "model")
            
            print(f"Successfully logged {model_name} with min loss: {best_loss:.4f}")