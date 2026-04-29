import os
import pickle
import numpy as np
import mlflow
from hyperopt import fmin, tpe, Trials, STATUS_OK
from darts.metrics import mape, rmse

# ---------------------------------------------------------
# 1. THE HYPEROPT OBJECTIVE
# ---------------------------------------------------------
class DartsObjective:
    """
    Evaluates a Darts time-series model using historical backtesting.
    No MLflow logging happens here to keep the tracking server clean.
    """
    def __init__(self, series, model_class, horizon, metric, exog=None, start_ratio=0.7):
        self.series = series
        self.model_class = model_class
        self.horizon = horizon
        self.metric = metric
        self.exog = exog
        self.start_ratio = start_ratio
        
        # Parameters that must strictly be integers for Darts models
        self.int_params = ['p', 'd', 'q', 'P', 'D', 'Q', 'n_changepoints', 'seasonal_periods']

    def __call__(self, params):
        # 1. Cast parameters correctly (Hyperopt often returns floats)
        clean_params = {}
        for k, v in params.items():
            clean_params[k] = int(v) if k in self.int_params else v

        # 2. Instantiate Model
        model = self.model_class(**clean_params)
        
        # 3. Handle models that take covariates vs those that don't
        # Darts will throw an error if you pass future_covariates to ExponentialSmoothing
        kwargs = {}
        if self.exog is not None and model.supports_future_covariates:
            kwargs['future_covariates'] = self.exog

        # 4. Rigorous Backtesting (Prevents Data Leakage)
        try:
            forecasts = model.historical_forecasts(
                self.series,
                start=self.start_ratio,
                forecast_horizon=self.horizon,
                retrain=True,
                last_points_only=True,
                **kwargs
            )
            # Calculate error
            loss = self.metric(self.series, forecasts)
        except Exception as e:
            # If the model fails to converge or errors out, penalize it heavily
            print(f"Trial failed with params {clean_params}: {e}")
            loss = np.inf

        return {'loss': loss, 'status': STATUS_OK, 'params': clean_params}


# ---------------------------------------------------------
# 2. THE ORCHESTRATOR
# ---------------------------------------------------------
class TimeSeriesOptimizer:
    """
    Manages batching, MLflow logging, and experiment state.
    """
    def __init__(self, experiment_name: str, trials_dir: str = "./trials"):
        self.experiment_name = experiment_name
        self.trials_dir = trials_dir
        os.makedirs(self.trials_dir, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    def optimize_and_log(self, model_name, model_class, space, series, horizon, metric, exog=None, max_evals=50):
        trials_path = os.path.join(self.trials_dir, f"{model_name}_trials.pkl")
        
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
        best_params = best_trial['result']['params']
        best_loss = best_trial['result']['loss']
        
        with mlflow.start_run(run_name=f"Champion_{model_name}") as run:
            print(f"Logging Champion {model_name} to MLflow...")
            
            # Re-train the final model on the ENTIRE dataset for production deployment
            final_model = model_class(**best_params)
            
            kwargs = {}
            if exog is not None and final_model.supports_future_covariates:
                kwargs['future_covariates'] = exog
                
            final_model.fit(series, **kwargs)
            
            # Log metrics, params, and artifacts
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_loss", best_loss)
            
            # Attach the trials history as a file so you don't lose the R&D data
            mlflow.log_artifact(trials_path, artifact_path="hyperopt_history")
            
            # Log the final model
            # Note: Depending on your MLflow version, you might need to use mlflow.pyfunc 
            # or save the model locally via final_model.save() and log it as an artifact.
            try:
                mlflow.darts.log_model(final_model, artifact_path="model")
            except AttributeError:
                # Fallback if darts flavor is missing in your mlflow version
                model_path = f"{model_name}_best.pkl"
                final_model.save(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
                os.remove(model_path)