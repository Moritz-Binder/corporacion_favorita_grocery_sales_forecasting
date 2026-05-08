# Improvement Backlog

## Implementation Order
To tackle these improvements systematically, follow this prioritized sequence:

1. **Start with Project Setup & Documentation (items 1-3)**: Establish a solid foundation with environment manifests, package structure, and clear README documentation.
2. **Data Pipeline & Validation (items 4-6)**: Build reliable data processing scripts and validation checks to ensure data quality.
3. **Code Quality & Testing (items 13-15)**: Add tests, linting, and type hints to make the codebase maintainable and professional.
4. **Modeling Improvements (items 16-18)**: Enhance model validation, metrics, and explainability for better forecasting accuracy.
5. **MLflow & Model Registry (items 7-9)**: Improve model management and deployment reliability.
6. **Streamlit App Enhancements (items 10-12)**: Add user-facing features like model comparison and error handling.
7. **Notebooks & Reproducibility (items 19-21)**: Polish notebooks with summaries and execution guides.
8. **Portfolio & Recruiter-Friendly Storytelling (items 22-30)**: Add executive summaries, screenshots, and skills mapping for maximum impact.

This order ensures foundational stability before adding advanced features and presentation polish.

## Project Setup & Documentation
1. Add a reproducible environment manifest (`environment.yml` or `pyproject.toml`) and pin dependency versions in `requirements.txt`.
2. Standardize the repository package layout with a proper `src` package and remove the current `sys.path.append` workaround in `streamlit_app/main.py`.
3. Improve the README to document the notebook execution order, MLflow registry tags, and the correct model naming/tagging conventions.

## Data Pipeline & Validation
4. Build a dedicated preprocessing script or module to generate `data/processed/timeseries_ABT.csv` from raw inputs, instead of relying only on notebooks.
5. Add data validation checks for raw inputs, missing dates, and holiday/oil feature alignment.
6. Add a small data catalogue or schema file for `data/raw` and `data/processed` to make dataset provenance clear.

## MLflow & Model Registry
7. Remove the hard-coded MLflow URI and replace it with configuration via environment variables or a settings file.
8. Improve registry logic to verify model artifacts and stages when loading a model, with clear fallback behavior if the registry is unavailable.
9. Add automated model promotion steps and explicit tagging for `forecast=weekly/monthly/quarterly` and `active=true`.

## Streamlit App Enhancements
10. Add a model comparison panel so users can compare weekly, monthly, and quarterly forecasts side-by-side.
11. Improve forecasting error handling and warnings in the app when the current model is missing or MLflow is unreachable.
12. Add dashboard features for scenario analysis, such as holiday/promotion impact, and optional inventory / demand planning signals.

## Code Quality & Testing
13. Add unit tests for `src/processing` transformers and the forecasting wrapper in `src/prod/ml_forecast.py`.
14. Add linting/formatting and static analysis (e.g., `ruff`/`flake8`, `black`, `mypy`) to enforce consistent style.
15. Add more complete type hints and docstrings across modules so the pipeline is easier to maintain.

## Modeling Improvements
16. Use time-series-aware validation such as rolling `TimeSeriesSplit` rather than a single train/test split.
17. Track multiple evaluation metrics (`MAE`, `RMSE`, `MAPE`, `coverage`) and log them consistently in MLflow.
18. Add feature importance or explainability analysis for tree-based models and a baseline forecast comparison.

## Notebooks & Reproducibility
19. Add result summary sections in notebooks describing the experiment outcome and next steps.
20. Consider converting the most important notebook workflows into reproducible Python scripts or modular pipeline code.
21. Add an index or notebook README that clearly explains the recommended execution flow.

## Portfolio & Recruiter-Friendly Storytelling
22. Add an executive summary in `readme.md` that highlights business impact, project scope, and why the solution matters to operations and forecasting stakeholders.
23. Include a concise "What I built" section with the end-to-end stack, your role, and the main technical highlights (data pipeline, MLOps, deployment, app UI).
24. Add a "Key results" / "Impact" section describing model accuracy, expected business benefits, and what the app enables for decision makers.
25. Add screenshots or annotated GIFs of the Streamlit app and key notebook outputs so recruiters can quickly see the product.
26. Add a "Skills demonstrated" section that maps the project to relevant senior data science and full-stack competencies: time-series modeling, model deployment, MLflow, pipeline automation, dashboarding, and cross-functional delivery.
27. Add a simple architecture diagram or flow chart showing raw data → feature engineering → modeling → MLflow → Streamlit app.
28. Add a "Next steps" / "Roadmap" subsection to show continuous improvement mindset, including model monitoring, deployment automation, and API-driven forecasting.
29. Add a "How to evaluate" or "How to run this project" checklist for reviewers to assess the repo quickly.
30. Add a section about collaborative delivery, such as working with stakeholders, interpreting business requirements, and translating forecasts into operational actions.

---------------------------------------------------------

1. Add a reproducible environment manifest (`environment.yml` or `pyproject.toml`) and pin dependency versions in `requirements.txt`.
2. Standardize the repository package layout with a proper `src` package and remove the current `sys.path.append` workaround in `streamlit_app/main.py`.
3. Improve the README to document the notebook execution order, MLflow registry tags, and the correct model naming/tagging conventions.
3. Connect to Kaggle Dataset and load the entire dataset.

## Data Pipeline & Validation
4. Build a dedicated preprocessing script or module to generate `data/processed/timeseries_ABT.csv` from raw inputs, instead of relying only on notebooks.
5. Add data validation checks for raw inputs, missing dates, and holiday/oil feature alignment.
6. Add a small data catalogue or schema file for `data/raw` and `data/processed` to make dataset provenance clear.

## MLflow & Model Registry
7. Remove the hard-coded MLflow URI and replace it with configuration via environment variables or a settings file.
8. Improve registry logic to verify model artifacts and stages when loading a model, with clear fallback behavior if the registry is unavailable.
9. Add automated model promotion steps and explicit tagging for `forecast=weekly/monthly/quarterly` and `active=true`.

## Streamlit App Enhancements
10. Add a model comparison panel so users can compare weekly, monthly, and quarterly forecasts side-by-side.
11. Improve forecasting error handling and warnings in the app when the current model is missing or MLflow is unreachable.
12. Add dashboard features for scenario analysis, such as holiday/promotion impact, and optional inventory / demand planning signals.

## Code Quality & Testing
13. Add unit tests for `src/processing` transformers and the forecasting wrapper in `src/prod/ml_forecast.py`.
14. Add linting/formatting and static analysis (e.g., `ruff`/`flake8`, `black`, `mypy`) to enforce consistent style.
15. Add more complete type hints and docstrings across modules so the pipeline is easier to maintain.

## Modeling Improvements
16. Use time-series-aware validation such as rolling `TimeSeriesSplit` rather than a single train/test split.
17. Track multiple evaluation metrics (`MAE`, `RMSE`, `MAPE`, `coverage`) and log them consistently in MLflow.
18. Add feature importance or explainability analysis for tree-based models and a baseline forecast comparison.

## Notebooks & Reproducibility
19. Add result summary sections in notebooks describing the experiment outcome and next steps.
20. Consider converting the most important notebook workflows into reproducible Python scripts or modular pipeline code.
21. Add an index or notebook README that clearly explains the recommended execution flow.

## Portfolio & Recruiter-Friendly Storytelling
22. Add an executive summary in `readme.md` that highlights business impact, project scope, and why the solution matters to operations and forecasting stakeholders.
23. Include a concise “What I built” section with the end-to-end stack, your role, and the main technical highlights (data pipeline, MLOps, deployment, app UI).
24. Add a “Key results” / “Impact” section describing model accuracy, expected business benefits, and what the app enables for decision makers.
25. Add screenshots or annotated GIFs of the Streamlit app and key notebook outputs so recruiters can quickly see the product.
26. Add a “Skills demonstrated” section that maps the project to relevant senior data science and full-stack competencies: time-series modeling, model deployment, MLflow, pipeline automation, dashboarding, and cross-functional delivery.
27. Add a simple architecture diagram or flow chart showing raw data → feature engineering → modeling → MLflow → Streamlit app.
28. Add a “Next steps” / “Roadmap” subsection to show continuous improvement mindset, including model monitoring, deployment automation, and API-driven forecasting.
29. Add a “How to evaluate” or “How to run this project” checklist for reviewers to assess the repo quickly.
30. Add a section about collaborative delivery, such as working with stakeholders, interpreting business requirements, and translating forecasts into operational actions.
