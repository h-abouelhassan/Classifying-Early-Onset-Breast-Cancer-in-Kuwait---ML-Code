# Early-Onset Breast Cancer Analysis Pipeline

This repository contains the Python analysis pipeline used for the early-onset breast cancer registry study.

## Contents

- `eobc_code_Final.py` — main Python script containing data preparation, imputation, regression, machine-learning model comparison, calibration, and publication-output generation.
- `requirements.txt` — Python package requirements.
- `.gitignore` — prevents private data files, generated outputs, and local environment files from being accidentally committed.

## Data availability

The analysis uses registry data and the raw dataset is not included in this repository. To run the pipeline, place the required input file in the project folder or update the `csv_path` value inside the `EOBCConfig` class.

Default expected input filename:

```text
early-onset_dta.csv
```

## How to run

Create a clean Python environment, install the dependencies, then run the script:

```bash
pip install -r requirements.txt
python eobc_analysis_pipeline_v3.py
```

## Outputs

By default, the script writes results to:

```text
eobc_outputs_v3/
```

Generated outputs may include model-performance tables, pooled logistic-regression outputs, calibration results, feature-importance files, SHAP outputs, and publication-ready figures.

## Reproducibility notes

- The pipeline uses `RANDOM_STATE = 42` for reproducibility where supported.
- The workflow is intended to avoid data leakage by splitting data before fitting preprocessing, imputation, tuning, calibration, and threshold-selection steps.
- Treatment variables are used only as auxiliary variables for targeted imputation where specified and are not included as final predictors.

## Citation

If this code is reused, cite the associated manuscript once available.
