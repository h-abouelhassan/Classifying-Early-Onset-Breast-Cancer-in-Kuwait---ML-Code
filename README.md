# Early-Onset Breast Cancer Analysis Pipeline

This repository contains the reproducible analysis code used for the study:

**Population-Level Classification of Early-Onset Breast Cancer Using National Cancer Registry Data: A Comparative Analysis of Epidemiologic and Machine Learning Models in Kuwait, 2000–2019**

## Repository contents

- `eobc_code_Final.py` — complete executable Python analysis pipeline.
- `requirements.txt` — required Python packages.
- `.gitignore` — excludes confidential data, generated outputs, and local environment files.

The original Jupyter notebook may also be retained in the repository for transparency, but the Python script is the main reusable analysis file.

## Data availability

The analysis uses confidential patient-level data from the Kuwait Cancer Registry. The raw dataset is not included in this repository and must not be uploaded to GitHub.

To run the pipeline with authorized data, update the `csv_path` value in the `EOBCConfig` class. The notebook currently expects:

```text
early-onset_dta.csv
```

## Python setup

Create and activate a clean virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## R requirement for MICE

The pooled inferential logistic-regression section calls R from Python. Install R and the following R packages before running the full pipeline:

```r
install.packages(
  c("mice", "broom", "future", "furrr"),
  repos = "https://cloud.r-project.org"
)
```

Confirm that `Rscript` is available from the command line:

```bash
Rscript --version
```

## Running the analysis

```bash
python eobc_analysis_pipeline.py
```

The complete workflow is computationally intensive because it includes model tuning, calibration, sensitivity analyses, multiple imputation, permutation importance, and SHAP analysis.

## Outputs

Outputs are written under the directory configured in `EOBCConfig.output_dir`. Generated files include:

- model-performance tables;
- sensitivity-analysis results;
- pooled MICE logistic-regression estimates;
- reference-category tables;
- XGBoost permutation-importance results;
- SHAP tables and figures;
- ROC curves;
- baseline-characteristics tables;
- publication-ready figures and output manifests.

## Reproducibility

- The pipeline uses `RANDOM_STATE = 42` where supported.
- The train/test split is stratified by EOBC status.
- Preprocessing, tuning, calibration, and threshold selection are learned from training data.
- Treatment variables are used only as auxiliary variables for imputation and are not included as final predictors.
- The test set is reserved for final performance evaluation.

## Citation

Please cite the associated manuscript when using or adapting this code. Full citation details will be added after publication.
