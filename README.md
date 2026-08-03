# Population-Level Classification of Early-Onset Breast Cancer in Kuwait

This repository contains the final analytical workflow supporting the study:

**“Population-Level Classification of Early-Onset Breast Cancer Using National Cancer Registry Data: A Comparative Analysis of Epidemiologic and Machine Learning Models in Kuwait, 2000–2019.”**

The analysis compares conventional epidemiologic regression with machine-learning approaches for classifying early-onset breast cancer among women diagnosed with breast cancer in the Kuwait Cancer Registry.

## Data availability and confidentiality

This repository contains **analysis code only**. It does **not** contain patient-level data, de-identified row-level registry data, imputed datasets, civil identifiers, linkage files, or other restricted Kuwait Cancer Registry materials.

The underlying registry data are not publicly available because they contain potentially identifiable health information and are subject to institutional, ethical, and regulatory restrictions. Access requires appropriate approval from the relevant institutions and data custodians.

## Analytical framework

The final workflow includes:

- data validation and preprocessing;
- targeted missing-data handling within leakage-safe predictive pipelines;
- MICE-pooled multivariable logistic regression for adjusted epidemiologic inference;
- predictive logistic regression, random forest, SVM-RBF, and XGBoost;
- five-fold nested cross-validation repeated five times for internal validation;
- independent held-out evaluation for final fitted models and interpretation;
- probability calibration using Platt scaling;
- training-data classification-threshold selection;
- ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, F1 score, and Brier score;
- calibration intercepts, calibration slopes, and calibration curves;
- paired ROC-AUC comparisons;
- permutation importance;
- feature-level and aggregated variable-level SHAP analyses;
- predictor-domain and missing-data sensitivity analyses;
- complete-case sensitivity analysis; and
- generation of publication-ready tables and figures.

## Repository files

- `eobc_codes_Final_nested_CV.ipynb` — final executable Jupyter notebook.
- `eobc_codes_Final.py` — standalone percent-format Python export of the same workflow.
- `requirements.txt` — pinned Python dependencies used for the final analysis.
- `CITATION.cff` — software citation metadata.
- `.zenodo.json` — Zenodo metadata.
- `LICENSE` — MIT License.

## Computational environment

The final analyses were conducted using:

- Python 3.11.14
- scikit-learn 1.7.2
- XGBoost 3.1.2
- SHAP 0.51.0
- R 4.4.1
- `mice` 3.19.0

The R-assisted inferential workflow also requires the R packages `mice` and `broom`.

## Running the analysis

1. Create and activate a Python environment matching the versions in `requirements.txt`.
2. Confirm that `Rscript` is available from the command line.
3. Install the required R packages:

   ```r
   install.packages(c("mice", "broom"))
   ```

4. Place the authorized analytic dataset in the working directory using the filename expected by the configuration:

   ```text
   early-onset_dta.csv
   ```

5. Review the configuration section before execution, including the data path and output directory.
6. Run the notebook from top to bottom after restarting the kernel and clearing previous outputs.
7. For the initial definitive nested-cross-validation run, use `overwrite=True`. After checkpoints have been created, use `overwrite=False` to resume an interrupted run safely.

Because the registry dataset is restricted and is not distributed with this repository, the code cannot reproduce the numerical results without authorized access to the same analytic data.

## Main validation design

Primary internal validation used five-fold stratified outer cross-validation repeated five times. Hyperparameter tuning was repeated independently within each outer-training fold using three-fold stratified inner cross-validation. Preprocessing, missing-data handling, calibration, threshold selection, and model fitting were performed without access to the corresponding outer validation fold.

A separate stratified 75/25 development split was retained for final fitted models, held-out performance visualization, calibration plots, permutation importance, SHAP interpretation, and sensitivity analyses.

## Outputs

By default, outputs are written to:

```text
eobc_outputs_final/
```

The workflow generates model metrics, nested-cross-validation checkpoints, MICE-pooled estimates, sensitivity-analysis results, calibration outputs, SHAP and permutation-importance results, and publication-ready tables and figures.

Do not commit generated files containing patient-level or row-level information. In particular, do not publish MICE input files, imputed row-level datasets, cached MICE objects containing registry records, prediction files indexed to individual records, or other restricted intermediate outputs.

## Citation

A fixed version of this software is archived on Zenodo.

**Zenodo DOI:** `REPLACE_WITH_VERSION_SPECIFIC_DOI`

The corresponding mutable development repository is:

`https://github.com/h-abouelhassan/Classifying-Early-Onset-Breast-Cancer-in-Kuwait---ML-Code`

For reproducibility, cite the Zenodo DOI associated with the exact release used for the manuscript.

## License

The analysis code is released under the MIT License. The license applies only to the code and documentation in this repository. It does not grant access to, or rights over, the underlying Kuwait Cancer Registry data.
