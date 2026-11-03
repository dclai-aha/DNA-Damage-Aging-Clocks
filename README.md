# DNA Damage-Based Aging Clocks

Python workflows for age prediction using genome-wide apurinic/apyrimidinic (AP) site and DNA single-strand break (SSB) profiles from mouse exon, gene-body, and transcription start site (TSS ±5 kb) regions.

## Repository structure

```text
DNA-Damage-Aging-Clocks/
├── scripts/
│   ├── clock_core.py
│   ├── run_original_pipeline.py
│   ├── run_reproduction.py
│   └── compare_supplementary.py
├── data/lpkm/
│   ├── AP_exon.csv
│   ├── SSB_exon.csv
│   ├── AP_gene.csv
│   ├── SSB_gene.csv
│   ├── AP_tss5k.csv
│   └── SSB_tss5k.csv
├── metadata/
│   ├── AP_Train_Sample_Metadata.csv
│   ├── AP_Test_Sample_Metadata.csv
│   ├── SSB_Train_Sample_Metadata.csv
│   └── SSB_Test_Sample_Metadata.csv
├── features/
│   ├── AP_exon/
│   ├── SSB_exon/
│   ├── AP_gene/
│   ├── SSB_gene/
│   ├── AP_tss5k/
│   └── SSB_tss5k/
├── reference/
│   ├── Supplementary_S4_CV.csv
│   └── Supplementary_S5_PredictedAge.csv
├── docs/
│   ├── VALIDATION.md
│   └── CHECKSUMS_SHA256.txt
├── requirements.txt
└── results/
```

Each `features/<dataset>/` directory contains `Spearman_corr_df.csv` and eight sets of LASSO outputs (`N100`, `N500`, `N1000`, `N1500`, `N2000`, `N2500`, `N3000`, and `N3500`):

- `N{TopN}_Step2_Lasso_All_Coefs.csv`
- `N{TopN}_Step2_Lasso_Nonzero_Coefs.csv`
- `N{TopN}_Selected_Features.csv`

## Software environment

| Component | Version |
| --- | --- |
| Python | 3.13.9 |
| NumPy | 2.4.4 |
| pandas | 2.3.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.6 |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |

The dependencies are specified in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Input data

The six CSV files under `data/lpkm/` contain processed LPKM values. The first column contains sample identifiers; `tissue`, `age`, and `group` are metadata columns, followed by gene-associated LPKM features.

The input matrices contain samples in groups `X`, `C`, and `F`. For groups `C` and `F`, tissues are brain (`B`), heart (`H`), kidney (`K`), liver (`L`), and sperm (`S`). Group `X` retains the tissue composition of the age-stratified dataset.

Sample-partition CSV files under `metadata/` contain `SampleID`, `group`, `tissue`, `age`, and `split`. AP has 103 training samples (83 X, 20 C) and 59 test samples (19 C, 40 F); SSB has 104 training samples (84 X, 20 C) and 59 test samples (19 C, 40 F). The reproduction workflow uses these partitions without random resampling.

## Original analysis workflow

```bash
python scripts/run_original_pipeline.py --dataset all
```

The original workflow performs:

1. Global feature nonzero-ratio filtering with a threshold of `0.0`.
2. Tissue-stratified random assignment of 50% of group `C` to training (`random_state=42`); group `X` is assigned to training and group `F` to testing.
3. Training-set feature nonzero-ratio filtering with a threshold of `0.3`, followed by `StandardScaler` fitted on the training data.
4. Spearman correlation ranking by P value and LASSO feature selection for TopN values `100, 500, 1000, 1500, 2000, 2500, 3000, 3500` (`LassoCV`, `cv=10`, `random_state=42`, `max_iter=10000`).
5. Training and evaluation of Linear Regression, Bayesian Ridge, Ridge, Elastic Net, and SVR, using 10-fold cross-validation (`KFold`, `shuffle=True`, `random_state=42`).
6. Selection of each model's TopN using the maximum CV R², with CV MAE as a tie-breaker.
7. Predictions on the test set and one-sided Mann–Whitney U comparisons (HFD greater than Control).

Hyperparameter grids and evaluation methods are defined in `scripts/clock_core.py`. This workflow regenerates feature rankings and LASSO selections; its output is not constrained to the stored feature sets.

## Reproduction using stored features

```bash
python scripts/run_reproduction.py --check-only
python scripts/run_reproduction.py --dataset all
python scripts/compare_supplementary.py --dataset all
```

The reproduction workflow:

1. Reads the same six LPKM matrices and the four fixed training/test sample lists.
2. Reads each dataset's `Spearman_corr_df.csv` and checks that the stored LASSO features belong to the corresponding Spearman TopN.
3. Uses the nonzero feature names in each `N{TopN}_Step2_Lasso_Nonzero_Coefs.csv` directly; the LASSO coefficients are not supplied as regression weights.
4. Applies the model configurations and cross-validation settings used in the original workflow.
5. Writes CV performance, test predictions, group and tissue summaries, and S5 comparison statistics.

`--dataset` accepts `all`, `AP_exon`, `SSB_exon`, `AP_gene`, `SSB_gene`, `AP_tss5k`, or `SSB_tss5k`. Outputs are stored under `results/original/` or `results/reproduction/`. The `results/` directory is excluded from version control.

## Statistical conventions

- Regression target: `log10(age + 1e-6)`.
- CV R² and MSE: evaluated on the log-transformed age scale.
- MAE: evaluated in months after inverse transformation.
- Δage: mean predicted age in group `F` minus mean predicted age in group `C`.
- Model output P values: one-sided SciPy `mannwhitneyu` using its default method.
- S5 reference comparisons: exact one-sided P values for exon models; asymptotic P values for gene-body and TSS ±5 kb pooled-tissue comparisons; exact P values for their within-tissue comparisons.
- Feature filtering and selection in the original workflow are performed before its outer model cross-validation, as in the source analysis.

## Supplementary-table comparison

`reference/` contains the S4 model-performance and S5 predicted-age statistics used by `scripts/compare_supplementary.py`. Numerical comparison rules and recorded discrepancies are documented in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Data scope

The repository contains filtered LPKM analysis matrices, not raw sequencing reads.
