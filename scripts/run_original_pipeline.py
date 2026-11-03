"""Run filtering, Spearman ranking, LassoCV, and regression evaluation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from clock_core import (CONFIG, DATASET_KEYS, save_csv, save_json, result_dir,
                        setup_dirs, evaluate_models)

ROOT = Path(__file__).resolve().parents[1]


def run(dataset):
    out = result_dir(ROOT, dataset, "original")
    setup_dirs(out)
    data = pd.read_csv(ROOT/"data"/"lpkm"/f"{dataset}.csv", index_col=0)
    metadata = ["tissue", "age", "group"]
    genes = [col for col in data.columns if col not in metadata]

    # Global nonzero-ratio filter.
    ratio = (data[genes] > 0).mean()
    stage1 = pd.DataFrame({"Feature": genes, "Nonzero_Ratio_AllSamples": ratio.values,
                           "Keep_Round1": ratio.values >= CONFIG["EXPRESSION_THRESHOLD_1"]})
    stage1 = stage1.sort_values("Nonzero_Ratio_AllSamples", ascending=False)
    save_csv(stage1, out/"02_Filtering"/"Round1_GeneExpressionRatios_AllSamples.csv")
    selected1 = stage1.loc[stage1.Keep_Round1, "Feature"].tolist()
    save_csv(pd.DataFrame({"Feature": selected1}), out/"02_Filtering"/"Round1_Selected_Genes.csv")
    filtered = data[metadata + selected1].copy()

    # Tissue-stratified C-group split; all X records enter training.
    control = filtered[filtered.group == "C"]
    train_control, test_control = [], []
    for tissue in sorted(control.tissue.dropna().unique()):
        group = control[control.tissue == tissue]
        if len(group) > 1:
            sampled = group.sample(frac=CONFIG["C_GROUP_SPLIT_FRAC"],
                                   random_state=CONFIG["RANDOM_STATE"])
            remainder = group.drop(sampled.index)
        else:
            sampled = group.copy()
            remainder = group.iloc[0:0].copy()
        train_control.append(sampled)
        test_control.append(remainder)
    train = pd.concat([filtered[filtered.group == "X"], *train_control])
    test = pd.concat([*test_control, filtered[filtered.group == "F"]])

    # Training-set nonzero-ratio filter and scaling.
    train_genes = [col for col in train.columns if col not in metadata]
    ratio2 = (train[train_genes] > 0).mean()
    stage2 = pd.DataFrame({"Feature": train_genes, "Nonzero_Ratio_TrainSamples": ratio2.values,
                           "Keep_Round2": ratio2.values >= CONFIG["EXPRESSION_THRESHOLD_2"]})
    stage2 = stage2.sort_values("Nonzero_Ratio_TrainSamples", ascending=False)
    save_csv(stage2, out/"02_Filtering"/"Round2_GeneExpressionRatios_TrainSamples.csv")
    selected2 = stage2.loc[stage2.Keep_Round2, "Feature"].tolist()
    save_csv(pd.DataFrame({"Feature": selected2}), out/"02_Filtering"/"Round2_Selected_Genes.csv")
    x = train[selected2]
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)
    scaled_test = pd.DataFrame(scaler.transform(test[selected2]),
                               index=test.index, columns=x.columns)
    save_csv(pd.DataFrame({"Feature": selected2, "Scaler_Mean": scaler.mean_,
                           "Scaler_Scale": scaler.scale_}),
             out/"02_Filtering"/"Scaler_Mean_Std_ByFeature.csv")
    log_age = np.log10(train.age.to_numpy() + 1e-6)

    # Spearman P-value ranking.
    correlations = []
    for feature in scaled.columns:
        rho, pvalue = spearmanr(scaled[feature], log_age, nan_policy="omit")
        correlations.append({"Feature": feature, "Spearman_Corr": rho, "P_Value": pvalue})
    ranked = pd.DataFrame(correlations).sort_values("P_Value", ascending=True)
    save_csv(ranked, out/"Spearman_corr_df.csv")
    save_csv(ranked, out/"03_Spearman"/"Spearman_corr_df.csv")

    sets = {}
    for topn in CONFIG["TARGET_FEATURES_LIST"]:
        first = ranked.head(min(topn, len(ranked)))
        save_csv(first, out/"03_Spearman"/f"N{topn}_Step1_Spearman_Corr.csv")
        candidate = scaled[first.Feature.tolist()]
        lasso = LassoCV(cv=10, random_state=CONFIG["RANDOM_STATE"], max_iter=10000)
        lasso.fit(candidate, log_age)
        coefficients = pd.DataFrame({"Feature": candidate.columns,
                                    "Coefficient": lasso.coef_})
        coefficients["Abs_Coefficient"] = coefficients.Coefficient.abs()
        coefficients = coefficients.sort_values("Abs_Coefficient", ascending=False)
        nonzero = coefficients[coefficients.Coefficient != 0].copy()
        save_csv(coefficients, out/"04_LassoFeatureSelection"/
                 f"N{topn}_Step2_Lasso_All_Coefs.csv")
        save_csv(nonzero, out/"04_LassoFeatureSelection"/
                 f"N{topn}_Step2_Lasso_Nonzero_Coefs.csv")
        sets[topn] = nonzero.Feature.tolist()
        save_csv(pd.DataFrame({"Feature": sets[topn]}), out/"04_LassoFeatureSelection"/
                 f"N{topn}_Selected_Features.csv")
        print(f"{dataset}: LassoCV N{topn} selected {len(sets[topn])} features", flush=True)
    evaluate_models(dataset, train, test, sets, out,
                    scaler_from_full=(scaled, scaled_test, scaler.mean_, scaler.scale_))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_KEYS])
    args = parser.parse_args()
    for dataset in DATASET_KEYS if args.dataset == "all" else [args.dataset]:
        run(dataset)


if __name__ == "__main__":
    main()
