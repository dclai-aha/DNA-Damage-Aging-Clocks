"""Model configuration, cross-validation, prediction, and statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LassoCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TOPN = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500]
CONFIG = {
    "EXPRESSION_THRESHOLD_1": 0.0,
    "EXPRESSION_THRESHOLD_2": 0.3,
    "RANDOM_STATE": 42,
    "C_GROUP_SPLIT_FRAC": 0.5,
    "TARGET_FEATURES_LIST": TOPN,
    "EXCLUDE_TEST_TISSUES": ["P"],
    "WILCOX_ALTERNATIVE": "greater",
    "SAVE_INTERMEDIATE_PLOTS": True,
    "PLOT_DPI": 300,
}

DATASET_KEYS = ("AP_exon", "SSB_exon", "AP_gene", "SSB_gene", "AP_tss5k", "SSB_tss5k")
MODEL_DIRS = {"Linear Regression": "Linear_Regression", "Bayesian Ridge": "Bayesian_Ridge",
              "Ridge": "Ridge", "Elastic Net": "Elastic_Net", "SVR": "SVR"}
TISSUE_NAMES = {"B": "Brain", "H": "Heart", "K": "Kidney", "L": "Liver", "S": "Sperm"}


def model_definitions():
    return [
        ("Linear Regression", LinearRegression()),
        ("Bayesian Ridge", BayesianRidge()),
        ("Ridge", GridSearchCV(Ridge(),
            {"alpha": [0.001, 0.01, 0.1, 1, 3, 5, 7, 9, 10]}, cv=10)),
        ("Elastic Net", GridSearchCV(ElasticNet(),
            {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
             "l1_ratio": [0.1, 0.5, 0.8, 0.9, 1]}, cv=10)),
        ("SVR", GridSearchCV(SVR(),
            {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "gamma": ["scale"]}, cv=10)),
    ]


def save_csv(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def significance(p):
    if pd.isna(p):
        return "NA"
    for cut, stars in ((0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cut:
            return stars
    return "ns"


def group_summary(df):
    c = df[df.group == "C"]
    f = df[df.group == "F"]
    p = mannwhitneyu(f.Predicted_Age_Raw, c.Predicted_Age_Raw,
                     alternative=CONFIG["WILCOX_ALTERNATIVE"]).pvalue
    return {
        "Count_C": len(c), "Count_F": len(f),
        "Mean_Predicted_Age_C": float(c.Predicted_Age_Raw.mean()),
        "Mean_Predicted_Age_F": float(f.Predicted_Age_Raw.mean()),
        "Delta_Age_F_minus_C": float(f.Predicted_Age_Raw.mean() - c.Predicted_Age_Raw.mean()),
        "Test_MAE_Total": float(mean_absolute_error(df.True_Age, df.Predicted_Age_Raw)),
        "Test_MAE_C": float(mean_absolute_error(c.True_Age, c.Predicted_Age_Raw)),
        "Test_MAE_F": float(mean_absolute_error(f.True_Age, f.Predicted_Age_Raw)),
        "Group_P_Value": float(p), "Significance": significance(p),
    }


def tissue_summary(df):
    rows = []
    for tissue in sorted(df.tissue.dropna().unique()):
        sub = df[df.tissue == tissue]
        c = sub[sub.group == "C"].Predicted_Age_Raw
        f = sub[sub.group == "F"].Predicted_Age_Raw
        p = mannwhitneyu(f, c, alternative=CONFIG["WILCOX_ALTERNATIVE"]).pvalue
        rows.append({"Tissue": tissue, "Count_C": len(c), "Count_F": len(f),
                     "Mean_Predicted_Age_C": float(c.mean()), "Mean_Predicted_Age_F": float(f.mean()),
                     "Delta_Age_F_minus_C": float(f.mean()-c.mean()),
                     "P_Value": float(p), "Significance": significance(p)})
    return pd.DataFrame(rows)


def make_s5_rows(df, model, topn, dataset):
    """Calculate S5 statistics from a fitted model's test predictions."""
    region = "Exon" if dataset.endswith("exon") else "Gene" if dataset.endswith("gene") else "TSS ±5 kb"
    rows = []
    for tissue in ("All tissues", "Brain", "Heart", "Kidney", "Liver", "Sperm"):
        inv = {v: k for k, v in TISSUE_NAMES.items()}
        d = df if tissue == "All tissues" else df[df.tissue == inv[tissue]]
        c = d.loc[d.group == "C", "Predicted_Age_Raw"].to_numpy()
        f = d.loc[d.group == "F", "Predicted_Age_Raw"].to_numpy()
        p_exact = mannwhitneyu(f, c, alternative=CONFIG["WILCOX_ALTERNATIVE"], method="exact").pvalue
        p_asym = mannwhitneyu(f, c, alternative=CONFIG["WILCOX_ALTERNATIVE"], method="asymptotic").pvalue
        # S5 reference methods: exact for Exon and within-tissue tests;
        # asymptotic for pooled Gene and TSS tests.
        paper_p = p_exact if region == "Exon" or tissue != "All tissues" else p_asym
        rows.append({"Dataset": dataset, "Region": region, "Lesion": dataset.split("_")[0],
            "Model": model, "TopN": topn, "Tissue": tissue,
            "N_C": len(c), "Mean_C": float(np.mean(c)), "SD_C": float(np.std(c, ddof=1)),
            "Median_C": float(np.median(c)),
            "N_F": len(f), "Mean_F": float(np.mean(f)), "SD_F": float(np.std(f, ddof=1)),
            "Median_F": float(np.median(f)), "Delta_Age": float(np.mean(f)-np.mean(c)),
            "P_Exact": float(p_exact), "P_Asymptotic": float(p_asym), "P_S5_Method": float(paper_p),
            "S5_Test_Method": "exact" if region == "Exon" or tissue != "All tissues" else "asymptotic"})
    return rows


def result_dir(root, dataset, mode):
    lesion, region = dataset.split("_", 1)
    parent = {"exon": "Exon", "gene": "Gene数据集", "tss5k": "Tss5k数据集"}[region]
    return Path(root) / "results" / mode / parent / ("Result_" + lesion + "_Exon")


def setup_dirs(out):
    for section in ("00_Config", "01_DataSplits", "02_Filtering", "03_Spearman",
                    "04_LassoFeatureSelection", "05_CV_Performance", "06_AllTopN_ModelOutputs",
                    "07_Stats", "08_Manifest"):
        (out/section).mkdir(parents=True, exist_ok=True)
    for slug in MODEL_DIRS.values():
        (out/slug/"BestTopN_Test").mkdir(parents=True, exist_ok=True)


def plot_cv(cv_df, topn, out):
    if not CONFIG["SAVE_INTERMEDIATE_PLOTS"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(cv_df.Model, cv_df.CV_R2, color="#2E86AB", alpha=0.85)
    axes[0].set_title(f"TopN {topn} | CV R²")
    axes[1].plot(cv_df.Model, cv_df.CV_MAE, "s-", color="#A23B72")
    axes[1].set_title(f"TopN {topn} | CV MAE")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(out/"05_CV_Performance"/f"N{topn}_Plot_CV_Metrics.png", dpi=CONFIG["PLOT_DPI"])
    plt.close(fig)


def plot_cv_scatter(train_age, predictions, topn, out):
    if not CONFIG["SAVE_INTERMEDIATE_PLOTS"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 4))
    for ax, (name, values) in zip(np.atleast_1d(axes), predictions.items()):
        ax.scatter(train_age, values, color="#C02D45", alpha=0.55, s=16)
        low = min(np.min(train_age), np.min(values))
        high = max(np.max(train_age), np.max(values))
        ax.plot([low, high], [low, high], "k--")
        ax.set(xlabel="True Age", ylabel="CV-Predicted Age",
               title=f"{name} | R²={r2_score(train_age, values):.3f}")
    fig.tight_layout()
    fig.savefig(out/"05_CV_Performance"/f"N{topn}_Plot_CV_Scatter.png", dpi=CONFIG["PLOT_DPI"])
    plt.close(fig)


def evaluate_models(dataset, train, test, selected_by_topn, out, scaler_from_full=None):
    """Run the original CV, model fitting, MAE and best-TopN selection for either workflow."""
    setup_dirs(out)
    save_json({"dataset": dataset, "workflow": out.parents[1].name, **CONFIG}, out/"00_Config"/"run_config.json")
    save_csv(train, out/"Raw_Data_Train_Split.csv", index=True)
    save_csv(test, out/"Raw_Data_Test_Split.csv", index=True)
    save_csv(train, out/"01_DataSplits"/"Raw_Data_Train_Split.csv", index=True)
    save_csv(test, out/"01_DataSplits"/"Raw_Data_Test_Split.csv", index=True)
    for label, df in (("Train", train), ("Test", test)):
        save_csv(pd.DataFrame({"SampleID": df.index, "group": df.group.to_numpy(),
            "tissue": df.tissue.to_numpy(), "age": df.age.to_numpy(), "split": label.lower()}),
            out/"01_DataSplits"/f"{label}_Sample_Metadata.csv")
    save_json({"n_train_total": len(train), "n_test_total": len(test),
               "train_group_counts": train.group.value_counts().to_dict(),
               "test_group_counts": test.group.value_counts().to_dict()},
              out/"01_DataSplits"/"split_manifest.json")
    y = np.log10(train.age.to_numpy(dtype=float) + 1e-6)
    train_age = np.power(10, y) - 1e-6
    kfold = KFold(n_splits=10, shuffle=True, random_state=CONFIG["RANDOM_STATE"])
    all_cv, all_group, all_tissue, best = [], [], [], {}
    manifests = []
    for topn in CONFIG["TARGET_FEATURES_LIST"]:
        features = selected_by_topn.get(topn, [])
        if not features:
            print(f"{dataset} TopN={topn}: zero features; skipped", flush=True)
            continue
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicated LASSO features: {dataset} N{topn}")
        unknown = set(features) - set(train.columns)
        if unknown:
            raise ValueError(f"LPKM is missing {len(unknown)} features: {sorted(unknown)[:5]}")
        if scaler_from_full is None:
            scaler = StandardScaler()
            x = scaler.fit_transform(train[features].to_numpy(dtype=float))
            xtest = scaler.transform(test[features].to_numpy(dtype=float))
            scaler_means = scaler.mean_; scaler_scales = scaler.scale_
        else:
            # Original workflow fits the scaler on all stage-2 features before TopN.
            full_x, full_xtest, full_means, full_scales = scaler_from_full
            positions = [full_x.columns.get_loc(f) for f in features]
            x = full_x.iloc[:, positions].to_numpy()
            xtest = full_xtest.iloc[:, positions].to_numpy()
            scaler_means = np.asarray(full_means)[positions]
            scaler_scales = np.asarray(full_scales)[positions]
        save_csv(pd.DataFrame({"Feature": features, "Scaler_Mean": scaler_means,
                               "Scaler_Scale": scaler_scales}),
                 out/"02_Filtering"/f"N{topn}_Scaler_Mean_Std_ByFeature.csv")
        cv_current = []
        cv_pred_for_plot = {}
        for name, estimator in model_definitions():
            cv_log = cross_val_predict(estimator, x, y, cv=kfold)
            cv_months = np.power(10, cv_log) - 1e-6
            cv_r2 = float(r2_score(y, cv_log))
            cv_mse = float(mean_squared_error(y, cv_log))
            cv_mae = float(mean_absolute_error(train_age, cv_months))
            record = {"Model": name, "TopN": topn, "N_Spearman": min(topn, len(train.columns)-3),
                      "N_Final": len(features), "CV_R2": cv_r2, "CV_MSE": cv_mse, "CV_MAE": cv_mae}
            cv_current.append(record)
            all_cv.append(record)
            cv_pred_for_plot[name] = cv_months
            fitted = clone(estimator).fit(x, y)
            fitted_train = np.power(10, fitted.predict(x)) - 1e-6
            predicted_test = np.power(10, fitted.predict(xtest)) - 1e-6
            train_prediction = pd.DataFrame({"SampleID": train.index,
                "True_Age": train.age.to_numpy(), "CV_Predicted_Age": cv_months,
                "Fitted_Predicted_Age": fitted_train, "group": train.group.to_numpy(),
                "tissue": train.tissue.to_numpy(), "Model": name, "TopN": topn})
            test_prediction = pd.DataFrame({"SampleID": test.index,
                "Predicted_Age_Raw": predicted_test, "True_Age": test.age.to_numpy(),
                "group": test.group.to_numpy(), "tissue": test.tissue.to_numpy(),
                "Model": name, "TopN": topn, "Value": predicted_test})
            if CONFIG["EXCLUDE_TEST_TISSUES"]:
                test_prediction = test_prediction[
                    ~test_prediction.tissue.isin(CONFIG["EXCLUDE_TEST_TISSUES"])].copy()
            slug = MODEL_DIRS[name]
            base = out/"06_AllTopN_ModelOutputs"/slug
            save_csv(train_prediction, base/f"{slug}_TopN_{topn}_Prediction_Train.csv")
            save_csv(test_prediction, base/f"{slug}_TopN_{topn}_Prediction_Test.csv")
            group = group_summary(test_prediction)
            tissues = tissue_summary(test_prediction)
            save_csv(pd.DataFrame([group]), base/f"{slug}_TopN_{topn}_GroupSummary.csv")
            save_csv(tissues, base/f"{slug}_TopN_{topn}_TissueSummary.csv")
            all_group.append({"Model": name, "TopN": topn, **group})
            for _, r in tissues.iterrows():
                all_tissue.append({"Model": name, "TopN": topn, **r.to_dict()})
            candidate = {"Model": name, "Best_TopN": topn, "Best_CV_R2": cv_r2,
                "Best_CV_MSE": cv_mse, "Best_CV_MAE": cv_mae, "N_Final": len(features),
                "Test_MAE_Total": group["Test_MAE_Total"], "Test_MAE_C": group["Test_MAE_C"],
                "Test_MAE_F": group["Test_MAE_F"], "Group_P_Value": group["Group_P_Value"]}
            prev = best.get(name)
            if (prev is None or cv_r2 > prev["Best_CV_R2"] or
                (np.isclose(cv_r2, prev["Best_CV_R2"]) and cv_mae < prev["Best_CV_MAE"])):
                best[name] = candidate
                # Retain the best candidate tables until the final TopN is determined.
                best[name]["_train_table"] = train_prediction
                best[name]["_test_table"] = test_prediction
            manifests.append({"Model": name, "TopN": topn,
                "File": str((base/f"{slug}_TopN_{topn}_Prediction_Test.csv").relative_to(out))})
        save_csv(pd.DataFrame(cv_current),
                 out/"05_CV_Performance"/f"N{topn}_CV_Performance_ByModel.csv")
        plot_cv(pd.DataFrame(cv_current), topn, out)
        plot_cv_scatter(train_age, cv_pred_for_plot, topn, out)
        print(f"{dataset}: completed TopN={topn}, features={len(features)}", flush=True)
    if not best:
        raise RuntimeError(f"All TopN feature sets are empty for {dataset}")
    s5_rows = []
    for name, candidate in best.items():
        slug = MODEL_DIRS[name]
        topn = candidate["Best_TopN"]
        save_csv(candidate.pop("_train_table"), out/slug/"BestTopN_Test"/
                 f"{slug}_BestTopN_{topn}_Prediction_Train.csv")
        predicted = candidate.pop("_test_table")
        save_csv(predicted, out/slug/"BestTopN_Test"/
                 f"{slug}_BestTopN_{topn}_Prediction_Test.csv")
        s5_rows.extend(make_s5_rows(predicted, name, topn, dataset))
    cv = pd.DataFrame(all_cv).sort_values(["Model", "TopN"])
    final = pd.DataFrame(best.values()).sort_values("Model")
    save_csv(cv, out/"All_Models_TopN_CV_Performance.csv")
    save_csv(final[["Model", "Best_TopN", "Best_CV_R2", "Best_CV_MAE", "Best_CV_MSE", "N_Final"]],
             out/"Model_BestTopN_Summary.csv")
    save_csv(final[["Model", "Best_TopN", "Test_MAE_Total", "Test_MAE_C", "Test_MAE_F", "Group_P_Value"]],
             out/"Final_Model_Test_Summary_ByModelBestTopN.csv")
    save_csv(pd.DataFrame(all_group), out/"07_Stats"/"AllTopN_GroupSummary.csv")
    save_csv(pd.DataFrame(all_tissue), out/"07_Stats"/"AllTopN_TissueSummary.csv")
    save_csv(pd.DataFrame(s5_rows), out/"07_Stats"/"BestModels_S5_Statistics.csv")
    save_csv(cv, out/"08_Manifest"/"All_Models_TopN_CV_Performance.csv")
    save_csv(final, out/"08_Manifest"/"Best_Model_Tracker_Full.csv")
    save_csv(pd.DataFrame(manifests), out/"08_Manifest"/"Detailed_File_Manifest.csv")
    return final, pd.DataFrame(s5_rows)
