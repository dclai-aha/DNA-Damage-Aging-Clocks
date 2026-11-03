"""Fit regression models using fixed samples and stored LASSO features."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from clock_core import (CONFIG, DATASET_KEYS, TOPN, save_csv, result_dir,
                        setup_dirs, evaluate_models)

ROOT = Path(__file__).resolve().parents[1]


def match_ids(data, ids, lesion, dataset):
    matched = []
    for sid in ids:
        if sid in data.index:
            matched.append(sid)
        elif dataset.endswith("tss5k") and f"{lesion}.{sid}" in data.index:
            matched.append(f"{lesion}.{sid}")
        else:
            raise ValueError(f"{dataset}: missing sample {sid!r}")
    return matched


def load_fixed_split(dataset, data):
    lesion = dataset.split("_")[0]
    fixed = ROOT/"metadata"
    training = pd.read_csv(fixed/f"{lesion}_Train_Sample_Metadata.csv")
    testing = pd.read_csv(fixed/f"{lesion}_Test_Sample_Metadata.csv")
    assert training.SampleID.is_unique and testing.SampleID.is_unique
    assert not set(training.SampleID).intersection(testing.SampleID)
    train = data.loc[match_ids(data, training.SampleID, lesion, dataset)].copy()
    test = data.loc[match_ids(data, testing.SampleID, lesion, dataset)].copy()
    for frame, meta, label in ((train, training, "training"), (test, testing, "test")):
        if not np.array_equal(frame.group.to_numpy(), meta.group.to_numpy()):
            raise ValueError(f"{dataset}: {label} group mismatch")
        if not np.array_equal(frame.tissue.to_numpy(), meta.tissue.to_numpy()):
            raise ValueError(f"{dataset}: {label} tissue mismatch")
        if not np.allclose(frame.age.to_numpy(), meta.age.to_numpy()):
            raise ValueError(f"{dataset}: {label} age mismatch")
    train.index = training.SampleID
    test.index = testing.SampleID
    expected_train = {"X": 83 if lesion == "AP" else 84, "C": 20}
    if train.group.value_counts().to_dict() != expected_train:
        raise ValueError(f"{dataset}: unexpected training groups")
    if test.group.value_counts().to_dict() != {"C": 19, "F": 40}:
        raise ValueError(f"{dataset}: unexpected test groups")
    if set(test.tissue) != {"B", "H", "K", "L", "S"}:
        raise ValueError(f"{dataset}: unexpected test tissues")
    return train, test


def load_historical_features(dataset, feature_columns):
    directory = ROOT/"features"/dataset
    correlations = pd.read_csv(directory/"Spearman_corr_df.csv")
    if not correlations.Feature.is_unique:
        raise ValueError(f"{dataset}: duplicated historical Spearman features")
    permitted = set(feature_columns)
    if not set(correlations.Feature).issubset(permitted):
        raise ValueError(f"{dataset}: Spearman file contains features absent from LPKM")
    sets = {}
    for topn in TOPN:
        path = directory/f"N{topn}_Step2_Lasso_Nonzero_Coefs.csv"
        df = pd.read_csv(path)
        if "Feature" not in df or "Coefficient" not in df:
            raise ValueError(f"{dataset}: missing LASSO columns in {path.name}")
        if (not df.Feature.is_unique or df.Feature.isna().any() or
            (df.Coefficient == 0).any() or df.Coefficient.isna().any()):
            raise ValueError(f"{dataset}: invalid LASSO nonzero list N{topn}")
        genes = df.Feature.tolist()
        missing = set(genes) - permitted
        if missing:
            raise ValueError(f"{dataset}: {len(missing)} LASSO genes not found in LPKM")
        if not set(genes).issubset(set(correlations.Feature.iloc[:topn])):
            raise ValueError(f"{dataset}: historical LASSO list is not a subset of Spearman TopN={topn}")
        selected = pd.read_csv(directory/f"N{topn}_Selected_Features.csv")
        if selected.Feature.tolist() != genes:
            raise ValueError(f"{dataset}: selected-features file differs from LASSO list N{topn}")
        sets[topn] = genes
    return correlations, sets


def run(dataset, check_only=False):
    data = pd.read_csv(ROOT/"data"/"lpkm"/f"{dataset}.csv", index_col=0)
    if not data.index.is_unique:
        raise ValueError(f"{dataset}: duplicated sample identifiers")
    train, test = load_fixed_split(dataset, data)
    correlations, features = load_historical_features(dataset, data.columns)
    print(f"{dataset}: train={len(train)}, test={len(test)}, "
          f"Spearman={len(correlations)}, "
          f"LASSO={[len(features[n]) for n in TOPN]}", flush=True)
    if check_only:
        return
    out = result_dir(ROOT, dataset, "reproduction")
    setup_dirs(out)
    save_csv(correlations, out/"03_Spearman"/"Spearman_corr_df.csv")
    for n in TOPN:
        directory = ROOT/"features"/dataset
        save_csv(pd.DataFrame({"Feature": features[n]}),
                 out/"04_LassoFeatureSelection"/f"N{n}_Selected_Features.csv")
        for kind in ("Nonzero_Coefs", "All_Coefs"):
            source = directory/f"N{n}_Step2_Lasso_{kind}.csv"
            save_csv(pd.read_csv(source), out/"04_LassoFeatureSelection"/source.name)
    evaluate_models(dataset, train, test, features, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_KEYS])
    parser.add_argument("--check-only", action="store_true",
                        help="Validate all inputs without fitting models")
    args = parser.parse_args()
    for dataset in DATASET_KEYS if args.dataset == "all" else [args.dataset]:
        run(dataset, check_only=args.check_only)


if __name__ == "__main__":
    main()
