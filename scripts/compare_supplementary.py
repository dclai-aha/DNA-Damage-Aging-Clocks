"""Compare regression outputs with S4 and S5 reference statistics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from clock_core import DATASET_KEYS, result_dir

ROOT = Path(__file__).resolve().parents[1]


def compare(dataset):
    output = result_dir(ROOT, dataset, "reproduction")
    s4_ref = pd.read_csv(ROOT/"reference"/"Supplementary_S4_CV.csv")
    s5_ref = pd.read_csv(ROOT/"reference"/"Supplementary_S5_PredictedAge.csv")
    cv = pd.read_csv(output/"All_Models_TopN_CV_Performance.csv")
    ref4 = s4_ref[s4_ref.Dataset == dataset]
    a = cv.merge(ref4, on=["Model", "TopN"], how="outer", validate="one_to_one", indicator=True,
                 suffixes=("_run", "_ref"))
    a["CV_Match"] = ((a._merge == "both") &
        (a.CV_R2_run.round(3) == a.CV_R2_ref.round(3)) &
        (a.CV_MAE_run.round(3) == a.CV_MAE_ref.round(3)))
    a["FeatureCount_Match"] = a.N_Final_run == a.N_Final_ref
    a["Known_N2000_AP_Exon_Discrepancy"] = (
        (dataset == "AP_exon") & (a.TopN == 2000) &
        (a.N_Final_run == 77) & (a.N_Final_ref == 76))
    s5 = pd.read_csv(output/"07_Stats"/"BestModels_S5_Statistics.csv")
    ref5 = s5_ref[s5_ref.Dataset == dataset]
    b = s5.merge(ref5, on=["Model", "Tissue"], how="outer", validate="one_to_one", indicator=True,
                 suffixes=("_run", "_ref"))
    b["Delta_Match"] = np.isclose(b.Delta_Age_run, b.Delta_Age_ref, atol=1e-6, rtol=0)
    b["P_Match"] = np.isclose(b.P_S5_Method, b.P_Value, atol=1e-8, rtol=0)
    b["Mean_Match"] = (np.isclose(b.Mean_C_run, b.Mean_C_ref, atol=1e-6, rtol=0) &
                       np.isclose(b.Mean_F_run, b.Mean_F_ref, atol=1e-6, rtol=0))
    a.to_csv(output/"07_Stats"/"S4_comparison.csv", index=False)
    b.to_csv(output/"07_Stats"/"S5_comparison.csv", index=False)
    counts_ok = bool((a.FeatureCount_Match | a.Known_N2000_AP_Exon_Discrepancy).all())
    ok = bool(a.CV_Match.all() and counts_ok and b.Delta_Match.all() and
              b.P_Match.all() and b.Mean_Match.all() and
              len(a) == len(ref4) and len(b) == len(ref5))
    print(f"{dataset}: S4 CV {a.CV_Match.sum()}/{len(a)}, feature counts "
          f"{a.FeatureCount_Match.sum()}/{len(a)} "
          f"(documented exceptions={a.Known_N2000_AP_Exon_Discrepancy.sum()}), "
          f"S5 Delta {b.Delta_Match.sum()}/{len(b)}, "
          f"P {b.P_Match.sum()}/{len(b)}, means {b.Mean_Match.sum()}/{len(b)} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_KEYS])
    args = parser.parse_args()
    keys = DATASET_KEYS if args.dataset == "all" else (args.dataset,)
    if not all([compare(key) for key in keys]):
        raise SystemExit("Supplementary comparison failed")


if __name__ == "__main__":
    main()
