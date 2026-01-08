import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(REPO_ROOT, "metrics")

FILES = {
    "EfficientNet-B2": os.path.join(METRICS_DIR, "xai_efficientnet_b2_deletion.csv"),
    "ResNet-50":       os.path.join(METRICS_DIR, "xai_resnet50_deletion.csv"),
    "MobileNet-V3":    os.path.join(METRICS_DIR, "xai_mobilenet_v3_deletion.csv"),
}

STABILITY = {
    "EfficientNet-B2": os.path.join(METRICS_DIR, "xai_efficientnet_b2_stability.csv"),
    "ResNet-50":       os.path.join(METRICS_DIR, "xai_resnet50_stability.csv"),
    "MobileNet-V3":    os.path.join(METRICS_DIR, "xai_mobilenet_v3_stability.csv"),
}

def deletion_auc_from_mean(df_del: pd.DataFrame) -> float:
    g = df_del.groupby("frac_masked")["prob_after"].mean().sort_index()
    x = g.index.to_numpy(dtype=float)
    y = g.to_numpy(dtype=float)
    return float(np.trapz(y, x))

def negative_drop_pct(df_del: pd.DataFrame) -> float:
    if "drop_abs" not in df_del.columns:
        df_del = df_del.copy()
        df_del["drop_abs"] = df_del["prob_before"] - df_del["prob_after"]
    return float((df_del["drop_abs"] < 0).mean() * 100.0)

def overall_stability(df_stab: pd.DataFrame) -> tuple[float, float]:
    pearson = float(df_stab["pearson"].mean()) if "pearson" in df_stab.columns else float("nan")
    ssim    = float(df_stab["ssim"].mean())    if "ssim" in df_stab.columns else float("nan")
    return pearson, ssim

def main():
    # 1) Combined deletion plot (box markers + error bars)
    plt.figure(figsize=(8, 5))
    for label, path in FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        df = pd.read_csv(path)
        grp = df.groupby("frac_masked")["prob_after"]
        mean = grp.mean().sort_index()
        std  = grp.std().sort_index()

        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        yerr = std.to_numpy(dtype=float)

        plt.errorbar(x, y, yerr=yerr, marker='s', linestyle='-', capsize=3, label=label)

    plt.title("Deletion faithfulness comparison (mean ± SD)")
    plt.xlabel("Fraction masked (top Grad-CAM++ pixels)")
    plt.ylabel("Mean P(target class) after masking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_fig = os.path.join(METRICS_DIR, "deletion_comparison_3models_box.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print("Saved:", out_fig)

    # 2) Compact summary table
    rows = []
    for model in FILES.keys():
        df_del = pd.read_csv(FILES[model])
        df_stb = pd.read_csv(STABILITY[model])
        pearson, ssim = overall_stability(df_stb)

        rows.append({
            "Model": model,
            "Deletion AUC (mean prob_after)": deletion_auc_from_mean(df_del),
            "Negative drop (%)": negative_drop_pct(df_del),
            "Overall Pearson": pearson,
            "Overall SSIM": ssim,
        })

    out_csv = os.path.join(METRICS_DIR, "xai_compact_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
