import os
import json
import time
import pandas as pd

def file_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def eval_artifacts(csv_path: str = None, fig_dir: str = None, draft_path: str = None, expected_figures: list = None, thresholds_override: dict = None):
    CSV_PATH = csv_path or "artifacts/nobel/laureates_prizes.csv"
    FIG_DIR = fig_dir or "artifacts/nobel/figures"
    FIG_FILES = expected_figures or [
        "top_countries.html",
        "yearly_trend.html",
        "category_stacked.html",
    ]
    DRAFT_PATH = draft_path or "artifacts/nobel/draft.md"

    checks = {
        "csv_exists": file_ok(CSV_PATH),
        "fig_dir_exists": os.path.isdir(FIG_DIR),
        "draft_exists": file_ok(DRAFT_PATH),
    }
    for f in FIG_FILES:
        checks[f"fig_{f}_exists"] = file_ok(os.path.join(FIG_DIR, f))

    metrics = {}
    notes = []

    if checks["csv_exists"]:
        df = pd.read_csv(CSV_PATH)
        # Coerce year to numeric
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
        # Basic metrics
        metrics["rows"] = int(len(df))
        metrics["laureates_unique"] = int(df["id"].nunique() if "id" in df.columns else 0)
        metrics["year_missing_rate"] = round(float(df["year"].isna().mean()), 4)
        metrics["category_missing_rate"] = round(float(df.get("category").isna().mean() if "category" in df.columns else 1.0), 4)
        metrics["bornCountry_missing_rate"] = round(float(df.get("bornCountry").isna().mean() if "bornCountry" in df.columns else 1.0), 4)

        # Year coverage
        df_year = df.dropna(subset=["year"])
        if len(df_year) > 0:
            metrics["year_min"] = int(df_year["year"].min())
            metrics["year_max"] = int(df_year["year"].max())
        else:
            metrics["year_min"] = None
            metrics["year_max"] = None
            notes.append("All year values are missing after coercion.")

        # Top country
        if "bornCountry" in df.columns:
            top = (
                df["bornCountry"]
                .fillna("Unknown")
                .value_counts()
                .reset_index(name="count")
                .rename(columns={"index": "bornCountry"})
            )
            if len(top) > 0:
                metrics["top_country_name"] = str(top.iloc[0]["bornCountry"])
                metrics["top_country_count"] = int(top.iloc[0]["count"])
            else:
                metrics["top_country_name"] = None
                metrics["top_country_count"] = 0

    # Simple rubric
    thresholds_default = {
        "min_rows": 1000,  # 粗略阈值：数据行数至少达到该值
        "max_missing_rate": 0.5  # 任一关键字段缺失占比不超过 50%
    }
    thresholds = thresholds_override or thresholds_default
    rubric = {
        "data_volume_ok": metrics.get("rows", 0) >= thresholds["min_rows"],
        "year_missing_ok": metrics.get("year_missing_rate", 1.0) <= thresholds["max_missing_rate"],
        "category_missing_ok": metrics.get("category_missing_rate", 1.0) <= thresholds["max_missing_rate"],
        "bornCountry_missing_ok": metrics.get("bornCountry_missing_rate", 1.0) <= thresholds["max_missing_rate"],
        "figures_ok": all(checks.get(f"fig_{f}_exists", False) for f in FIG_FILES),
        "draft_ok": checks.get("draft_exists", False),
    }

    ok = all([
        checks["csv_exists"],
        checks["fig_dir_exists"],
        rubric["data_volume_ok"],
        rubric["year_missing_ok"],
        rubric["category_missing_ok"],
        rubric["bornCountry_missing_ok"],
        rubric["figures_ok"],
        rubric["draft_ok"],
    ])

    report = {
        "timestamp": int(time.time()),
        "checks": checks,
        "metrics": metrics,
        "rubric": rubric,
        "thresholds": thresholds,
        "status_ok": ok,
        "notes": notes,
        "artifacts": {
            "csv": CSV_PATH,
            "figures_dir": FIG_DIR,
            "expected_figures": FIG_FILES,
            "draft_md": DRAFT_PATH
        }
    }

    # 若提供了 run_dir，则将评估报告写入该目录；否则写入默认目录
    out_dir = os.path.dirname(DRAFT_PATH) if draft_path else "artifacts/nobel"
    os.makedirs(out_dir, exist_ok=True)
    # 根据草稿文件名区分报告前缀，避免覆盖
    tag = "eval_agent" if os.path.basename(DRAFT_PATH) != "draft.md" else "eval"
    out_path = os.path.join(out_dir, f"{tag}_{report['timestamp']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Eval complete. Report saved -> {out_path}")

def main():
    # 统一从 run_dir 派生图表目录，并优先使用 --draft（如 draft_agent.md）
    out_dir = args.run_dir
    fig_dir = os.path.join(out_dir, "figures")
    draft_path = args.draft if args.draft else os.path.join(out_dir, "draft.md")

    # 若提供 --thresholds，则加载 JSON 覆盖默认阈值
    thresholds_data = None
    if getattr(args, "thresholds", None):
        try:
            with open(args.thresholds, "r", encoding="utf-8") as f:
                thresholds_data = json.load(f)
        except Exception as e:
            print(f"[warn] Failed to load thresholds from {args.thresholds}: {e}")

    eval_artifacts(csv_path=args.csv, fig_dir=fig_dir, draft_path=draft_path, thresholds_override=thresholds_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Nobel pipeline artifacts")
    parser.add_argument("--csv", type=str, required=True, help="Path to source CSV.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing artifacts.")
    parser.add_argument("--draft", type=str, default=None, help="Optional draft markdown path to evaluate.")
    parser.add_argument("--thresholds", type=str, default=None, help="Optional JSON file to override evaluation thresholds.")
    args = parser.parse_args()
    main()