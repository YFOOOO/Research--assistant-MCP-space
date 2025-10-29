import os
import pandas as pd
import altair as alt

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["bornCountry"] = df["bornCountry"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")
    return df.dropna(subset=["year"])

def save_chart(chart: alt.Chart, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    chart.save(out_path)

def top_countries_chart(df: pd.DataFrame, top_n: int = 15) -> alt.Chart:
    c = (
        df.groupby("bornCountry")
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .head(top_n)
    )
    return alt.Chart(c).mark_bar().encode(
        x=alt.X("count:Q", title="获奖人数"),
        y=alt.Y("bornCountry:N", sort="-x", title="出生国家"),
        tooltip=["bornCountry", "count"]
    ).properties(title=f"诺贝尔奖获奖者出生国家分布 Top {top_n}", width=700)

def yearly_trend_chart(df: pd.DataFrame) -> alt.Chart:
    y = (
        df.groupby("year")
          .size()
          .reset_index(name="count")
          .sort_values("year")
    )
    return alt.Chart(y).mark_line(point=True).encode(
        x=alt.X("year:Q", title="年份"),
        y=alt.Y("count:Q", title="获奖人数"),
        tooltip=["year", "count"]
    ).properties(title="年度获奖人数趋势", width=700)

def category_stacked_chart(df: pd.DataFrame) -> alt.Chart:
    yc = (
        df.groupby(["year", "category"])
          .size()
          .reset_index(name="count")
          .sort_values(["year", "category"])
    )
    return alt.Chart(yc).mark_area().encode(
        x=alt.X("year:Q", title="年份"),
        y=alt.Y("count:Q", stack="normalize", title="占比"),
        color=alt.Color("category:N", title="奖项类别"),
        tooltip=["year", "category", "count"]
    ).properties(title="按奖项类别的年度占比（归一化）", width=700)

def write_draft(df: pd.DataFrame, out_path: str):
    total = len(df["id"].unique())
    top_country = (
        df.groupby("bornCountry").size().reset_index(name="count").sort_values("count", ascending=False).iloc[0]
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 诺贝尔奖获得者分布简要结论\n\n")
        f.write(f"- 数据集包含约 {total} 位得主的记录。\n")
        f.write(f"- 出生国家最多的是：{top_country['bornCountry']}（{int(top_country['count'])}人）。\n")
        f.write("- 年度趋势图显示不同历史时期的获奖人数变化，可进一步结合重大事件解释。\n")
        f.write("- 学科占比（归一化面积图）反映长期结构与学科更迭，建议按时间切片细化分析。\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nobel analysis charts and draft")
    parser.add_argument("--csv", default="artifacts/nobel/laureates_prizes.csv", help="Input CSV path")
    parser.add_argument("--run-dir", default=None, help="Run directory for outputs (figures/, draft.md)")
    args = parser.parse_args()

    src = args.csv
    if args.run_dir:
        out_dir = os.path.join(args.run_dir, "figures")
        draft_path = os.path.join(args.run_dir, "draft.md")
    else:
        out_dir = "artifacts/nobel/figures"
        draft_path = "artifacts/nobel/draft.md"

    os.makedirs(out_dir, exist_ok=True)

    df = load_data(src)
    chart1 = top_countries_chart(df)
    chart2 = yearly_trend_chart(df)
    chart3 = category_stacked_chart(df)

    save_chart(chart1, f"{out_dir}/top_countries.html")
    save_chart(chart2, f"{out_dir}/yearly_trend.html")
    save_chart(chart3, f"{out_dir}/category_stacked.html")

    write_draft(df, draft_path)
    print(f"Charts saved to {out_dir} and draft.md created at {draft_path}")

if __name__ == "__main__":
    main()