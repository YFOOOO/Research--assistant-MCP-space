from typing import Dict, Any
import os
import time
import pandas as pd

class WritingAgent:
    def __init__(self, base_dir: str = "artifacts/nobel", llm=None, model: str = None):
        self.base_dir = base_dir
        self.llm = llm  # 注入式 LLM 客户端（Aisuite 适配）
        self.model = model  # 可选模型名，交由 llm 客户端使用
        self.last_llm_error = None

    def handle(self, task: dict) -> dict:
        run_id = str(task.get("task_id")) if task.get("task_id") else str(int(time.time()))
        run_dir = os.path.join(self.base_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        draft_path = os.path.join(run_dir, "draft_agent.md")

        # 读取 CSV 并计算基础指标
        csv_path = task.get("csv")
        rows = 0
        year_min = None
        year_max = None
        top_country_name = None
        top_country_count = 0
        top_category_name = None
        top_category_count = 0
        laureates_unique = 0

        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                rows = int(len(df))

                if "year" in df.columns:
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")
                    year_series = df["year"].dropna()
                    if not year_series.empty:
                        year_min = int(year_series.min())
                        year_max = int(year_series.max())

                if "id" in df.columns:
                    laureates_unique = int(df["id"].nunique())
                elif "name" in df.columns:
                    laureates_unique = int(df["name"].nunique())

                # 国家分布（优先使用常见字段）
                country_col = next((c for c in ["bornCountry", "birth_country", "country"] if c in df.columns), None)
                if country_col:
                    vc = df[country_col].value_counts(dropna=True).head(1)
                    if not vc.empty:
                        top_country_name = str(vc.index[0])
                        top_country_count = int(vc.values[0])

                # 类别分布
                if "category" in df.columns:
                    vc = df["category"].value_counts(dropna=True).head(1)
                    if not vc.empty:
                        top_category_name = str(vc.index[0])
                        top_category_count = int(vc.values[0])
            except Exception:
                # 容错：读取或计算失败时保留默认指标
                pass

        metrics = {
            "rows": rows,
            "laureates_unique": laureates_unique,
            "year_min": year_min,
            "year_max": year_max,
            "top_country_name": top_country_name,
            "top_country_count": top_country_count,
            "top_category_name": top_category_name,
            "top_category_count": top_category_count,
        }

        # 默认草稿（作为回退）
        lines = ["# 诺贝尔奖获得者分布简要结论（自动生成）", ""]
        lines.append(f"- 覆盖得主数：约 {laureates_unique} 位。" if laureates_unique else "- 覆盖得主数：数据不足。")
        if year_min is not None and year_max is not None:
            lines.append(f"- 年份范围：{year_min}–{year_max}。")
        if top_country_name:
            lines.append(f"- 出生国家最多：{top_country_name}（{top_country_count}人）。")
        if top_category_name:
            lines.append(f"- 最多奖项类别：{top_category_name}（{top_category_count}条获奖记录）。")
        lines.append("- 建议：按历史时期切片对比学科结构与地理分布，或加入机构国家口径进行双视角分析。")
        markdown = "\n".join(lines)

        # 主题与 LLM 生成（成功则覆盖默认草稿）
        theme = task.get("theme", "诺贝尔奖")
        prompt = self.build_prompt(theme, metrics)
        # 新增：保存实际使用的 Prompt 到运行目录
        prompt_path = os.path.join(run_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        llm_used = False
        llm_text = self.call_llm(prompt)
        if isinstance(llm_text, str) and llm_text.strip():
            markdown = llm_text
            llm_used = True

        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return {
            "artifacts": {
                "draft_md": draft_path,
                "run_dir": run_dir,
                "prompt_txt": prompt_path,  # 新增：返回 Prompt 路径
            },
            "metrics": metrics,
            "llm": {
                "used": llm_used,
                "theme": theme,
                "model": self.model or (getattr(self.llm, "default_model", None) if self.llm else None),
                "error": self.last_llm_error,
            },
        }

    def build_prompt(self, theme: str, metrics: Dict[str, Any]) -> str:
        title = f"{theme} 数据洞察草稿"
        lines = [
            f"# {title}",
            "请基于以下结构生成一份简短、结构化的中文草稿：",
            "1. 背景与数据来源（1段）",
            "2. 关键发现（要点列表，3-6条）",
            "3. 趋势与分布（1-2段）",
            "4. 结论与建议（1段）",
            "",
            "可参考的量化指标：",
            f"- 数据行数: {metrics.get('rows')}",
            f"- 年份范围: {metrics.get('year_min')}–{metrics.get('year_max')}",
            f"- Top 国家: {metrics.get('top_country_name')} ({metrics.get('top_country_count')})",
            f"- 唯一获奖者数: {metrics.get('laureates_unique')}",
        ]
        return "\n".join(lines)

    def call_llm(self, prompt: str) -> Any:
        if not self.llm:
            self.last_llm_error = "LLM client not injected"
            return None
        try:
            # 统一适配：约定 llm.generate(prompt, model=..., **kwargs) -> str
            self.last_llm_error = None
            result = self.llm.generate(prompt, model=self.model)
            # 新增：若生成为空，透出底层适配器错误
            if not (isinstance(result, str) and result.strip()):
                le = getattr(self.llm, "last_error", None)
                if le:
                    self.last_llm_error = le
            return result
        except Exception as e:
            self.last_llm_error = str(e)
            return None