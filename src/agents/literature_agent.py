from typing import Dict, Any
import os
import time
import pandas as pd

# 复用已有的数据拉取与规范化函数
from src.data.nobel_fetch import fetch_laureates, normalize

class LiteratureAgent:
    def __init__(self, base_dir: str = "artifacts/nobel"):
        self.base_dir = base_dir

    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # ... existing code ...
        # 运行 ID：优先使用 task_id，否则用当前时间戳
        run_id = str(task.get("task_id")) if task.get("task_id") else str(int(time.time()))
        out_dir = os.path.join(self.base_dir, "runs", run_id)
        os.makedirs(out_dir, exist_ok=True)

        # 拉取与标准化
        data = fetch_laureates()
        df = normalize(data)
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")

        # 落盘工件
        csv_path = os.path.join(out_dir, "laureates_prizes.csv")
        df.to_csv(csv_path, index=False)

        # 基本指标
        metrics = {
            "rows": int(len(df)),
            "laureates_unique": int(df["id"].nunique() if "id" in df.columns else 0),
            "year_missing_rate": round(float(df["year"].isna().mean()), 4),
        }

        # 返回标准结构，供 MCP 使用
        result = {
            "type": "literature_collect",
            "task_id": run_id,
            "artifacts": {
                "csv": csv_path,
                "run_dir": out_dir
            },
            "metrics": metrics,
            "notes": "source=https://api.nobelprize.org/v1/laureate.json"
        }
        return result
        # ... existing code ...