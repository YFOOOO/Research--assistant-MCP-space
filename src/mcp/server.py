# 模块导入区（修复 No module named 'src'）
import os
import sys
from typing import Dict, Any

# 将项目根目录加入 sys.path，确保可以按包名 `src.mcp` 导入
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.mcp.orchestrator import run_nobel_pipeline
except ModuleNotFoundError:
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from mcp.orchestrator import run_nobel_pipeline

# 新增：本地 JSON 读取函数，替代从 orchestrator 导入
import json
def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def orchestrate_all(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    最小 MCP 入口：透传 CLI 级覆盖到环境变量，调用 orchestrator，并返回结构化结果。
    后续可替换为正式 MCP Server（协议握手/工具注册）。
    """
    # 透传覆盖（约定与 CLI 一致）
    if "theme" in args:
        os.environ["NOBEL_THEME"] = str(args["theme"])
    if "model" in args:
        os.environ["NOBEL_LLM_MODEL"] = str(args["model"])
    if "temperature" in args and args["temperature"] is not None:
        os.environ["NOBEL_LLM_TEMPERATURE"] = str(args["temperature"])
    if "max_tokens" in args and args["max_tokens"] is not None:
        os.environ["NOBEL_LLM_MAX_TOKENS"] = str(args["max_tokens"])
    if "thresholds_path" in args and args["thresholds_path"]:
        os.environ["NOBEL_EVAL_THRESHOLDS_PATH"] = str(args["thresholds_path"])
    # 新增：强制重算开关透传
    if args.get("force"):
        os.environ["NOBEL_FORCE_RECOMPUTE"] = "1"

    # 调用 orchestrator
    run_log_path = run_nobel_pipeline()

    # 加载并裁剪返回值（summary + artifacts）
    run_log = _load_json(run_log_path)
    summary = run_log.get("summary", {}) or {}
    artifacts = summary.get("artifacts", {}) or {}
    errors = []

    return {
        "schema_version": "1.0",
        "summary": summary,
        "artifacts": artifacts,
        "errors": errors,
    }

if __name__ == "__main__":
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description="MCP orchestrate_all CLI wrapper")
    parser.add_argument("--theme", type=str, help="Override write theme")
    parser.add_argument("--model", type=str, help="Override LLM model in provider:model format")
    parser.add_argument("--temperature", type=float, help="Override LLM sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="Override LLM max generation tokens")
    parser.add_argument("--thresholds", type=str, help="Override eval thresholds JSON path")
    parser.add_argument("--force", action="store_true", help="Force recompute all steps, ignoring cache")
    args = parser.parse_args()

    payload = {}
    if args.theme is not None:
        payload["theme"] = args.theme
    if args.model is not None:
        payload["model"] = args.model
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.thresholds is not None:
        payload["thresholds_path"] = args.thresholds
    if args.force:
        payload["force"] = True

    result = orchestrate_all(payload)
    summary = result.get("summary", {}) or {}
    artifacts = result.get("artifacts", {}) or {}
    ok = summary.get("ok")
    run_id = summary.get("run_id")
    run_dir = summary.get("agent_run_dir")
    print(f"MCP orchestrate_all: ok={ok}, run_id={run_id}, run_dir={run_dir}")
    # 新增：打印关键工件路径，便于外部客户端直接消费
    published = artifacts.get("draft_published") or artifacts.get("draft_md_agent")
    if published:
        print(f"Draft path: {published}")

    ts = int(time.time())
    out_path = f"artifacts/nobel/mcp_result_{ts}.json"
    os.makedirs("artifacts/nobel", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Result saved -> {out_path}")