import os
import sys
import time
import json
import re
import subprocess
from pathlib import Path
import argparse

# 修复 Python 路径：确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
from src.agents.literature_agent import LiteratureAgent
from src.agents.writing_agent import WritingAgent
from dotenv import load_dotenv

def get_llm_client_from_env():
    import os
    import sys
    from pathlib import Path

    # 优先从环境变量注入本地 Aisuite 路径
    aisuite_path = os.environ.get("AISUITE_PATH")
    if aisuite_path:
        p = Path(aisuite_path).resolve()
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        import aisuite as ai
    except Exception:
        return None

    client = ai.Client()

    # 选择模型：优先使用 NOBEL_LLM_MODEL；否则根据已配置的 API Key 推断默认 provider:model
    default_model = os.environ.get("NOBEL_LLM_MODEL")
    if not default_model:
        if os.environ.get("DASHSCOPE_API_KEY"):
            default_model = "dashscope:qwen3-max"      # Qwen（DashScope）
        elif os.environ.get("DEEPSEEK_API_KEY"):
            default_model = "deepseek:deepseek-chat"   # DeepSeek

    class AisuiteAdapter:
        def __init__(self, client, default_model=None):
            self.client = client
            self.default_model = default_model
            # 读取温度配置（默认 0.2）
            try:
                self.temperature = float(os.environ.get("NOBEL_LLM_TEMPERATURE", "0.2"))
            except Exception:
                self.temperature = 0.2
            # 新增：读取最大生成长度（可选）
            try:
                mt = os.environ.get("NOBEL_LLM_MAX_TOKENS")
                self.max_tokens = int(mt) if mt is not None else None
            except Exception:
                self.max_tokens = None
            # 新增：保存最近一次错误信息
            self.last_error = None

        def generate(self, prompt: str, model: str = None) -> str:
            use_model = model or self.default_model
            if not use_model:
                return ""
            messages = [{"role": "user", "content": prompt}]
            try:
                # 成功调用前清空错误
                self.last_error = None
                # 新增：仅在配置存在时传入 max_tokens
                kwargs = {
                    "model": use_model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                resp = self.client.chat.completions.create(**kwargs)
                choices = getattr(resp, "choices", [])
                if choices:
                    msg = getattr(choices[0], "message", None)
                    content = getattr(msg, "content", None) if msg else None
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = []
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                parts.append(c["text"])
                        if parts:
                            return "\n".join(parts)
                return ""
            except Exception as e:
                # 新增：记录错误并返回空字符串
                self.last_error = str(e)
                return ""

    return AisuiteAdapter(client, default_model)

def _print_run_summary(run_log: dict) -> None:
    print("=== Run Summary ===")
    summary = run_log.get("summary", {})
    write = run_log.get("write", {})

    status = "OK" if summary.get("ok") else "FAILED"
    print(f"Status: {status}")
    print(f"Total duration: {summary.get('total_duration_s', 'n/a')}s")

    print(f"LLM used: {write.get('llm_used', False)} (enabled={write.get('llm_enabled', False)})")
    if write.get("llm_model"):
        provider = write.get("llm_provider") or "n/a"
        model_source = summary.get("write_llm_model_source") or write.get("llm_model_source")
        temperature = write.get("llm_temperature")
        temp_source = summary.get("write_llm_temperature_source") or write.get("llm_temperature_source")
        max_tokens = summary.get("write_llm_max_tokens") or write.get("llm_max_tokens")
        max_tokens_source = summary.get("write_llm_max_tokens_source") or write.get("llm_max_tokens_source")
        print(f"Provider: {provider}")
        print(f"Model: {write.get('llm_model')}")
        if model_source:
            print(f"Model source: {model_source}")
        if temperature is not None:
            print(f"Temperature: {temperature}")
        if temp_source:
            print(f"Temperature source: {temp_source}")
        # 修正：打印时保留写阶段的回退，不再重置为仅 summary
        if max_tokens is not None:
            print(f"Max tokens: {max_tokens}")
        else:
            print("Max tokens: (not set)")
        if max_tokens_source:
            print(f"Max tokens source: {max_tokens_source}")
        # 新增：打印主题与来源
        theme = summary.get("write_theme") or write.get("theme")
        theme_source = summary.get("write_theme_source") or write.get("theme_source")
        if theme:
            print(f"Theme: {theme}")
        if theme_source:
            print(f"Theme source: {theme_source}")
    if write.get("llm_error"):
        print(f"LLM error: {write.get('llm_error')}")

    artifacts = summary.get("artifacts", {})
    prompt_path = artifacts.get("prompt_txt_agent")
    if prompt_path:
        print(f"Prompt: {prompt_path}")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if first_line:
                print(f"Prompt title: {first_line}")
        except Exception:
            pass
    draft_path = artifacts.get("draft_published") or artifacts.get("draft_md_agent")
    if draft_path:
        print(f"Published draft: {draft_path}")
    bundle = artifacts.get("bundle_zip")
    if bundle:
        print(f"Bundle: {bundle}")
    # 新增：打印最佳草稿来源与评估通过数
    best = summary.get("best_draft", {}) or {}
    best_type = best.get("type")
    best_pass = best.get("pass_count")
    best_path = best.get("path")
    if best_type:
        print(f"Best draft: {best_type} (pass_count={best_pass})")
    if best_path:
        print(f"Best draft path: {best_path}")
    # 新增：打印选择依据与对比
    rule = best.get("decision_rule")
    compare = best.get("compare")
    tie = best.get("tie_breaker")
    if rule:
        print(f"Selection rule: {rule}")
    if compare:
        print(f"Compare: analysis={compare.get('analysis')}, agent={compare.get('agent')}")
    if tie:
        print(f"Tie breaker: {tie}")

    # 新增：打印两条路线的评估通过数
    eval_pass = summary.get("eval_pass", {}) or {}
    ap = eval_pass.get("analysis")
    gp = eval_pass.get("agent")
    if (ap is not None) or (gp is not None):
        print(f"Eval pass counts: analysis={ap}, agent={gp}")
    # 新增：打印 rubric 细目计数（通过/失败）
    rubric_counts = summary.get("eval_rubric_counts", {}) or {}
    rc_a = rubric_counts.get("analysis") or {}
    rc_g = rubric_counts.get("agent") or {}
    if rc_a or rc_g:
        print(
            f"Rubric counts: analysis(pass={rc_a.get('pass')}, fail={rc_a.get('fail')}), "
            f"agent(pass={rc_g.get('pass')}, fail={rc_g.get('fail')})"
        )
    # 新增：打印失败的 rubric 项（最多 3 个）
    failed = summary.get("eval_failed_rubric", {}) or {}
    fa = failed.get("analysis") or []
    fg = failed.get("agent") or []
    if fa or fg:
        print(f"Failed rubric: analysis=[{', '.join(fa)}], agent=[{', '.join(fg)}]")
    # 新增：打印评估阈值来源与路径
    thr_path = artifacts.get("thresholds_path")
    thr_source = summary.get("thresholds_source")
    if thr_source:
        print(f"Eval thresholds source: {thr_source}")
    if thr_path:
        print(f"Eval thresholds path: {thr_path}")
    # 新增：打印配置快照路径
    config_snapshot = artifacts.get("config_json_run_dir")
    if config_snapshot:
        print(f"Config snapshot: {config_snapshot}")
    print("====================")

# 新增：将运行摘要持久化到 run_dir/summary.txt
def _save_run_summary_text(run_log: dict) -> None:
    summary = run_log.get("summary", {})
    write = run_log.get("write", {})
    run_dir = summary.get("agent_run_dir")
    if not run_dir:
        return

    lines = []
    lines.append("=== Run Summary ===")
    lines.append(f"Status: {'OK' if summary.get('ok') else 'FAILED'}")
    lines.append(f"Total duration: {summary.get('total_duration_s', 'n/a')}s")
    lines.append(f"LLM used: {write.get('llm_used', False)} (enabled={write.get('llm_enabled', False)})")

    if write.get("llm_model"):
        provider = write.get("llm_provider") or "n/a"
        model_source = summary.get("write_llm_model_source") or write.get("llm_model_source")
        temperature = write.get("llm_temperature")
        temp_source = summary.get("write_llm_temperature_source") or write.get("llm_temperature_source")
        max_tokens = summary.get("write_llm_max_tokens") or write.get("llm_max_tokens")
        max_tokens_source = summary.get("write_llm_max_tokens_source") or write.get("llm_max_tokens_source")
        lines.append(f"Provider: {provider}")
        lines.append(f"Model: {write.get('llm_model')}")
        if model_source:
            lines.append(f"Model source: {model_source}")
        if temperature is not None:
            lines.append(f"Temperature: {temperature}")
        if temp_source:
            lines.append(f"Temperature source: {temp_source}")
        lines.append(f"Max tokens: {max_tokens if max_tokens is not None else '(not set)'}")
        if max_tokens_source:
            lines.append(f"Max tokens source: {max_tokens_source}")

    artifacts = summary.get("artifacts", {})
    prompt_path = artifacts.get("prompt_txt_agent")
    if prompt_path:
        lines.append(f"Prompt: {prompt_path}")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if first_line:
                lines.append(f"Prompt title: {first_line}")
        except Exception:
            pass

    draft_path = artifacts.get("draft_published") or artifacts.get("draft_md_agent")
    if draft_path:
        lines.append(f"Published draft: {draft_path}")
    bundle = artifacts.get("bundle_zip")
    if bundle:
        lines.append(f"Bundle: {bundle}")

    best = summary.get("best_draft", {}) or {}
    best_type = best.get("type")
    best_pass = best.get("pass_count")
    best_path = best.get("path")
    if best_type:
        lines.append(f"Best draft: {best_type} (pass_count={best_pass})")
    if best_path:
        lines.append(f"Best draft path: {best_path}")
    # 新增：打印选择依据与对比
    rule = best.get("decision_rule")
    compare = best.get("compare")
    tie = best.get("tie_breaker")
    if rule:
        lines.append(f"Selection rule: {rule}")
    if compare:
        lines.append(f"Compare: analysis={compare.get('analysis')}, agent={compare.get('agent')}")
    if tie:
        lines.append(f"Tie breaker: {tie}")
    # 新增：打印评估阈值来源与路径
    thr_path = artifacts.get("thresholds_path")
    thr_source = summary.get("thresholds_source")
    if thr_source:
        lines.append(f"Eval thresholds source: {thr_source}")
    if thr_path:
        lines.append(f"Eval thresholds path: {thr_path}")
    # 新增：打印配置快照路径
    config_snapshot = artifacts.get("config_json_run_dir")
    if config_snapshot:
        lines.append(f"Config snapshot: {config_snapshot}")
    lines.append("====================")
    out_path = os.path.join(run_dir, "summary.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        # 把路径加入日志工件
        run_log.setdefault("summary", {}).setdefault("artifacts", {})["summary_txt_run_dir"] = out_path
    except Exception:
        pass

def _save_run_config_snapshot(run_log: dict) -> str | None:
    # 保存本次运行的关键配置快照到 run_dir/config.json，便于复现与审计
    summary = run_log.get("summary", {})
    write = run_log.get("write", {})
    try:
        # 更健壮地查找 run_dir
        run_dir = (
            summary.get("agent_run_dir")
            or summary.get("artifacts", {}).get("run_dir")
            or write.get("artifacts", {}).get("run_dir")
            or run_log.get("fetch", {}).get("run_dir")
        )
        if not run_dir:
            return None

        # 从日志与环境拼出配置快照（不包含敏感值）
        snapshot = {
            "model": write.get("llm_model") or summary.get("write_llm_model"),
            "model_source": write.get("llm_model_source") or summary.get("write_llm_model_source"),
            "max_tokens": write.get("llm_max_tokens") or summary.get("write_llm_max_tokens"),
            "max_tokens_source": write.get("llm_max_tokens_source") or summary.get("write_llm_max_tokens_source"),
            "temperature": write.get("llm_temperature") or summary.get("write_llm_temperature"),
            "temperature_source": write.get("llm_temperature_source") or summary.get("write_llm_temperature_source"),
            "theme": summary.get("write_theme") or write.get("theme") or summary.get("theme"),
            "theme_source": summary.get("write_theme_source") or write.get("theme_source") or summary.get("theme_source"),
            # 新增：评估阈值来源与路径
            "eval_thresholds_source": summary.get("thresholds_source"),
            "eval_thresholds_path": summary.get("artifacts", {}).get("thresholds_path"),
            "env": {
                "NOBEL_MODEL": os.getenv("NOBEL_MODEL"),
                "NOBEL_LLM_MAX_TOKENS": os.getenv("NOBEL_LLM_MAX_TOKENS"),
                "NOBEL_TEMPERATURE": os.getenv("NOBEL_TEMPERATURE"),
                "NOBEL_THEME": os.getenv("NOBEL_THEME"),
            },
            "secrets_present": {
                "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
                "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
                "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
            },
        }

        path = os.path.join(run_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        run_log.setdefault("summary", {}).setdefault("artifacts", {})
        run_log["summary"]["artifacts"]["config_json_run_dir"] = path
        return path
    except Exception as e:
        print(f"Warn: failed to save run config snapshot: {e}")
        return None


def run_nobel_pipeline() -> str:
    os.makedirs("artifacts/nobel", exist_ok=True)
    py = sys.executable
    run_log = {}
    start = time.time()
    load_dotenv()
    force = os.environ.get("NOBEL_FORCE_RECOMPUTE") == "1"
    # Step 1: Fetch data via LiteratureAgent
    t0 = time.time()
    agent = LiteratureAgent(base_dir="artifacts/nobel")
    task_id = str(int(time.time()))
    agent_result = agent.handle({"task_id": task_id})
    src_csv = agent_result["artifacts"]["csv"]
    default_csv = "artifacts/nobel/laureates_prizes.csv"
    shutil.copyfile(src_csv, default_csv)

    run_log["fetch"] = {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "duration_s": round(time.time() - t0, 2),
        "rows": agent_result.get("metrics", {}).get("rows"),
        "run_dir": agent_result["artifacts"]["run_dir"],
        "csv_src": src_csv,
        "csv_default": default_csv,
    }

    # Step 2: Analysis — cache-aware
    t1 = time.time()
    run_dir = agent_result["artifacts"]["run_dir"]
    figures_dir = os.path.join(run_dir, "figures")
    draft_analysis_path = os.path.join(run_dir, "draft.md")
    need_analysis = force or not (
        os.path.isdir(figures_dir) and
        os.path.exists(os.path.join(figures_dir, "top_countries.html")) and
        os.path.exists(os.path.join(figures_dir, "yearly_trend.html")) and
        os.path.exists(os.path.join(figures_dir, "category_stacked.html")) and
        os.path.exists(draft_analysis_path)
    )
    if need_analysis:
        p2 = subprocess.run(
            [py, "src/analysis/nobel_analysis.py", "--csv", src_csv, "--run-dir", run_dir],
            capture_output=True, text=True,
        )
        analysis_rc = p2.returncode
        analysis_out = p2.stdout
        analysis_err = p2.stderr
    else:
        analysis_rc = 0
        analysis_out = "[cache] analysis skipped (artifacts found)"
        analysis_err = ""
    run_log["analysis"] = {
        "returncode": analysis_rc,
        "stdout": analysis_out,
        "stderr": analysis_err,
        "duration_s": round(time.time() - t1, 2),
        "cache_hit": not need_analysis and not force,
    }

    # Step 3: Eval — cache-aware for analysis eval
    t2 = time.time()
    thresholds_path = os.environ.get("NOBEL_EVAL_THRESHOLDS_PATH", "artifacts/nobel/eval_thresholds.json")
    expected_eval_tag = "eval"  # analysis draft
    expected_eval_json = None
    if os.path.exists(draft_analysis_path):
        # 搜索最近的 eval_* 文件（简单近似：存在则认为可用）
        # 也可精细匹配文件名规则；此处先保守处理
        pass
    need_eval_analysis = True if force else True  # 保守：先不命中缓存
    cmd3 = [py, "src/eval/nobel_eval.py", "--csv", src_csv, "--run-dir", run_dir]
    if os.path.exists(thresholds_path):
        cmd3.extend(["--thresholds", thresholds_path])
    if need_eval_analysis:
        p3 = subprocess.run(cmd3, capture_output=True, text=True)
        eval_report_path = None
        m_eval = re.search(r"Report saved -> (.+)", p3.stdout or "")
        if m_eval: eval_report_path = m_eval.group(1).strip()
        eval_rc = p3.returncode
        eval_out = p3.stdout
        eval_err = p3.stderr
    else:
        eval_report_path = expected_eval_json
        eval_rc = 0
        eval_out = "[cache] eval (analysis) skipped"
        eval_err = ""
    run_log["eval"] = {
        "returncode": eval_rc,
        "stdout": eval_out,
        "stderr": eval_err,
        "duration_s": round(time.time() - t2, 2),
        "report_path": eval_report_path,
        "thresholds_path": thresholds_path if os.path.exists(thresholds_path) else "",
        "cache_hit": not need_eval_analysis and not force,
    }

    # Step 4: Writing — cache-aware for agent draft
    t3 = time.time()
    draft_agent_path = os.path.join(run_dir, "draft_agent.md")
    need_write = force or not os.path.exists(draft_agent_path)
    llm_client = get_llm_client_from_env()
    llm_model = os.environ.get("NOBEL_LLM_MODEL")
    writer = WritingAgent(base_dir="artifacts/nobel", llm=llm_client, model=llm_model)
    theme = os.environ.get("NOBEL_THEME", "诺贝尔奖")
    # 新增：记录主题来源（env/default）
    theme_source = "env" if "NOBEL_THEME" in os.environ else "default"
    write_result = writer.handle({"task_id": task_id, "csv": src_csv, "theme": theme})

    # 计算实际使用的模型（优先 agent 返回，其次 env，最后适配器默认）
    resolved_model = (
        write_result.get("llm", {}).get("model")
        or llm_model
        or getattr(llm_client, "default_model", None)
    )
    # 新增：记录模型来源（agent/env/adapter_default）
    if write_result.get("llm", {}).get("model"):
        resolved_model_source = "agent"
    elif llm_model:
        resolved_model_source = "env"
    elif getattr(llm_client, "default_model", None):
        resolved_model_source = "adapter_default"
    else:
        resolved_model_source = None

    # 解析 provider
    resolved_provider = None
    if isinstance(resolved_model, str) and ":" in resolved_model:
        resolved_provider = resolved_model.split(":", 1)[0].strip()
    # 记录 LLM 温度（来自适配器）
    resolved_temperature = getattr(llm_client, "temperature", None)
    # 温度来源：env 优先，其次适配器默认
    env_temperature_raw = os.environ.get("NOBEL_LLM_TEMPERATURE")
    resolved_temperature_source = (
        "env" if env_temperature_raw is not None
        else ("adapter_default" if resolved_temperature is not None else None)
    )

    # 新增：max_tokens 优先采用环境变量，其次适配器默认，并记录来源
    env_max_tokens_raw = os.environ.get("NOBEL_LLM_MAX_TOKENS")
    resolved_max_tokens = None
    resolved_max_tokens_source = None
    if env_max_tokens_raw is not None:
        try:
            resolved_max_tokens = int(env_max_tokens_raw)
            resolved_max_tokens_source = "env"
        except ValueError:
            resolved_max_tokens = getattr(llm_client, "max_tokens", None)
            resolved_max_tokens_source = "env_invalid"
    else:
        resolved_max_tokens = getattr(llm_client, "max_tokens", None)
        if resolved_max_tokens is not None:
            resolved_max_tokens_source = "adapter_default"

    run_log["write"] = {
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "duration_s": round(time.time() - t3, 2),
        "draft_md": write_result["artifacts"]["draft_md"],
        "metrics": write_result.get("metrics", {}),
        "theme": theme,
        "theme_source": theme_source,  # 新增
        "llm_used": write_result.get("llm", {}).get("used", False),
        "llm_enabled": bool(llm_client),
        "llm_model": resolved_model,
        "llm_model_source": resolved_model_source,  # 新增
        "llm_provider": resolved_provider,
        "llm_error": write_result.get("llm", {}).get("error"),
        "llm_temperature": resolved_temperature,
        "llm_temperature_source": resolved_temperature_source,
        "llm_max_tokens": resolved_max_tokens,
        "llm_max_tokens_source": resolved_max_tokens_source,
    }
    # 新增：同步到摘要，便于控制台正确打印
    run_log.setdefault("summary", {})
    run_log["summary"]["write_llm_max_tokens"] = run_log["write"].get("llm_max_tokens")
    run_log["summary"]["write_llm_max_tokens_source"] = run_log["write"].get("llm_max_tokens_source")
    run_log["summary"]["write_theme"] = run_log["write"]["theme"]
    run_log["summary"]["write_theme_source"] = run_log["write"]["theme_source"]  # 新增

    # Step 5: Evaluation — agent draft
    t4 = time.time()
    draft_agent_path = write_result["artifacts"]["draft_md"]
    cmd4 = [
        sys.executable,
        "src/eval/nobel_eval.py",
        "--csv", src_csv,
        "--run-dir", agent_result["artifacts"]["run_dir"],
        "--draft", draft_agent_path,
    ]
    if os.path.exists(thresholds_path):
        cmd4.extend(["--thresholds", thresholds_path])
    p4 = subprocess.run(
        cmd4,
        capture_output=True,
        text=True,
    )
    m2 = re.search(r"Report saved -> (.+\.json)", p4.stdout or "")
    eval_agent_report_path = m2.group(1) if m2 else ""
    run_log["eval_agent"] = {
        "returncode": p4.returncode,
        "stdout": p4.stdout,
        "stderr": p4.stderr,
        "duration_s": round(time.time() - t4, 2),
        "report_path": eval_agent_report_path,
        "thresholds_path": thresholds_path if os.path.exists(thresholds_path) else "",
    }

    # Compare evaluation reports to select best draft
    def _load_json(path: str):
        try:
            if path:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _pass_count(report: dict):
        if not report or "rubric" not in report:
            return None
        rubric = report["rubric"]
        return sum(1 for v in rubric.values() if isinstance(v, bool) and v)

    analysis_report = _load_json(eval_report_path)
    agent_report = _load_json(eval_agent_report_path)
    analysis_pass = _pass_count(analysis_report)
    agent_pass = _pass_count(agent_report)

    # 新增：统计 rubric 的通过/失败计数
    def _rubric_counts(report: dict):
        if not report or "rubric" not in report:
            return None
        rubric = report["rubric"]
        bool_items = [v for v in rubric.values() if isinstance(v, bool)]
        passed = sum(1 for v in bool_items if v)
        failed = len(bool_items) - passed
        return {"pass": passed, "fail": failed}
    analysis_counts = _rubric_counts(analysis_report)
    agent_counts = _rubric_counts(agent_report)

    # 新增：提取失败的 rubric 名称（最多 3 个）
    def _failed_rubrics(report: dict):
        if not report or "rubric" not in report:
            return None
        items = [(k, v) for k, v in report["rubric"].items() if isinstance(v, bool)]
        failed = [k for k, v in items if v is False]
        return failed[:3] if failed else []

    analysis_failed = _failed_rubrics(analysis_report)
    agent_failed = _failed_rubrics(agent_report)
    best_draft = {}
    if (analysis_pass is not None) or (agent_pass is not None):
        if (agent_pass or -1) > (analysis_pass or -1):
            best_draft = {
                "type": "agent",
                "path": draft_agent_path,
                "pass_count": agent_pass,
            }
        else:
            best_draft = {
                "type": "analysis",
                "path": os.path.join(agent_result["artifacts"]["run_dir"], "draft.md"),
                "pass_count": analysis_pass,
            }

    # Step 6: Publish — copy best draft to canonical path
    t5 = time.time()
    published_draft_path = "artifacts/nobel/draft.md"
    published_source_path = best_draft.get("path")
    published_source_type = best_draft.get("type")
    if published_source_path:
        shutil.copyfile(published_source_path, published_draft_path)
        run_log["publish"] = {
            "returncode": 0,
            "source_type": published_source_type,
            "source_path": published_source_path,
            "target_path": published_draft_path,
            "duration_s": round(time.time() - t5, 2),
        }
    else:
        run_log["publish"] = {
            "returncode": 1,
            "error": "No best draft available to publish.",
            "duration_s": round(time.time() - t5, 2),
        }

    # Step 7: Bundle — zip run_dir for archival
    t6 = time.time()
    base_name = os.path.join("artifacts/nobel", f"run_{task_id}")
    try:
        bundle_path = shutil.make_archive(base_name, "zip", root_dir=agent_result["artifacts"]["run_dir"])
        run_log["bundle"] = {
            "returncode": 0,
            "path": bundle_path,
            "root_dir": agent_result["artifacts"]["run_dir"],
            "duration_s": round(time.time() - t6, 2),
        }
    except Exception as e:
        run_log["bundle"] = {
            "returncode": 1,
            "error": str(e),
            "root_dir": agent_result["artifacts"]["run_dir"],
            "duration_s": round(time.time() - t6, 2),
        }

    ok = (run_log["fetch"]["returncode"] == 0) and (p2.returncode == 0) and (p3.returncode == 0) and (p4.returncode == 0)
    run_log["summary"] = {
        "ok": ok,
        "total_duration_s": round(time.time() - start, 2),
        "artifacts": {
            "csv": src_csv,
            "figures_dir": os.path.join(agent_result["artifacts"]["run_dir"], "figures"),
            "draft_md_analysis": os.path.join(agent_result["artifacts"]["run_dir"], "draft.md"),
            "draft_md_agent": draft_agent_path,
            "prompt_txt_agent": write_result["artifacts"].get("prompt_txt", ""),
            "eval_report": eval_report_path,
            "eval_report_agent": eval_agent_report_path,
            "draft_published": published_draft_path if published_source_path else "",
            "bundle_zip": bundle_path if run_log.get("bundle", {}).get("returncode") == 0 else "",
            "thresholds_path": thresholds_path if os.path.exists(thresholds_path) else "",
        },
        "best_draft": best_draft,
        "eval_pass": {"analysis": analysis_pass, "agent": agent_pass},
        "eval_rubric_counts": {"analysis": analysis_counts, "agent": agent_counts},
        # 新增：记录失败的 rubric 项名（最多 3 个）
        "eval_failed_rubric": {"analysis": analysis_failed, "agent": agent_failed},
        "run_id": task_id,
        "agent_run_dir": agent_result["artifacts"]["run_dir"],
        "thresholds_source": "file" if os.path.exists(thresholds_path) else "default",
    }

    # 新增：记录环境来源，便于审计与复现
    run_log["summary"]["env"] = {
        "aisuite_path_env": os.environ.get("AISUITE_PATH") or "",
        "aisuite_module_file": (sys.modules.get("aisuite").__file__
                                if "aisuite" in sys.modules and hasattr(sys.modules.get("aisuite"), "__file__")
                                else ""),
        "nobel_llm_temperature_env": os.environ.get("NOBEL_LLM_TEMPERATURE") or "",
        "nobel_llm_model_env": os.environ.get("NOBEL_LLM_MODEL") or "",
        "nobel_theme_env": os.environ.get("NOBEL_THEME") or "",
    }

    ts = int(time.time())
    out_path = f"artifacts/nobel/run_log_{ts}.json"
    # 修复：先定义 run_dir_log_path 再使用
    run_dir_log_path = os.path.join(agent_result["artifacts"]["run_dir"], "run_log.json")
    # 确保 artifacts 字典存在
    run_log.setdefault("summary", {}).setdefault("artifacts", {})
    run_log["summary"]["artifacts"]["run_log_global"] = out_path
    run_log["summary"]["artifacts"]["run_log_run_dir"] = run_dir_log_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    with open(run_dir_log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    print(f"Run complete. Log saved -> {out_path} (and {run_dir_log_path})")

    # 已有：打印运行摘要并保存 summary.txt
    _print_run_summary(run_log)
    _save_run_summary_text(run_log)

    # 新增：保存运行配置快照并打印路径
    config_snapshot_path = _save_run_config_snapshot(run_log)
    if config_snapshot_path:
        print(f"Config snapshot saved -> {config_snapshot_path}")

    return out_path

if __name__ == "__main__":  # CLI entry
    parser = argparse.ArgumentParser(description="Run Nobel orchestrator")
    parser.add_argument("--theme", type=str, help="Override write theme")
    parser.add_argument("--model", type=str, help="Override LLM model in provider:model format")
    parser.add_argument("--temperature", type=float, help="Override LLM sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="Override LLM max generation tokens")  # 新增
    # 新增：允许覆盖评估阈值文件路径
    parser.add_argument("--thresholds", type=str, help="Override eval thresholds JSON path")
    # 新增：强制重算开关
    parser.add_argument("--force", action="store_true", help="Force recompute all steps, ignoring cache")
    args = parser.parse_args()

    if args.theme:
        os.environ["NOBEL_THEME"] = args.theme
    if args.model:
        # 新增：禁止占位符写法，要求传入真实 provider:model
        if "<" in args.model or ">" in args.model:
            print("Error: --model expects provider:model like 'dashscope:qwen3-max'. Do not use angle brackets.")
            sys.exit(2)
        os.environ["NOBEL_LLM_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["NOBEL_LLM_TEMPERATURE"] = str(args.temperature)
    if args.max_tokens is not None:
        os.environ["NOBEL_LLM_MAX_TOKENS"] = str(args.max_tokens)  # 新增
    # 新增：写入评估阈值路径到环境变量
    if args.thresholds:
        os.environ["NOBEL_EVAL_THRESHOLDS_PATH"] = args.thresholds
    # 新增：环境开关
    if args.force:
        os.environ["NOBEL_FORCE_RECOMPUTE"] = "1"

    run_nobel_pipeline()