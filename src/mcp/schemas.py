from typing import TypedDict, Optional, Dict, List

class OrchestrateArgs(TypedDict, total=False):
    task_id: str
    theme: str
    model: str
    temperature: float
    max_tokens: int
    thresholds_path: str
    force: bool

class ArtifactMap(TypedDict, total=False):
    csv: str
    figures_dir: str
    draft_md_analysis: str
    draft_md_agent: str
    prompt_txt_agent: str
    eval_report: str
    eval_report_agent: str
    draft_published: str
    bundle_zip: str
    thresholds_path: str
    config_json_run_dir: str
    run_dir_log_json: str

class BestDraft(TypedDict, total=False):
    type: str
    path: str
    pass_count: int
    decision_rule: Optional[str]
    compare: Optional[Dict[str, int]]
    tie_breaker: Optional[str]

class Summary(TypedDict, total=False):
    ok: bool
    total_duration_s: float
    artifacts: ArtifactMap
    best_draft: BestDraft
    eval_pass: Dict[str, Optional[int]]
    eval_rubric_counts: Dict[str, Optional[Dict[str, int]]]
    eval_failed_rubric: Dict[str, List[str]]
    run_id: str
    agent_run_dir: str
    thresholds_source: str

class OrchestrateResult(TypedDict, total=False):
    summary: Summary
    errors: List[str]