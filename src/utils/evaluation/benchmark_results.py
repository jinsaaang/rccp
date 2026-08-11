from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numbers


_CONFORMAL_METRIC_RE = re.compile(
    r"Evaluation metrics \(alpha (?P<alpha>[-+]?\d*\.?\d+)\: (?P<payload>\{.*\})"
)
_RESCP_FLOAT_RE = {
    "coverage": re.compile(r"Average test coverage:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "delta_cov": re.compile(r"Average test delta coverage:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "width": re.compile(r"Average test pi width:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "winkler": re.compile(r"Average test winkler:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}


@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    forecaster: str
    alpha: float
    coverage: Optional[float]
    delta_cov: Optional[float]
    width: Optional[float]
    winkler: Optional[float]
    winkler_norm: Optional[float]
    time_sec: Optional[float]
    wall_clock_time_sec: Optional[float] = None
    method_wall_clock_time_sec: Optional[float] = None
    calibration_time_sec: Optional[float] = None
    method_wall_clock_excluding_g_time_sec: Optional[float] = None
    method_wall_clock_including_g_time_sec: Optional[float] = None
    ct_ssf_g_fit_time_sec: Optional[float] = None
    state_save_overhead_sec: Optional[float] = None
    time_sec_without_state_save_overhead: Optional[float] = None
    wall_clock_time_sec_without_state_save_overhead: Optional[float] = None
    run_dir: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_conformal_run_dir(run_dir: Path) -> Dict[str, Any]:
    log_files = sorted(run_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No log file found in {run_dir}")
    text = log_files[-1].read_text(encoding="utf-8", errors="ignore")
    matches = list(_CONFORMAL_METRIC_RE.finditer(text))
    if not matches:
        raise ValueError(f"No evaluation metric payload found in {log_files[-1]}")
    rows = [_literal_eval_metrics(match.group("payload")) for match in matches]
    alpha = float(matches[-1].group("alpha"))
    numeric_keys = sorted(
        set().union(*(row.keys() for row in rows))
    )
    payload: Dict[str, Any] = {"alpha": alpha}
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), numbers.Number) and math.isfinite(float(row[key]))
        ]
        if values:
            payload[key] = sum(values) / len(values)
    runtime_path = run_dir / "method_runtime.json"
    if runtime_path.exists():
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        for key, value in runtime_payload.items():
            if isinstance(value, numbers.Number) and math.isfinite(float(value)):
                payload[key] = float(value)
            else:
                payload[key] = value
    return payload


def _literal_eval_metrics(payload: str) -> Dict[str, Any]:
    """Parse metric dicts that may contain Python-style nan/inf tokens."""

    def convert(node):
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Dict):
            return {convert(k): convert(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.List):
            return [convert(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(convert(item) for item in node.elts)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Name):
            name = node.id.lower()
            if name == "nan":
                return math.nan
            if name in {"inf", "infinity"}:
                return math.inf
        raise ValueError(f"Unsupported metric payload node: {ast.dump(node)}")

    parsed = ast.parse(payload, mode="eval")
    value = convert(parsed)
    if not isinstance(value, dict):
        raise ValueError(f"Expected metric payload dict, got {type(value).__name__}")
    return value


def parse_rescp_stdout(stdout: str) -> Dict[str, Optional[float]]:
    parsed = {}
    for key, pattern in _RESCP_FLOAT_RE.items():
        match = pattern.search(stdout)
        parsed[key] = float(match.group(1)) if match else None
    return parsed


def dump_result(result: BenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
