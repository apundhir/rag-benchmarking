from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def generate_markdown_report(result: Dict[str, Any], title: str = "RAG Evaluation Report") -> str:
    metrics: Dict[str, float] = result.get("metrics", {})  # type: ignore[assignment]
    lines = [f"# {title}", "", "## Metrics", "", "| Metric | Score |", "|---|---:|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value:.3f} |")
    lines.append("")
    return "\n".join(lines)


def write_report_files(
    result: Dict[str, Any], out_json: Optional[Path] = None, out_md: Optional[Path] = None
) -> Dict[str, Optional[str]]:
    written: Dict[str, Optional[str]] = {"json": None, "md": None}
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        written["json"] = str(out_json)
    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = generate_markdown_report(result)
        out_md.write_text(md, encoding="utf-8")
        written["md"] = str(out_md)
    return written


