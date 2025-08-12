from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.eval.ragas_runner as rr
from app.eval.reporting import write_report_files
from pathlib import Path


router = APIRouter(prefix="/v1", tags=["evaluate"])


class EvalSample(BaseModel):
    question: str
    contexts: List[str] = Field(default_factory=list)
    answer: str
    ground_truths: List[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    samples: List[EvalSample]
    metrics: List[str] | None = None
    out_json: Optional[str] = None
    out_md: Optional[str] = None


@router.post("/evaluate")
def post_evaluate(req: EvalRequest) -> Dict[str, Any]:
    try:
        samples = [s.model_dump() for s in req.samples]
        result = rr.run_evaluation(samples, metrics=req.metrics)
        paths = write_report_files(
            result,
            out_json=Path(req.out_json) if req.out_json else None,
            out_md=Path(req.out_md) if req.out_md else None,
        )
        return {"result": result, "written": paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


