from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import Roles, require_role, require_service_secret

router = APIRouter(prefix="/infer/tables", tags=["infer-tables"])


class TableRecord(BaseModel):
    # keep generic for now; tighten later to your marks schema
    data: dict = Field(default_factory=dict)


class TableInferIn(BaseModel):
    records: list[TableRecord] = Field(..., min_length=1)


@router.post("/anomaly-detection")
def table_anomaly_detection(
    body: TableInferIn,
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.PROFESSOR, Roles.ADMIN})),
):
    # TODO: anomaly model
    return {"task": "tables.anomaly_detection", "records": len(body.records)}


@router.post("/performance-predictor")
def academic_performance_predictor(
    body: TableInferIn,
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.PROFESSOR, Roles.ADMIN})),
):
    # TODO: predictor model
    return {"task": "tables.performance_predictor", "records": len(body.records)}
