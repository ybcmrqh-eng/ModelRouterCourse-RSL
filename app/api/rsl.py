from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.core.auth import resolve_request_context
from app.core.rsl import VERIFIER_VERSION
from app.schemas.rsl import (
    RSLMemoryList,
    RSLMemoryRollbackRequest,
    RSLRunList,
    RSLRunRequest,
    RSLRunResponse,
)

router = APIRouter(prefix="/v1/rsl", tags=["rsl"])


def _engine(request: Request):
    return getattr(request.app.state, "rsl", None)


@router.get("/health")
async def rsl_health(request: Request):
    engine = _engine(request)
    return {
        "enabled": engine is not None,
        "status": "ok" if engine is not None else "disabled",
        "verifier_version": VERIFIER_VERSION if engine is not None else None,
        "typesafe_available": bool(engine is not None and engine.typesafe.available()),
    }


@router.post("/run", response_model=RSLRunResponse)
async def run_rsl(body: RSLRunRequest, request: Request):
    engine = _engine(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "RSL engine is disabled"})
    request_context = await resolve_request_context(request)
    return await engine.run(body, request_context)


@router.get("/runs", response_model=RSLRunList)
async def list_rsl_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
):
    engine = _engine(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "RSL engine is disabled"})
    await resolve_request_context(request)
    return {"runs": await engine.list_runs(limit=limit)}


@router.get("/runs/{run_id}", response_model=RSLRunResponse)
async def get_rsl_run(run_id: str, request: Request):
    engine = _engine(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "RSL engine is disabled"})
    await resolve_request_context(request)
    run = await engine.get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"error": "RSL run not found"})
    return run


@router.get("/memory", response_model=RSLMemoryList)
async def list_rsl_memory(
    request: Request,
    model: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    engine = _engine(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "RSL engine is disabled"})
    await resolve_request_context(request)
    return {"memories": await engine.list_memories(model=model, status=status, limit=limit)}


@router.post("/memory/{memory_id}/rollback")
async def rollback_rsl_memory(
    memory_id: int,
    body: RSLMemoryRollbackRequest,
    request: Request,
):
    engine = _engine(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "RSL engine is disabled"})
    await resolve_request_context(request)
    memory = await engine.rollback_memory(memory_id, reason=body.reason)
    if memory is None:
        return JSONResponse(status_code=404, content={"error": "RSL memory not found"})
    return {"memory": memory}
