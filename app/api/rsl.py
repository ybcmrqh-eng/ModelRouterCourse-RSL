from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.core.auth import resolve_request_context
from app.schemas.rsl import RSLMemoryList, RSLRunList, RSLRunRequest, RSLRunResponse

router = APIRouter(prefix="/v1/rsl", tags=["rsl"])


def _engine(request: Request):
    engine = getattr(request.app.state, "rsl", None)
    if engine is None:
        return None
    return engine


@router.get("/health")
async def rsl_health(request: Request):
    engine = _engine(request)
    return {
        "enabled": engine is not None,
        "status": "ok" if engine is not None else "disabled",
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
    limit: int = Query(default=20, ge=1, le=100),
):
    engine = _engine(request)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "RSL engine is disabled"})
    await resolve_request_context(request)
    return {"memories": await engine.list_memories(model=model, limit=limit)}
