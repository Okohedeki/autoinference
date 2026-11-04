"""The HTTP API (FastAPI). Every route is a thin call into ``Platform``.

Authentication is an API key in ``Authorization: Bearer <key>``. In local
mode, file uploads and downloads go through ``/v1/files/_local/...`` because
there is no object store to presign against.
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from .pipelines import PipelineError
from .service import Platform


def create_app(platform: Platform | None = None) -> FastAPI:
    platform = platform or Platform()
    app = FastAPI(title="AutoInference platform", version="0.1",
                  description="One API over specialised models and pipelines of them.")
    app.state.platform = platform

    def project(authorization: str = Header(default="")) -> dict[str, Any]:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing API key")
        p = platform.authenticate(authorization[len("Bearer "):].strip())
        if p is None:
            raise HTTPException(401, "invalid API key")
        return p

    @app.exception_handler(KeyError)
    async def _not_found(request: Request, exc: KeyError):
        return _error(404, "not_found", f"unknown {exc.args[0] if exc.args else 'object'}")

    @app.exception_handler(ValueError)
    async def _bad(request: Request, exc: ValueError):
        return _error(400, "invalid_request", str(exc))

    @app.exception_handler(PermissionError)
    async def _denied(request: Request, exc: PermissionError):
        return _error(402, "payment_required", str(exc))

    @app.exception_handler(FileNotFoundError)
    async def _not_ready(request: Request, exc: FileNotFoundError):
        return _error(409, "not_ready", str(exc))

    @app.get("/health")
    def health():
        return {"status": "ok", "environment": platform.settings.environment, "models": len(platform.catalog),
                "executor": platform.executor.name}

    # --- models -----------------------------------------------------------------------------------

    @app.get("/v1/models")
    def list_models(domain: str | None = None):
        models = platform.models()
        if domain:
            models = [m for m in models if m["domain"] == domain]
        return {"models": models}

    @app.get("/v1/models/{domain}/{name}")
    def get_model(domain: str, name: str):
        return platform.model(f"{domain}/{name}")

    @app.post("/v1/models/{domain}/{name}/estimate")
    def estimate(domain: str, name: str, body: dict = Body(...), p: dict = Depends(project)):
        return platform.estimate(f"{domain}/{name}", body.get("inputs", body), body.get("units"))

    @app.post("/v1/models/{domain}/{name}/run")
    def run(domain: str, name: str, body: dict = Body(...), p: dict = Depends(project),
            idempotency_key: str | None = Header(default=None)):
        st = platform.submit(p, f"{domain}/{name}", body.get("inputs", body), "run",
                             webhook_url=body.get("webhook_url"), idempotency_key=idempotency_key,
                             seed=int(body.get("seed", 0)))
        st = platform.wait(p["id"], st["request_id"], platform.settings.sync_timeout)
        if st["status"] == "succeeded":
            return platform.result(p["id"], st["request_id"])
        if st["status"] == "failed":
            return _error(500, "model_error", st["error"], request_id=st["request_id"])
        return JSONResponse({**st, "message": "still running; poll /v1/requests/{request_id}"}, status_code=202)

    @app.post("/v1/models/{domain}/{name}/queue", status_code=202)
    def queue(domain: str, name: str, body: dict = Body(...), p: dict = Depends(project),
              idempotency_key: str | None = Header(default=None)):
        return platform.submit(p, f"{domain}/{name}", body.get("inputs", body), "queue",
                               webhook_url=body.get("webhook_url"), idempotency_key=idempotency_key,
                               seed=int(body.get("seed", 0)))

    # --- requests ---------------------------------------------------------------------------------

    @app.get("/v1/requests")
    def list_requests(limit: int = Query(50, le=500), p: dict = Depends(project)):
        return {"requests": platform.list_requests(p["id"], limit)}

    @app.get("/v1/requests/{request_id}")
    def request_status(request_id: str, p: dict = Depends(project)):
        return platform.status(p["id"], request_id)

    @app.get("/v1/requests/{request_id}/result")
    def request_result(request_id: str, p: dict = Depends(project)):
        return platform.result(p["id"], request_id)

    @app.post("/v1/requests/{request_id}/cancel")
    def request_cancel(request_id: str, p: dict = Depends(project)):
        return platform.cancel(p["id"], request_id)

    # --- pipelines --------------------------------------------------------------------------------

    @app.post("/v1/pipelines/estimate")
    def pipeline_estimate(body: dict = Body(...), p: dict = Depends(project)):
        return platform.estimate_pipeline(body["pipeline"], body.get("inputs", {}))

    @app.post("/v1/pipelines/run", status_code=202)
    def pipeline_run(body: dict = Body(...), p: dict = Depends(project)):
        rerun = body.get("rerun_from")
        return platform.submit_pipeline(p, body["pipeline"], body.get("inputs", {}), body.get("webhook_url"),
                                        rerun_from=(rerun["run_id"], rerun["step"]) if rerun else None)

    @app.get("/v1/pipelines/runs/{run_id}")
    def pipeline_status(run_id: str, p: dict = Depends(project)):
        return platform.pipeline_status(p["id"], run_id)

    @app.get("/v1/pipelines/runs/{run_id}/steps/{step_id}")
    def pipeline_step(run_id: str, step_id: str, p: dict = Depends(project)):
        return platform.pipeline_step(p["id"], run_id, step_id)

    @app.post("/v1/pipelines/runs/{run_id}/steps/{step_id}/approve")
    def pipeline_approve(run_id: str, step_id: str, p: dict = Depends(project)):
        return platform.approve_step(p["id"], run_id, step_id)

    @app.post("/v1/pipelines/runs/{run_id}/cancel")
    def pipeline_cancel(run_id: str, p: dict = Depends(project)):
        return platform.cancel_pipeline(p["id"], run_id)

    # --- files ------------------------------------------------------------------------------------

    @app.post("/v1/files", status_code=201)
    def create_file(body: dict = Body(...), p: dict = Depends(project)):
        return platform.create_file(p["id"], body["filename"], body.get("content_type"), body.get("size"))

    @app.get("/v1/files/{file_id}")
    def get_file(file_id: str, p: dict = Depends(project)):
        return platform.file(p["id"], file_id)

    if platform.settings.storage == "local":
        @app.put("/v1/files/_local/{key:path}")
        async def local_upload(key: str, request: Request):
            platform.put_local_file(key, await request.body())
            return Response(status_code=204)

        @app.get("/v1/files/_local/{key:path}")
        def local_download(key: str):
            try:
                return Response(platform.read_local_file(key), media_type="application/octet-stream")
            except (FileNotFoundError, ValueError):
                raise HTTPException(404, "no such file") from None

    # --- account ----------------------------------------------------------------------------------

    @app.get("/v1/usage")
    def usage(p: dict = Depends(project)):
        return platform.usage(p["id"])

    @app.get("/v1/keys")
    def keys(p: dict = Depends(project)):
        return {"keys": platform.list_keys(p["id"])}

    @app.post("/v1/keys", status_code=201)
    def create_key(body: dict = Body(default={}), p: dict = Depends(project)):
        return platform.create_key(p["id"], body.get("name", "key"))

    @app.delete("/v1/keys/{key_id}", status_code=204)
    def revoke_key(key_id: str, p: dict = Depends(project)):
        platform.revoke_key(p["id"], key_id)
        return Response(status_code=204)

    @app.get("/v1/audit")
    def audit(limit: int = Query(200, le=2000), p: dict = Depends(project)):
        return platform.audit(p["id"], limit)

    return app


def _error(status: int, kind: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message, **extra}}, status_code=status)
