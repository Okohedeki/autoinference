"""MCP server: the same service exposed as tools an agent can call.

Agents get a cost estimate before committing, submit work that runs in the
background, poll it and read ranked results, so long campaigns never block a
tool call. Run it with ``autoinference bio mcp --root <workspace>`` and point
an MCP client at that command over stdio.
"""

from __future__ import annotations

from typing import Any

from .service import BioService


def build_server(service: BioService):
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp 2.x
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
        except ImportError as exc:
            raise RuntimeError("the MCP server needs the mcp package: pip install 'autoinference[bio]'") from exc

    mcp = Server("autoinference-bio")

    @mcp.tool()
    def list_backends() -> list[dict]:
        """List the models that can score candidates, with their estimated GPU seconds per candidate."""
        return service.backends()

    @mcp.tool()
    def estimate_campaign(spec: dict) -> dict:
        """Estimate GPU time and cost for a campaign spec without running it.

        A spec has a target {id, sequence}, candidates [{id, smiles} or {id, sequence}],
        and stages [{backend, keep_top or keep_fraction, rank_by}] run in order, each on the
        survivors of the previous one. Optional: seed, consensus (backends that re-score the
        final survivors) and tuned_model (a model fitted on assay data).
        """
        return service.estimate(spec)

    @mcp.tool()
    def submit_campaign(spec: dict) -> dict:
        """Queue a campaign and return its job id and cost estimate immediately."""
        return service.submit(spec)

    @mcp.tool()
    def job_status(job_id: str) -> dict:
        """Status of a job: queued, running (with progress), done or failed."""
        return service.status(job_id)

    @mcp.tool()
    def job_results(job_id: str, top: int = 20) -> dict:
        """Ranked results of a finished job: scores per stage, pose validity checks, model
        consensus, calibrated probability of activity when a tuned model was used, and the
        run manifest."""
        data = service.results(job_id, top)
        data.pop("dropped", None)
        return data

    @mcp.tool()
    def fit_tuned_model(job_id: str, labels: dict[str, int], model_id: str | None = None) -> dict[str, Any]:
        """Fit a re-scorer on assay outcomes (candidate id -> 1 active, 0 inactive) for candidates
        scored in a finished job. Pass the returned id as tuned_model in later campaigns."""
        model = service.fit_tuned_model(job_id, labels, model_id)
        return {"id": model["id"], "metrics": model["metrics"]}

    @mcp.tool()
    def verify_job(job_id: str) -> dict:
        """Replay a finished job from its seed and confirm the output is identical."""
        return service.verify(job_id)

    return mcp


def run_stdio(service: BioService) -> None:
    build_server(service).run()
