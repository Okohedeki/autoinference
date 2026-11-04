"""The bio service: funnels, trust checks, tuning, reproducibility, the HTTP API and the MCP tools.

Everything runs on the synthetic backends, so no GPU or model weights are needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from autoinference.bio import BioService, CampaignSpec, estimate, run_campaign
from autoinference.bio.backends import _boltz_yaml, _pdb_atom, hidden_activity
from autoinference.bio.provenance import AuditLog
from autoinference.bio.service import make_handler
from autoinference.bio.trust import IsotonicCalibrator, check_pose, rank_agreement
from autoinference.bio.types import Candidate, Target

TARGET = {"id": "tgt", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"}


def spec_dict(n: int = 200, seed: int = 0, **extra) -> dict:
    return {
        "name": "test",
        "target": TARGET,
        "candidates": [{"id": f"c{i:03d}", "smiles": "C" * (1 + i % 7) + "O" * (i % 3) + f"N{i}"} for i in range(n)],
        "stages": [
            {"backend": "synthetic-screen", "keep_fraction": 0.1},
            {"backend": "synthetic-pose", "keep_top": 5, "drop_invalid": True},
        ],
        "seed": seed,
        **extra,
    }


def labels_for(spec: dict, cutoff: float = 0.7) -> dict[str, int]:
    target = Target(**spec["target"])
    return {c["id"]: int(hidden_activity(target, Candidate(**c)) >= cutoff) for c in spec["candidates"]}


def test_estimate_counts_the_funnel():
    est = estimate(CampaignSpec.from_dict(spec_dict()), gpu_hour_price=2.0)
    first, second = est["stages"]
    assert (first["candidates"], first["kept_at_most"]) == (200, 20)
    assert (second["candidates"], second["kept_at_most"]) == (20, 5)
    assert est["gpu_seconds"] == pytest.approx(200 * 0.01 + 20 * 20.0)
    assert est["funnel_saving"] > 0.85
    assert est["cost"] == pytest.approx(est["gpu_hours"] * 2.0, abs=1e-4)


def test_funnel_keeps_the_best_survivors():
    result = run_campaign(CampaignSpec.from_dict(spec_dict()))
    assert len(result.results) == 5
    assert [r["rank"] for r in result.results] == [1, 2, 3, 4, 5]
    scores = [r["stage_scores"]["synthetic-pose"]["score"] for r in result.results]
    assert scores == sorted(scores, reverse=True)
    assert all(r["pose"]["valid"] for r in result.results)  # drop_invalid removed the broken poses
    assert result.stage_reports[0]["scored"] == 200 and result.stage_reports[1]["scored"] == 20
    assert len(result.results) + len(result.dropped) == 200


def test_funnel_finds_more_actives_than_random():
    spec = spec_dict(400)
    labels = labels_for(spec, cutoff=0.8)
    result = run_campaign(CampaignSpec.from_dict(spec))
    hit_rate = sum(labels[r["id"]] for r in result.results) / len(result.results)
    assert hit_rate > 2 * (sum(labels.values()) / len(labels))


def test_pose_checks():
    def pose(ligand_x: float) -> str:
        return (_pdb_atom("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C") + "\n"
                + _pdb_atom("HETATM", 2, "C1", "LIG", "B", 1, ligand_x, 0.0, 0.0, "C") + "\n")

    assert check_pose(pose(4.0)).valid
    clash = check_pose(pose(1.0))
    assert not clash.valid and clash.clashes == 1
    assert "not bound" in check_pose(pose(30.0)).issues[0]
    assert not check_pose(pose(4.0).splitlines()[0]).valid  # no ligand


def test_synthetic_pose_faults_are_caught():
    spec = spec_dict(200)
    spec["stages"] = [{"backend": "synthetic-pose"}]
    result = run_campaign(CampaignSpec.from_dict(spec))
    invalid = result.stage_reports[0]["invalid_poses"]
    assert 5 < invalid < 50


def test_agreement_and_calibration():
    a = {"x": 1.0, "y": 2.0, "z": 3.0, "w": 4.0}
    assert rank_agreement(a, a) == pytest.approx(1.0)
    assert rank_agreement(a, {k: -v for k, v in a.items()}) == pytest.approx(-1.0)
    cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 0, 1, 1, 1])
    probs = [cal(s) for s in (0.0, 0.25, 0.45, 0.9)]
    assert probs == sorted(probs) and 0.0 <= probs[0] and probs[-1] == 1.0


def test_consensus_is_reported():
    result = run_campaign(CampaignSpec.from_dict(spec_dict(consensus=["synthetic-screen"])))
    assert "synthetic-pose~synthetic-screen" in result.rank_agreement
    assert all("consensus" in r for r in result.results)


def test_runs_are_reproducible():
    one = run_campaign(CampaignSpec.from_dict(spec_dict(seed=3)))
    two = run_campaign(CampaignSpec.from_dict(spec_dict(seed=3)))
    other = run_campaign(CampaignSpec.from_dict(spec_dict(seed=4)))
    assert one.manifest["output_digest"] == two.manifest["output_digest"]
    assert one.manifest["output_digest"] != other.manifest["output_digest"]
    assert one.manifest["spec_digest"] == two.manifest["spec_digest"]


def test_audit_log_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.append("event", n=i)
    assert log.verify()[0]
    lines = log.path.read_text().splitlines()
    lines[1] = lines[1].replace('"n": 1', '"n": 9')
    log.path.write_text("\n".join(lines) + "\n")
    ok, detail = log.verify()
    assert not ok and "entry 1" in detail


def test_boltz_input_reuses_the_alignment():
    target = Target("t", "MKV")
    ligand = _boltz_yaml(target, Candidate("a", smiles="CCO"), "msa/t.csv")
    assert "smiles: \"CCO\"" in ligand and "msa: \"msa/t.csv\"" in ligand and "binder: B" in ligand
    binder = _boltz_yaml(target, Candidate("b", sequence="GGG"), None)
    assert "affinity" not in binder and "sequence: GGG" in binder


def test_wsl_paths():
    from autoinference.bio.backends import _wsl_path

    assert _wsl_path("/home/user/x") == "/home/user/x"
    if os.name == "nt":
        assert _wsl_path(r"D:\data\inputs") == "/mnt/d/data/inputs"


@pytest.fixture
def service(tmp_path):
    return BioService(tmp_path / "ws", gpu_hour_price=2.0)


def test_service_campaign_tuning_and_replay(service):
    spec = spec_dict(300)
    job = service.submit(spec)
    assert service.wait(job["job_id"])["status"] == "done"
    data = service.results(job["job_id"])
    assert len(data["results"]) == 5 and data["cost"] >= 0
    assert all("structure" not in r for r in data["results"])
    assert (service.root / "jobs" / job["job_id"] / data["results"][0]["structure_file"]).exists()

    # The customer tests the funnel's first-stage picks in the lab and sends the outcomes back.
    reached = [r["id"] for r in data["results"] + data["dropped"] if r["stage_reached"] == 2]
    labels = {cid: v for cid, v in labels_for(spec).items() if cid in reached}
    model = service.fit_tuned_model(job["job_id"], labels, "assay-1")
    assert model["metrics"]["n"] == len(reached)

    tuned_job = service.submit(spec_dict(300, tuned_model="assay-1"))
    service.wait(tuned_job["job_id"])
    tuned = service.results(tuned_job["job_id"])
    assert all(r["p_active"] is not None for r in tuned["results"])
    assert tuned["manifest"]["tuned_model"]["id"] == "assay-1"

    assert service.verify(job["job_id"])["identical"]
    assert service.audit_check()["intact"]
    events = [e["event"] for e in service.audit.entries()]
    assert events.count("submitted") == 2 and "tuned" in events and "verified" in events


def test_service_learns_backend_timings(service):
    job = service.submit(spec_dict(50))
    service.wait(job["job_id"])
    est = service.estimate(spec_dict(50))
    assert all(s["seconds_source"] == "measured" for s in est["stages"])


def test_bad_spec_is_rejected(service):
    with pytest.raises(ValueError):
        service.submit({**spec_dict(3), "stages": []})
    with pytest.raises(KeyError):
        service.estimate({**spec_dict(3), "stages": [{"backend": "no-such-model"}]})


def test_http_api(service):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())

    try:
        assert call("GET", "/v1/backends")[1]
        status, est = call("POST", "/v1/estimate", spec_dict(40))
        assert status == 200 and est["stages"][0]["candidates"] == 40
        status, job = call("POST", "/v1/campaigns", spec_dict(40))
        assert status == 202
        service.wait(job["job_id"])
        _, results = call("GET", f"/v1/jobs/{job['job_id']}/results?top=2")
        assert len(results["results"]) == 2
        _, check = call("POST", f"/v1/jobs/{job['job_id']}/verify")
        assert check["identical"]
        with pytest.raises(urllib.error.HTTPError) as err:
            call("GET", "/v1/jobs/missing")
        assert err.value.code == 404
    finally:
        httpd.shutdown()


def test_mcp_tools(service):
    pytest.importorskip("mcp")
    from autoinference.bio.mcp_server import build_server

    server = build_server(service)

    async def flow():
        names = {t.name for t in await server.list_tools()}
        assert {"estimate_campaign", "submit_campaign", "job_results", "fit_tuned_model", "verify_job"} <= names
        submitted = await server.call_tool("submit_campaign", {"spec": spec_dict(30)})
        return submitted

    out = asyncio.run(flow())
    if hasattr(out, "content"):  # mcp 2.x returns a CallToolResult
        content = out.content
    else:  # mcp 1.x returns content, or (content, structured)
        content = out[0] if isinstance(out, tuple) else out
    job_id = json.loads(content[0].text)["job_id"]
    assert service.wait(job_id)["status"] == "done"


class FakeRunner:
    """Stands in for the Modal workers: scores each chunk with the synthetic pose model."""

    def __init__(self):
        from autoinference.bio.backends import SyntheticPose

        self.model = SyntheticPose()
        self.calls = []

    def call(self, request):
        self.calls.append(request)
        target = Target(**request["target"])
        cands = [Candidate(**c) for c in request["candidates"]]
        preds = self.model.predict(target, cands, request["seed"])
        return {"predictions": [{"candidate_id": p.candidate_id, "scores": p.scores, "structure": p.structure}
                                for p in preds],
                "msa": request["msa"] or "fake-alignment", "gpu_seconds": 10.0 * len(cands),
                "version": "9.9", "gpu": "FAKE-GPU"}

    def map(self, requests):
        return [self.call(r) for r in requests]


def test_cloud_backend_fans_out_and_bills_gpu_time(tmp_path):
    from autoinference.bio.cloud import Boltz2Cloud

    runner = FakeRunner()
    backend = Boltz2Cloud(chunk_size=3, msa_cache=str(tmp_path / "msa"), runner=runner)
    target = Target(**TARGET)
    cands = [Candidate(f"m{i}", smiles="C" * (i + 1)) for i in range(10)]
    preds = backend.predict(target, cands, seed=1)

    assert [p.candidate_id for p in preds] == [c.id for c in cands]
    assert [len(r["candidates"]) for r in runner.calls] == [1, 3, 3, 3]  # alignment first, then chunks
    assert runner.calls[0]["msa"] is None and all(r["msa"] == "fake-alignment" for r in runner.calls[1:])
    assert sum(p.extras["gpu_seconds"] for p in preds) == pytest.approx(100.0)
    assert backend.fingerprint()["gpu"] == "FAKE-GPU" and backend.version == "9.9"

    runner.calls.clear()  # a later campaign on the same target reuses the cached alignment
    backend.predict(target, cands[:4], seed=1)
    assert [len(r["candidates"]) for r in runner.calls] == [3, 1]


def test_campaign_cost_uses_billed_gpu_seconds(tmp_path):
    from autoinference.bio.cloud import Boltz2Cloud

    spec = spec_dict(12)
    spec["stages"] = [{"backend": "boltz2-cloud", "keep_top": 3}]
    backend = Boltz2Cloud(chunk_size=4, msa_cache=str(tmp_path / "msa"), runner=FakeRunner())
    result = run_campaign(CampaignSpec.from_dict(spec), {"boltz2-cloud": backend})
    report = result.stage_reports[0]
    assert report["gpu_seconds"] == pytest.approx(120.0) and report["seconds"] < 120.0
    assert report["seconds_per_item"] == pytest.approx(10.0)


def test_runpod_runner_queues_and_polls(monkeypatch):
    from autoinference.bio.cloud import RunPodRunner

    fake = FakeRunner()
    jobs, polls = {}, []

    def http(self, method, path, body=None):
        assert self.url.endswith("/ep123")
        if method == "POST" and path == "/run":
            jid = f"job{len(jobs)}"
            jobs[jid] = fake.call(body["input"])
            return {"id": jid, "status": "IN_QUEUE"}
        jid = path.rsplit("/", 1)[1]
        polls.append(jid)
        if polls.count(jid) == 1:
            return {"id": jid, "status": "IN_PROGRESS"}
        return {"id": jid, "status": "COMPLETED", "output": jobs[jid]}

    monkeypatch.setattr(RunPodRunner, "_http", http)
    runner = RunPodRunner("ep123", api_key="test-key", poll_seconds=0)
    request = {"target": TARGET, "candidates": [{"id": "a", "smiles": "CCO", "sequence": None}], "seed": 0,
               "msa": None, "diffusion_samples": 1, "extra_args": ""}
    out = runner.map([request, request])
    assert len(out) == 2 and all(o["predictions"][0]["candidate_id"] == "a" for o in out)

    def failing(self, method, path, body=None):
        return {"id": "x"} if path == "/run" else {"id": "x", "status": "FAILED", "error": "out of memory"}

    monkeypatch.setattr(RunPodRunner, "_http", failing)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.call(request)


def test_cloud_provider_is_checked():
    from autoinference.bio.cloud import Boltz2Cloud

    with pytest.raises(ValueError):
        Boltz2Cloud(provider="nowhere")
    assert Boltz2Cloud(provider="runpod", runner=FakeRunner()).fingerprint()["provider"] == "runpod"


def test_runpod_handler_rejects_unknown_fields():
    from autoinference.bio.runpod_worker import handler

    assert "error" in handler({"input": {"target": TARGET, "rm": "-rf"}})
