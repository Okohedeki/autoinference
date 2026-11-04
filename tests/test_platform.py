"""The platform in local mode: catalog, requests, files, pipelines with per-step control, usage, audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoinference.platform import Catalog, Platform, Settings
from autoinference.platform.api import create_app
from autoinference.platform.pipelines import KeepRule, PipelineError, parse_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
FUNNEL = json.loads((MODELS / "demo" / "funnel.json").read_text())


@pytest.fixture
def platform(tmp_path):
    settings = Settings(storage_dir=tmp_path, models_dir=MODELS,
                        database_url=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}", sync_timeout=10)
    p = Platform(settings)
    yield p
    p.close()


@pytest.fixture
def client(platform):
    app = create_app(platform)
    project = platform.create_project("test")
    key = platform.create_key(project["id"])["key"]
    c = TestClient(app, base_url="http://testserver")
    c.headers["Authorization"] = f"Bearer {key}"
    c.project_id = project["id"]
    return c


def test_catalog_loads_the_contract():
    catalog = Catalog(MODELS)
    demo = catalog.get("demo/score")
    assert demo.file_inputs() == ["report"] and demo.file_outputs() == ["pose"]
    boltz = catalog.get("biology/boltz-2@2.2.1")
    assert boltz.hardware.gpu == ("L40S", "A100") and not boltz.sync and boltz.worker_name == "W_biology_boltz_2"
    with pytest.raises(KeyError):
        catalog.get("biology/boltz-2@9.9")
    with pytest.raises(ValueError, match="molecule"):
        demo.validate_inputs({"target": TARGET})
    assert "demo/score" not in {m["id"] for m in [s.to_public() for s in catalog.public()]}  # not public


def test_auth_and_keys(client, platform):
    assert client.get("/v1/models").status_code == 200
    bad = TestClient(create_app(platform))
    assert bad.get("/v1/requests").status_code == 401
    assert bad.get("/v1/requests", headers={"Authorization": "Bearer ai_nope_nope"}).status_code == 401
    created = client.post("/v1/keys", json={"name": "second"}).json()
    assert created["key"].startswith("ai_")
    assert client.delete(f"/v1/keys/{created['id']}").status_code == 204
    assert platform.authenticate(created["key"]) is None


def test_run_sync_and_queue(client):
    r = client.post("/v1/models/demo/score/run", json={"inputs": {"target": TARGET, "molecule": "CCO"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["outputs"]["score"] <= 1 and body["outputs"]["pose"]["url"].endswith("pose.pdb")
    assert body["manifest"]["fingerprint"]["model"] == "demo/score" and body["price"] == pytest.approx(0.0001)
    pose = client.get(body["outputs"]["pose"]["url"])
    assert pose.status_code == 200 and b"demo pose" in pose.content

    q = client.post("/v1/models/demo/score/queue", json={"inputs": {"target": TARGET, "molecule": "CCN"}})
    assert q.status_code == 202
    rid = q.json()["request_id"]
    st = client.app.state.platform.wait(client.project_id, rid, 10)
    assert st["status"] == "succeeded"
    assert client.get(f"/v1/requests/{rid}/result").json()["outputs"]["score"] > 0
    again = client.post("/v1/models/demo/score/run", json={"inputs": {"target": TARGET, "molecule": "CCO"}})
    assert again.json()["outputs"]["score"] == body["outputs"]["score"]  # deterministic

    bad = client.post("/v1/models/demo/score/run", json={"inputs": {"target": TARGET}})
    assert bad.status_code == 400 and "molecule" in bad.json()["error"]["message"]
    failed = client.post("/v1/models/demo/score/run",
                         json={"inputs": {"target": TARGET, "molecule": "CCX", "fail_if_contains": "X"}})
    assert failed.status_code == 500 and "contains" in failed.json()["error"]["message"]
    assert client.get("/v1/models/nope/none").status_code == 404
    assert client.post("/v1/models/biology/boltz-2/run", json={"inputs": {"protein": TARGET}}).status_code == 400


def test_idempotency(client):
    body = {"inputs": {"target": TARGET, "molecule": "CCO"}}
    a = client.post("/v1/models/demo/score/queue", json=body, headers={"Idempotency-Key": "k1"}).json()
    b = client.post("/v1/models/demo/score/queue", json=body, headers={"Idempotency-Key": "k1"}).json()
    assert a["request_id"] == b["request_id"]


def test_files_flow_through_storage(client):
    f = client.post("/v1/files", json={"filename": "report.txt", "content_type": "text/plain"}).json()
    assert client.get(f"/v1/files/{f['file_id']}").json()["uploaded"] is False
    missing = client.post("/v1/models/demo/score/run",
                          json={"inputs": {"target": TARGET, "molecule": "CCO", "report": f["file_id"]}})
    assert missing.status_code == 400 and "not been uploaded" in missing.json()["error"]["message"]
    assert client.put(f["upload_url"], content=b"hello world").status_code == 204
    r = client.post("/v1/models/demo/score/run",
                    json={"inputs": {"target": TARGET, "molecule": "CCO", "report": f["file_id"]}})
    assert r.json()["outputs"]["report_size"] == 11
    assert client.post("/v1/models/demo/score/run",
                       json={"inputs": {"target": TARGET, "molecule": "CCO", "report": "nope"}}).status_code == 400


def test_pipeline_parsing_and_keep():
    with pytest.raises(PipelineError, match="earlier step"):
        parse_pipeline({"steps": [{"id": "a", "model": "m", "inputs": {"x": "$b.outputs.y"}}]})
    with pytest.raises(PipelineError, match="keep needs map"):
        parse_pipeline({"steps": [{"id": "a", "model": "m", "keep": {"top": 1}}]})
    items = [{"index": i, "outputs": {"score": s, "qed": q}} for i, (s, q) in
             enumerate([(0.9, 0.1), (0.5, 0.9), (0.7, 0.5), (0.2, 0.8)])]
    kept = KeepRule(top=2, by="score", min={"qed": 0.3}).apply(items)
    assert [it["index"] for it in kept] == [2, 1]
    assert [it["index"] for it in KeepRule(fraction=0.5, by="qed").apply(items)] == [1, 3]


def test_pipeline_runs_with_keep_approval_and_rerun(client):
    platform = client.app.state.platform
    molecules = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CCOC", "CCS"]
    inputs = {"target": TARGET, "molecules": molecules}
    est = client.post("/v1/pipelines/estimate", json={"pipeline": FUNNEL, "inputs": inputs}).json()
    assert [s["calls"] for s in est["steps"]] == [8, 3] and est["steps"][1]["approve"]

    run = client.post("/v1/pipelines/run", json={"pipeline": FUNNEL, "inputs": inputs})
    assert run.status_code == 202, run.text
    run_id = run.json()["run_id"]
    deadline = 20
    import time
    while time.time() < time.time() + 0 and False:
        pass
    end = time.time() + deadline
    while time.time() < end:
        st = client.get(f"/v1/pipelines/runs/{run_id}").json()
        if st["status"] == "waiting_approval":
            break
        time.sleep(0.05)
    assert st["status"] == "waiting_approval" and st["current_step"] == "rescore"
    screen = client.get(f"/v1/pipelines/runs/{run_id}/steps/screen").json()
    assert screen["status"] == "succeeded" and len(screen["calls"]) == 8 and len(screen["items"]) == 3
    assert len(screen["dropped"]) == 5 and all(c["status"]["status"] == "succeeded" for c in screen["calls"])
    scores = [it["outputs"]["score"] for it in screen["items"]]
    assert scores == sorted(scores, reverse=True)
    assert client.get(f"/v1/pipelines/runs/{run_id}/steps/rescore").json()["status"] == "pending"

    client.post(f"/v1/pipelines/runs/{run_id}/steps/rescore/approve")
    end = time.time() + deadline
    while time.time() < end:
        st = client.get(f"/v1/pipelines/runs/{run_id}").json()
        if st["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert st["status"] == "succeeded", st
    assert len(st["outputs"]["shortlist"]) == 3 and st["price"] == pytest.approx(11 * 0.0001)
    parent = client.get(f"/v1/requests/{run_id}").json()
    assert parent["status"] == "succeeded" and parent["model"] == "pipeline:demo-funnel"
    child = client.get(f"/v1/requests/{st['steps'][0]['request_ids'][0]}").json()
    assert child["pipeline"] == {"run_id": run_id, "step": "screen", "index": 0}
    usage = client.get("/v1/usage").json()
    assert usage["by_model"][0]["requests"] == 11  # children are billed, the parent is not

    # Rerun from the second step: the screen results are reused, nothing is recomputed for them.
    rerun = client.post("/v1/pipelines/run", json={"pipeline": FUNNEL, "inputs": inputs,
                                                   "rerun_from": {"run_id": run_id, "step": "rescore"}}).json()
    assert rerun["steps"][0]["reused_from"] == run_id and rerun["status"] == "waiting_approval"
    platform.approve_step(client.project_id, rerun["run_id"], "rescore")
    end = time.time() + deadline
    while time.time() < end and platform.pipeline_status(client.project_id, rerun["run_id"])["status"] != "succeeded":
        time.sleep(0.05)
    assert platform.pipeline_status(client.project_id, rerun["run_id"])["price"] == pytest.approx(3 * 0.0001)
    assert client.get("/v1/usage").json()["by_model"][0]["requests"] == 14


def test_pipeline_failure_skip_and_cancel(client):
    platform = client.app.state.platform
    import time

    failing = {"name": "f", "steps": [
        {"id": "a", "model": "demo/score", "map": "$inputs.molecules",
         "inputs": {"target": TARGET, "molecule": "$item", "fail_if_contains": "X"}, "retries": 1},
        {"id": "b", "model": "demo/score", "when": "$inputs.run_b",
         "inputs": {"target": TARGET, "molecule": "CCO"}}]}
    run = platform.submit_pipeline({"id": client.project_id}, failing, {"molecules": ["CCO", "CCX"], "run_b": False})
    end = time.time() + 20
    while time.time() < end and platform.pipeline_status(client.project_id, run["run_id"])["status"] == "running":
        time.sleep(0.05)
    st = platform.pipeline_status(client.project_id, run["run_id"])
    assert st["status"] == "failed" and st["steps"][0]["attempts"] == 2 and "contains" in st["steps"][0]["error"]

    ok = platform.submit_pipeline({"id": client.project_id}, failing, {"molecules": ["CCO"], "run_b": False})
    end = time.time() + 20
    while time.time() < end and platform.pipeline_status(client.project_id, ok["run_id"])["status"] == "running":
        time.sleep(0.05)
    st = platform.pipeline_status(client.project_id, ok["run_id"])
    assert st["status"] == "succeeded" and st["steps"][1]["status"] == "skipped"

    gated = {**FUNNEL}
    run = platform.submit_pipeline({"id": client.project_id}, gated, {"target": TARGET, "molecules": ["CCO", "CCN"]})
    end = time.time() + 20
    while time.time() < end and platform.pipeline_status(client.project_id, run["run_id"])["status"] == "running":
        time.sleep(0.05)
    cancelled = client.post(f"/v1/pipelines/runs/{run['run_id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    assert client.get(f"/v1/requests/{run['run_id']}").json()["status"] == "cancelled"


def test_audit_chain_and_estimate(client):
    client.post("/v1/models/demo/score/run", json={"inputs": {"target": TARGET, "molecule": "CCO"}})
    audit = client.get("/v1/audit").json()
    assert audit["intact"] and {"request_submitted", "request_succeeded"} <= {e["event"] for e in audit["recent"]}
    est = client.post("/v1/models/demo/score/estimate", json={"inputs": {"target": TARGET, "molecule": "CCO"},
                                                               "units": 100}).json()
    assert est["price"] == pytest.approx(0.01) and est["seconds_source"] == "measured"
