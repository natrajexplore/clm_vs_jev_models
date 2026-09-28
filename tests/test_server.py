import json

import pytest
from fastapi.testclient import TestClient

from clm_jev_model_comparison.app import server


def _write_run(root, name, method, variant, seed, acc, probs, labels):
    d = root / name
    d.mkdir()
    metrics = {"task": "t", "method": method, "variant": variant, "model": "m", "seed": seed, "n": len(labels),
               "accuracy": acc, "accuracy_ci95": [acc - 0.1, acc + 0.1], "macro_f1": acc, "ece": 0.1, "brier": 0.2,
               "nll": 0.3, "cost_usd": 0.0, "labeled_examples_used": 64, "ms_per_example": 10.0}
    (d / "metrics.json").write_text(json.dumps(metrics))
    (d / "predictions.jsonl").write_text("\n".join(json.dumps({"probs": p, "label": y}) for p, y in zip(probs, labels)))


@pytest.fixture
def client(tmp_path, monkeypatch):
    probs, labels = [[0.9, 0.1], [0.3, 0.7]], [0, 0]
    _write_run(tmp_path, "a", "clm_probe", "cv_C", 0, 0.8, probs, labels)
    _write_run(tmp_path, "b", "clm_probe", "cv_C", 1, 0.6, probs, labels)
    _write_run(tmp_path, "c", "jev", "zero_shot", 0, 0.9, probs, labels)
    (tmp_path / "old").mkdir()  # pre-variant runs are skipped
    (tmp_path / "old" / "metrics.json").write_text(json.dumps({"task": "t", "method": "x"}))
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    return TestClient(server.app)


def test_results_grouped_over_seeds(client):
    groups = {(g["method"], g["variant"]): g for g in client.get("/api/results").json()["groups"]}
    assert set(groups) == {("clm_probe", "cv_C"), ("jev", "zero_shot")}
    probe = groups[("clm_probe", "cv_C")]
    assert probe["n_runs"] == 2 and probe["seeds"] == [0, 1]
    assert probe["accuracy"]["mean"] == pytest.approx(0.7) and probe["accuracy"]["std"] == pytest.approx(0.1)
    assert probe["accuracy_ci95"] is None  # CI only for single runs; seeds give the spread
    assert sum(b["count"] for b in probe["reliability"]) == 4  # both seeds pooled


def test_predict_validates_input(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 422


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CLM vs Jev" in r.text
