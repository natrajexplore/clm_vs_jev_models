"""Local comparison app: results dashboard + live playground.

    uv run --env-file .env uvicorn clm_jev_model_comparison.app.server:app --port 8006

The Jev API key is read from the environment on the server; it never reaches the browser.
"""

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data.tasks import load_task, sample_per_class
from ..eval.metrics import reliability_bins
from ..models.clm import ContrastiveEncoder, fit_probe, fit_temperature, zero_shot_probs
from ..models.jev import JevClient
from ..utils.config import load_config

RUNS_DIR = Path(os.environ.get("CLM_JEV_RUNS", "results/runs"))
TASK_CONFIG = os.environ.get("CLM_JEV_TASK", "configs/tasks/ag_news.yaml")
METHOD_CONFIGS = {
    "jev": "configs/methods/jev.yaml",
    "clm_zeroshot": "configs/methods/clm_zeroshot.yaml",
    "clm_probe": "configs/methods/clm_probe.yaml",
}
STATIC = Path(__file__).parent / "static"
METRICS = ["accuracy", "macro_f1", "ece", "brier", "nll", "cost_usd"]

app = FastAPI(title="CLM vs Jev")


# ---------- results ----------

def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Every run with a metrics.json that records a variant (older runs are skipped)."""
    runs = []
    for path in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if "variant" in m:
            m["run"] = path.parent.name
            m["latency_ms"] = m["latency"]["p50_ms"] if "latency" in m else m.get("ms_per_example")
            runs.append(m)
    return runs


def _predictions(runs_dir: Path, run: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in (runs_dir / run / "predictions.jsonl").read_text().splitlines()]
    return np.array([r["probs"] for r in rows]), np.array([r["label"] for r in rows])


def summarize(runs: list[dict[str, Any]], runs_dir: Path) -> list[dict[str, Any]]:
    """Group runs by (task, method, variant): mean/std over seeds + pooled reliability diagram."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["task"], r["method"], r["variant"]), []).append(r)

    out = []
    for (task, method, variant), rs in sorted(groups.items()):
        stats = {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))} for k in METRICS}
        preds = [_predictions(runs_dir, r["run"]) for r in rs]
        probs, labels = np.concatenate([p for p, _ in preds]), np.concatenate([y for _, y in preds])
        out.append({
            "task": task, "method": method, "variant": variant, "model": rs[0]["model"],
            "n_runs": len(rs), "n_test": rs[0]["n"], "seeds": sorted(r["seed"] for r in rs),
            "labeled_examples_used": rs[0]["labeled_examples_used"],
            "latency_ms": float(np.mean([r["latency_ms"] for r in rs])),
            "latency_kind": "network p50" if method == "jev" else "local CPU, amortized",
            "accuracy_ci95": rs[0]["accuracy_ci95"] if len(rs) == 1 else None,
            "model_version": rs[0].get("model_version"),
            "C": [r["C"] for r in rs] if "C" in rs[0] else None,
            "temperature": [r["temperature"] for r in rs] if "temperature" in rs[0] else None,
            "reliability": reliability_bins(probs, labels),
            **stats,
        })
    return out


@app.get("/api/results")
def results() -> dict[str, Any]:
    runs = load_runs(RUNS_DIR)
    return {"groups": summarize(runs, RUNS_DIR), "runs": runs}


# ---------- live playground ----------

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class Playground:
    """CLM zero-shot (temperature-scaled), CLM probe (CV-chosen C) and Jev, set up exactly as in run.py."""

    def __init__(self) -> None:
        task_cfg = load_config(TASK_CONFIG)
        zs, pr, jv = (load_config(METHOD_CONFIGS[k]) for k in ("clm_zeroshot", "clm_probe", "jev"))
        self.task = load_task(task_cfg)
        self.encoder = ContrastiveEncoder(zs["model"], zs["batch_size"])
        self.label_emb = self.encoder.embed([zs["label_template"].format(description=d) for d in self.task.label_descriptions])

        idx = np.random.default_rng(zs["seed"]).permutation(len(self.task.train_texts))[: zs["calibration_size"]]
        sims = self.encoder.embed([self.task.train_texts[i] for i in idx]) @ self.label_emb.T
        self.temperature = fit_temperature(sims, self.task.train_labels[idx])

        idx = sample_per_class(self.task.train_labels, pr["shots_per_class"], pr["seed"])
        x = self.encoder.embed([self.task.train_texts[i] for i in idx])
        self.probe, self.probe_C = fit_probe(x, self.task.train_labels[idx], pr["C"], pr["max_iter"], pr["cv_folds"], pr["seed"])

        try:  # no cache: playground latency should be real
            self.jev: JevClient | None = JevClient(
                model=jv["model"], base_url=jv["base_url"], timeout_s=jv["timeout_s"], max_retries=jv["max_retries"]
            )
            self.jev_error = None
        except RuntimeError as e:
            self.jev, self.jev_error = None, str(e)

    def _as_result(self, probs: np.ndarray, latency_s: float) -> dict[str, Any]:
        keys = self.task.label_keys
        return {"choice": keys[int(probs.argmax())], "probabilities": dict(zip(keys, probs.round(4).tolist())),
                "latency_ms": latency_s * 1e3}

    def predict(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        emb = self.encoder.embed([text])
        embed_s = time.perf_counter() - t0

        zs = zero_shot_probs(emb @ self.label_emb.T, self.temperature)[0]
        t1 = time.perf_counter()
        pr = np.zeros(len(self.task.label_keys))
        pr[self.probe.classes_] = self.probe.predict_proba(emb)[0]
        probe_s = embed_s + time.perf_counter() - t1

        out: dict[str, Any] = {
            "clm_zeroshot": {**self._as_result(zs, embed_s), "note": f"temperature {self.temperature:.3f}"},
            "clm_probe": {**self._as_result(pr, probe_s), "note": f"16-shot probe, C={self.probe_C:g}"},
        }
        if self.jev is None:
            out["jev"] = {"error": self.jev_error}
        else:
            try:
                a = self.jev.choice([text], self.task.instructions, self.task.criteria)[0]
                probs = np.array([a.probabilities.get(k, 0.0) for k in self.task.label_keys])
                out["jev"] = {**self._as_result(probs, a.latency_s), "confidence": a.confidence, "note": a.model}
            except Exception as e:  # show API failures in the UI instead of failing the whole request
                out["jev"] = {"error": f"{type(e).__name__}: {e}"}
        return out


@lru_cache(maxsize=1)
def get_playground() -> Playground:
    return Playground()


@app.get("/api/task")
def task_info() -> dict[str, Any]:
    cfg = load_config(TASK_CONFIG)
    return {"name": cfg["name"], "instructions": cfg["instructions"], "labels": cfg["labels"]}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    # Sync endpoint: FastAPI runs it in a worker thread, where JevClient's asyncio.run is allowed.
    try:
        return get_playground().predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
