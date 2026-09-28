"""Run one method on one task and write predictions + metrics.

    uv run python -m clm_jev_model_comparison.run --task configs/tasks/ag_news.yaml --method configs/methods/jev.yaml
    uv run python -m clm_jev_model_comparison.run --task ... --method ... --set task.test_size=50
"""

import argparse
import json
import time
from typing import Any

import numpy as np
import yaml

from .data.tasks import TaskData, load_task, sample_per_class
from .eval.metrics import classification_metrics, latency_summary
from .models.clm import ContrastiveEncoder, fit_probe, fit_temperature, zero_shot_probs
from .models.jev import JevClient
from .utils.config import apply_overrides
from .utils.run_logger import make_run_dir, save_config, save_json
from .utils.seed import seed_everything

# Each runner returns probs (N_test, C) in task.label_keys order, per-example extras, and run-level info.
RunResult = tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]


def run_jev(task: TaskData, m: dict[str, Any], seed: int) -> RunResult:
    client = JevClient(
        model=m["model"], base_url=m["base_url"], max_concurrency=m["max_concurrency"],
        timeout_s=m["timeout_s"], max_retries=m["max_retries"], cache_path=m["cache"],
    )
    t0 = time.perf_counter()
    answers = client.choice(task.test_texts, task.instructions, task.criteria)
    wall_s = time.perf_counter() - t0

    probs = np.array([[a.probabilities.get(k, 0.0) for k in task.label_keys] for a in answers])
    tokens = [a.input_tokens for a in answers]
    known_tokens = sum(t for t in tokens if t is not None)
    info = {
        "model_version": answers[0].model if answers else None,
        "latency": latency_summary([a.latency_s for a in answers]),
        "wall_seconds": round(wall_s, 2),
        "cached_fraction": float(np.mean([a.cached for a in answers])),
        "input_tokens": known_tokens,
        "input_tokens_missing": sum(t is None for t in tokens),
        "cost_usd": known_tokens / 1e6 * m["price_per_mtok_input"],
        "labeled_examples_used": 0,
    }
    extras = [{"jev_choice": a.choice, "jev_confidence": a.confidence, "latency_s": a.latency_s} for a in answers]
    return probs, extras, info


def run_clm_zeroshot(task: TaskData, m: dict[str, Any], seed: int) -> RunResult:
    enc = ContrastiveEncoder(m["model"], m["batch_size"])
    label_emb = enc.embed([m["label_template"].format(description=d) for d in task.label_descriptions])  # (C, D)

    info: dict[str, Any] = {"labeled_examples_used": 0, "temperature": m["temperature"]}
    if m["calibration_size"]:
        # Temperature scaling on labeled TRAIN examples only; test labels are never seen.
        idx = np.random.default_rng(seed).permutation(len(task.train_texts))[: m["calibration_size"]]
        cal_sims = enc.embed([task.train_texts[i] for i in idx]) @ label_emb.T
        info["temperature"] = fit_temperature(cal_sims, task.train_labels[idx])
        info["labeled_examples_used"] = len(idx)

    t0 = time.perf_counter()
    sims = enc.embed(task.test_texts) @ label_emb.T  # (N, C)
    embed_s = time.perf_counter() - t0
    info["embed_seconds"] = round(embed_s, 2)
    info["ms_per_example"] = embed_s / len(task.test_texts) * 1e3
    info["cost_usd"] = 0.0  # local compute
    return zero_shot_probs(sims, info["temperature"]), [{} for _ in task.test_texts], info


def run_clm_probe(task: TaskData, m: dict[str, Any], seed: int) -> RunResult:
    enc = ContrastiveEncoder(m["model"], m["batch_size"])
    idx = sample_per_class(task.train_labels, m["shots_per_class"], seed)
    probe = fit_probe(enc.embed([task.train_texts[i] for i in idx]), task.train_labels[idx], m["C"], m["max_iter"])

    t0 = time.perf_counter()
    x = enc.embed(task.test_texts)
    embed_s = time.perf_counter() - t0
    probs = np.zeros((len(x), len(task.label_keys)))
    probs[:, probe.classes_] = probe.predict_proba(x)
    info = {
        "labeled_examples_used": len(idx),
        "embed_seconds": round(embed_s, 2),
        "ms_per_example": embed_s / len(x) * 1e3,
        "cost_usd": 0.0,
    }
    return probs, [{} for _ in task.test_texts], info


RUNNERS = {"jev": run_jev, "clm_zeroshot": run_clm_zeroshot, "clm_probe": run_clm_probe}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. task.test_size=50")
    args = parser.parse_args()

    with open(args.task) as f_task, open(args.method) as f_method:
        cfg = {"task": yaml.safe_load(f_task), "method": yaml.safe_load(f_method)}
    apply_overrides(cfg, args.set)
    seed = cfg["task"]["seed"]
    seed_everything(seed)

    task = load_task(cfg["task"])
    method = cfg["method"]["name"]
    probs, extras, info = RUNNERS[method](task, cfg["method"], seed)

    run_dir = make_run_dir(args.output_root, f"{task.name}_{method}")
    save_config(run_dir, cfg)
    with open(run_dir / "predictions.jsonl", "w") as f:
        for i, (label, p, extra) in enumerate(zip(task.test_labels, probs, extras)):
            f.write(json.dumps({"i": i, "label": int(label), "pred": int(p.argmax()), "probs": p.round(6).tolist(), **extra}) + "\n")

    metrics = {"task": task.name, "method": method, "model": cfg["method"]["model"], "seed": seed,
               **classification_metrics(probs, task.test_labels), **info}
    save_json(run_dir / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
