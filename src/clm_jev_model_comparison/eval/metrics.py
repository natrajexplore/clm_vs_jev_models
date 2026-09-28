import numpy as np
from sklearn.metrics import f1_score

EPS = 1e-12


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: probs (N, C), labels (N,). Weighted mean |accuracy - confidence| over equal-width bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)  # conf == 1.0 goes in the last bin
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Accuracy, macro-F1 and calibration (ECE, Brier, NLL) from probs (N, C) and labels (N,).

    NLL clips probabilities at 1e-12: a confidently wrong 0.0 costs ~27.6 nats instead of infinity.
    """
    n, c = probs.shape
    preds = probs.argmax(axis=1)
    onehot = np.eye(c)[labels]
    return {
        "n": int(n),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=np.arange(c), zero_division=0)),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(n), labels], EPS, 1.0)).mean()),
    }


def latency_summary(latencies_s: list[float]) -> dict[str, float]:
    a = np.asarray(latencies_s)
    return {"p50_ms": float(np.percentile(a, 50) * 1e3), "p95_ms": float(np.percentile(a, 95) * 1e3), "mean_ms": float(a.mean() * 1e3)}
