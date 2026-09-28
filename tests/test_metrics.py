import math

import numpy as np

from clm_jev_model_comparison.eval.metrics import classification_metrics, expected_calibration_error, latency_summary


def test_perfect_one_hot_predictions():
    labels = np.array([0, 1, 2, 1])
    m = classification_metrics(np.eye(3)[labels], labels)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["ece"] == 0.0 and m["brier"] == 0.0 and m["nll"] < 1e-9


def test_uniform_predictions_known_values():
    labels = np.array([0, 1, 2, 3])
    m = classification_metrics(np.full((4, 4), 0.25), labels)
    assert math.isclose(m["brier"], 0.75)  # 0.75^2 + 3 * 0.25^2
    assert math.isclose(m["nll"], math.log(4))


def test_ece_known_value():
    # All predictions at confidence 0.8, half of them correct -> ECE = |0.5 - 0.8| = 0.3
    probs = np.array([[0.8, 0.2]] * 4)
    labels = np.array([0, 0, 1, 1])
    assert math.isclose(expected_calibration_error(probs, labels), 0.3)


def test_ece_handles_confidence_one():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 0])) == 0.5


def test_nll_is_finite_for_confidently_wrong_zero():
    m = classification_metrics(np.array([[1.0, 0.0]]), np.array([1]))
    assert math.isfinite(m["nll"]) and m["nll"] > 20


def test_latency_summary():
    s = latency_summary([0.1, 0.2, 0.3])
    assert math.isclose(s["p50_ms"], 200.0) and math.isclose(s["mean_ms"], 200.0)
