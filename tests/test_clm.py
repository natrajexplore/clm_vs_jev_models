import numpy as np
from scipy.special import log_softmax

from clm_jev_model_comparison.data.tasks import sample_per_class
from clm_jev_model_comparison.models.clm import fit_probe, fit_temperature, zero_shot_probs


def _nll(sims, labels, t):
    return -log_softmax(sims / t, axis=1)[np.arange(len(labels)), labels].mean()


def test_zero_shot_probs_normalized_and_ranked():
    sims = np.array([[0.9, 0.1, 0.3], [0.2, 0.1, 0.8]])
    p = zero_shot_probs(sims, 0.05)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.argmax(axis=1).tolist() == [0, 2]


def test_fit_temperature_minimizes_nll():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400)
    sims = rng.normal(0.2, 0.05, (400, 4))
    sims[np.arange(400), labels] += 0.08  # informative but noisy
    t = fit_temperature(sims, labels)
    for other in (0.005, 0.2, 1.0):
        assert _nll(sims, labels, t) <= _nll(sims, labels, other) + 1e-9


def test_fit_probe_fixed_and_cv():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 16)
    x = rng.normal(size=(48, 8))
    x[np.arange(48), y] += 3.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # unit-norm, like sentence embeddings

    _, c_fixed = fit_probe(x, y, 1.0, max_iter=500)
    assert c_fixed == 1.0
    probe, c_cv = fit_probe(x, y, [0.1, 1.0, 100.0], max_iter=500, cv_folds=4, seed=0)
    assert c_cv in (0.1, 1.0, 100.0)
    # Separable unit-norm data: CV on log-loss should prefer weaker regularization than C=0.1.
    assert c_cv > 0.1
    assert probe.predict_proba(x).max(axis=1).mean() > 0.5


def test_sample_per_class():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 3)
    idx = sample_per_class(labels, 4, seed=0)
    assert np.bincount(labels[idx]).tolist() == [4, 4, 3]
    assert np.array_equal(idx, sample_per_class(labels, 4, seed=0))
