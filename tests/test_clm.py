import numpy as np
from scipy.special import log_softmax

from clm_jev_model_comparison.data.tasks import sample_per_class
from clm_jev_model_comparison.models.clm import fit_temperature, zero_shot_probs


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


def test_sample_per_class():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 3)
    idx = sample_per_class(labels, 4, seed=0)
    assert np.bincount(labels[idx]).tolist() == [4, 4, 3]
    assert np.array_equal(idx, sample_per_class(labels, 4, seed=0))
