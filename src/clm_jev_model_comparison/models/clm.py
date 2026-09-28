"""Contrastive language model (CLM): a contrastively trained sentence encoder.

Two ways to turn embeddings into class probabilities:
- zero-shot: softmax over cosine similarity to embedded label descriptions
- linear probe: logistic regression trained on k labeled examples per class
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class ContrastiveEncoder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts -> L2-normalized embeddings (N, D)."""
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)


def zero_shot_probs(sims: np.ndarray, temperature: float) -> np.ndarray:
    """Cosine similarities (N, C) -> class probabilities (N, C)."""
    return softmax(sims / temperature, axis=1)


def fit_temperature(sims: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (1e-3, 10.0)) -> float:
    """Single softmax temperature minimizing NLL on labeled (N, C) similarities (temperature scaling)."""

    def nll(log_t: float) -> float:
        logp = log_softmax(sims / np.exp(log_t), axis=1)
        return -logp[np.arange(len(labels)), labels].mean()

    res = minimize_scalar(nll, bounds=np.log(bounds), method="bounded")
    return float(np.exp(res.x))


def fit_probe(x: np.ndarray, y: np.ndarray, C: float, max_iter: int) -> LogisticRegression:
    """Logistic regression on frozen embeddings x (N, D)."""
    return LogisticRegression(C=C, max_iter=max_iter).fit(x, y)
