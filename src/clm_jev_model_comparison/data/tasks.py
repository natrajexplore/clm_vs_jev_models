from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import load_dataset


@dataclass
class TaskData:
    """A text classification task. Labels are ints indexing ``label_keys``."""

    name: str
    instructions: str
    label_keys: list[str]
    label_descriptions: list[str]
    train_texts: list[str]
    train_labels: np.ndarray  # (N_train,)
    test_texts: list[str]
    test_labels: np.ndarray  # (N_test,)

    @property
    def criteria(self) -> dict[str, str]:
        """Option name -> description, as sent to Jev and embedded for CLM zero-shot."""
        return dict(zip(self.label_keys, self.label_descriptions))


def load_task(cfg: dict[str, Any]) -> TaskData:
    """Load a Hugging Face text classification dataset.

    The test set is a fixed random subset of ``test_size`` examples (seeded by ``cfg['seed']``),
    so every method is scored on exactly the same examples.
    """
    ds = load_dataset(cfg["hf_path"], revision=cfg.get("revision"))
    train, test = ds[cfg["train_split"]], ds[cfg["test_split"]]

    names = train.features[cfg["label_field"]].names
    if len(names) != len(cfg["labels"]):
        raise ValueError(f"Config lists {len(cfg['labels'])} labels but dataset has {len(names)}: {names}")

    rng = np.random.default_rng(cfg["seed"])
    test_idx = rng.permutation(len(test))[: cfg["test_size"]]
    test = test.select(test_idx)

    return TaskData(
        name=cfg["name"],
        instructions=cfg["instructions"],
        label_keys=[lab["key"] for lab in cfg["labels"]],
        label_descriptions=[lab["description"] for lab in cfg["labels"]],
        train_texts=list(train[cfg["text_field"]]),
        train_labels=np.asarray(train[cfg["label_field"]]),
        test_texts=list(test[cfg["text_field"]]),
        test_labels=np.asarray(test[cfg["label_field"]]),
    )


def sample_per_class(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of ``k`` random examples per class (fewer if a class is smaller)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:k] for c in np.unique(labels)])
