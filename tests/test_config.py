import pytest
import yaml

from clm_jev_model_comparison.utils.config import apply_overrides


def test_overrides_parse_types():
    cfg = {"task": {"test_size": 500, "seed": 0}, "method": {"C": 1.0}}
    apply_overrides(cfg, ["task.test_size=50", "method.C=0.1"])
    assert cfg["task"]["test_size"] == 50 and cfg["method"]["C"] == 0.1


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        apply_overrides({"task": {}}, ["task.typo=1"])


def test_task_config_labels_well_formed():
    with open("configs/tasks/ag_news.yaml") as f:
        cfg = yaml.safe_load(f)
    keys = [lab["key"] for lab in cfg["labels"]]
    assert len(keys) == 4 and len(set(keys)) == 4
    assert all(lab["description"] for lab in cfg["labels"])
