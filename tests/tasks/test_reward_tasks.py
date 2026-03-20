from tasks.base_task import BaseTask
from tasks.reward_tasks import (
    MaxScore, ThresholdBeater, UpperBonus,
    YahtzeeHunter, Conservative, TASK_REGISTRY,
)


def test_all_tasks_in_registry():
    expected = {"MaxScore", "ThresholdBeater", "UpperBonus", "YahtzeeHunter", "Conservative"}
    assert set(TASK_REGISTRY.keys()) == expected


def test_each_task_is_base_task_subclass():
    for name, cls in TASK_REGISTRY.items():
        instance = cls() if name != "ThresholdBeater" else cls(threshold=250)
        assert isinstance(instance, BaseTask), f"{name} must subclass BaseTask"


def test_threshold_beater_stores_threshold():
    task = ThresholdBeater(threshold=300)
    assert task.threshold == 300


def test_each_task_has_name_property():
    for name, cls in TASK_REGISTRY.items():
        instance = cls() if name != "ThresholdBeater" else cls(threshold=250)
        assert isinstance(instance.name, str)
        assert len(instance.name) > 0
