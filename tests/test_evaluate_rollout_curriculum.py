import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_rollout import _trained_rollout_steps_for_checkpoint  # noqa: E402


def test_trained_rollout_steps_without_curriculum() -> None:
    config = {"training": {"rollout_steps": 12}}
    assert _trained_rollout_steps_for_checkpoint(config, checkpoint_epoch=20) == 12


def test_trained_rollout_steps_follow_checkpoint_epoch() -> None:
    config = {
        "training": {
            "rollout_steps": 18,
            "rollout_curriculum": [
                {"until_epoch": 5, "steps": 15},
                {"steps": 18},
            ],
        }
    }

    assert _trained_rollout_steps_for_checkpoint(config, checkpoint_epoch=4) == 15
    assert _trained_rollout_steps_for_checkpoint(config, checkpoint_epoch=5) == 15
    assert _trained_rollout_steps_for_checkpoint(config, checkpoint_epoch=6) == 18
    assert _trained_rollout_steps_for_checkpoint(config, checkpoint_epoch=30) == 18
