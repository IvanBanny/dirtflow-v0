"""Continuous material-stage optimizer and presentation schedule."""

from dirtflow.training import Config


def get_config() -> Config:
    return Config(
        stage="materials",
        presentations=2_400_000,
        hold_fraction=0.7,
        replay_initial=0,
        replay_final=0,
    )
