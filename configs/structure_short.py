"""Shortest terminal-budget structure recipe: 18,240 presentations."""

from dirtflow.training import Config


def get_config() -> Config:
    return Config(presentations=18_240, warmup=912)
