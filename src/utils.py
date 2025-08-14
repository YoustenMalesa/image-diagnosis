import os
from contextlib import contextmanager


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


@contextmanager
def torch_no_grad():
    import torch

    with torch.no_grad():
        yield
