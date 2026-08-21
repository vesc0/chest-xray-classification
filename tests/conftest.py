"""
Shared fixtures. Everything is synthetic: the suite must pass with the dataset
unmounted, no GPU and no checkpoint, and may not write into outputs/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Importable regardless of pytest's rootdir inference.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@pytest.fixture(autouse=True)
def restore_config():
    """
    Undo config mutations after every test — it is module-level mutable state
    that main.py rewrites from the CLI, so changes leak between tests.
    """
    saved = {
        name: getattr(config, name)
        for name in dir(config)
        if name.isupper() and not name.startswith("_")
    }
    yield
    for name, value in saved.items():
        setattr(config, name, value)


@pytest.fixture(autouse=True)
def disable_bootstrap(restore_config):
    """
    Keep the bootstrap out of the tests that are not about it, which otherwise
    turns the suite from two seconds into forty. Depends on restore_config so
    it runs after the snapshot and the real value comes back at teardown.
    """
    config.BOOTSTRAP_ENABLED = False


@pytest.fixture
def rng():
    return np.random.default_rng(20260811)


def make_label_frame(
    n_patients: int = 300,
    max_studies: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """
    A metadata frame shaped like load_metadata()'s output, with prevalences
    spanning the real long tail (40% down to 0.5%) — the stratification code is
    only interesting when a class can be lost by a naive split.
    """
    generator = np.random.default_rng(seed)
    prevalence = np.linspace(0.40, 0.005, config.NUM_CLASSES)

    rows = []
    for patient in range(n_patients):
        # One label vector per patient, so a patient is a coherent group.
        vector = (generator.random(config.NUM_CLASSES) < prevalence).astype(np.float32)
        for study in range(generator.integers(1, max_studies + 1)):
            rows.append(
                {
                    "Image Index": f"{patient:05d}_{study:03d}.png",
                    config.PATIENT_ID_COLUMN: str(patient),
                    "label_vector": vector.copy(),
                    "filepath": Path(f"/nonexistent/{patient:05d}_{study:03d}.png"),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def label_frame():
    return make_label_frame()


@pytest.fixture(scope="session")
def model_cache():
    """
    Lazily built, randomly initialized backbones, shared across the session.

    pretrained=False keeps it offline: the architecture-specific wiring (head
    location, Grad-CAM target) is what breaks on a torchvision upgrade.
    """
    from models import build_model

    cache: dict[str, object] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = build_model(name, pretrained=False).eval()
        return cache[name]

    return get
