"""Path compatibility helpers for archived, directly runnable app versions."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    COPPER_PRICE_PATH,
    GOLD_PRICE_PATH,
    ITEM_DATA_DIRS,
    LEGACY_SCENARIO_DIR,
    SILVER_PRICE_PATH,
    legacy_default_path,
    legacy_scenario_path,
)

__all__ = [
    "COPPER_PRICE_PATH",
    "GOLD_PRICE_PATH",
    "ITEM_DATA_DIRS",
    "LEGACY_SCENARIO_DIR",
    "SILVER_PRICE_PATH",
    "legacy_default_path",
    "legacy_scenario_path",
]
