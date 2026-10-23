"""Central filesystem paths for the Streamlit commodity dashboard."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

ITEM_DATA_DIRS = {
    "gold": DATA_DIR / "gold",
    "copper": DATA_DIR / "copper",
    "aluminum": DATA_DIR / "aluminum",
    "silver": DATA_DIR / "silver",
}

LEGACY_DEFAULT_DIR = DATA_DIR / "legacy"
LEGACY_SCENARIO_DIR = DATA_DIR / "legacy"

COPPER_PRICE_PATH = ITEM_DATA_DIRS["copper"] / "streamlit_q_data.csv"
GOLD_PRICE_PATH = ITEM_DATA_DIRS["gold"] / "gold_price.csv"
SILVER_PRICE_PATH = ITEM_DATA_DIRS["silver"] / "silver_price.csv"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESEARCH_DIR = PROJECT_ROOT / "research"
SOURCE_DATA_DIR = RESEARCH_DIR / "source_data"
RESEARCH_IMAGES_DIR = RESEARCH_DIR / "images"
MODEL_OUTPUTS_DIR = RESEARCH_DIR / "model_outputs"
PRESENTATIONS_DIR = RESEARCH_DIR / "presentations"
ARCHIVE_VERSIONS_DIR = PROJECT_ROOT / "archive" / "versions"


def legacy_default_path(scenario: str) -> Path:
    """Return a legacy base/high/low CSV path used by versions 1.1-1.5."""
    return LEGACY_DEFAULT_DIR / f"{scenario}.csv"


def legacy_scenario_path(scenario: str, dataset_number: int) -> Path:
    """Return a numbered legacy scenario CSV path used by later old versions."""
    return LEGACY_SCENARIO_DIR / f"{scenario}_{dataset_number}.csv"
