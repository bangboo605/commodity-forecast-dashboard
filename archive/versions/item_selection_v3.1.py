# Version 3.1
from copy import deepcopy
from html import escape
import random
from pathlib import Path
import unicodedata

import altair as alt
import pandas as pd
import streamlit as st

from _compat import (
    COPPER_PRICE_PATH,
    GOLD_PRICE_PATH,
    ITEM_DATA_DIRS,
    legacy_scenario_path,
)


st.set_page_config(
    page_title="원자재 가격 대시보드",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ITEMS = ["알루미늄", "전기동", "금"]
ITEM_CODES = {
    "알루미늄": "aluminum",
    "전기동": "copper",
    "금": "gold",
}
SELECTED_ITEM_LABELS = {code: label for label, code in ITEM_CODES.items()}
VARIABLES = ["수요", "공급", "원재료", "투자", "거시경제"]
GOLD_VARIABLES = (
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
)
GOLD_VARIABLE_LABELS = {
    "X1": "X1",
    "X2": "X2",
    "X3": "X3",
    "X4": "X4",
    "X5": "X5",
    "X6": "X6",
}
GOLD_VARIABLE_BUTTON_LABELS = {
    "X1": "X1",
    "X2": "X2",
    "X3": "X3",
    "X4": "X4",
    "X5": "X5",
    "X6": "X6",
}
GOLD_HISTORY_PERIOD_LABELS = {
    "1y": "1년",
    "2y": "2년",
    "3y": "3년",
}
GOLD_PREDICTION_DATES = (
    "2026-01-21",
    "2026-02-21",
    "2026-03-21",
)
ITEM_VARIABLE_CONFIG = {
    "gold": {
        "actual_csv": "gold_major_variables.csv",
        "forecast_csv": "gold_major_variables_forecast.csv",
        "actual_column_aliases": {
            "X1_l1": "X1",
            "X3_l1": "X3",
            "X5_l1": "X5",
            "X6_l1": "X6",
        },
        "labels": GOLD_VARIABLE_BUTTON_LABELS,
        "full_labels": GOLD_VARIABLE_LABELS,
        "prediction_dates": GOLD_PREDICTION_DATES,
    },
    "copper": {
        "actual_csv": "copper_major_variables.csv",
        "forecast_csv": "copper_major_variables_forecast.csv",
        "labels": {
            "demand": "수요",
            "supply": "공급",
            "raw_material": "원재료",
            "investment": "투자",
            "macro_economy": "거시경제",
            "inventory": "재고",
            "exchange_rate": "환율",
        },
        "prediction_dates": GOLD_PREDICTION_DATES,
    },
    "aluminum": {
        "actual_csv": "aluminum_major_variables.csv",
        "forecast_csv": "aluminum_major_variables_forecast.csv",
        "labels": {
            "demand": "수요",
            "supply": "공급",
            "energy_cost": "에너지 비용",
        },
        "prediction_dates": GOLD_PREDICTION_DATES,
    },
}
DATASET_NUMBERS = (1, 2, 3)
FORECAST_COLUMNS = ("90", "L1", "L2", "P1", "P2")
PREDICTION_DATES = [
    "2026-01-21",
    "2026-02-21",
    "2026-03-21",
    "2026-04-21",
    "2026-05-21",
    "2026-06-21",
    "2026-07-21",
    "2026-08-21",
    "2026-09-21",
]


@st.cache_data
def load_scenarios(dataset_number: int) -> dict[str, pd.DataFrame]:
    scenarios = {}
    for name in ("base", "high", "low"):
        path = legacy_scenario_path(name, dataset_number)
        if not path.exists():
            scenarios[name] = pd.DataFrame()
            continue

        for encoding in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                scenarios[name] = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            scenarios[name] = pd.DataFrame()
    return scenarios


@st.cache_data
def load_price_forecast_data() -> pd.DataFrame:
    path = COPPER_PRICE_PATH
    if not path.exists():
        return pd.DataFrame()

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, skiprows=1)
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


@st.cache_data
def load_gold_price_history(
    path_text: str,
    file_modified_ns: int,
) -> pd.DataFrame:
    """Load the gold price CSV while invalidating cache after file edits."""
    del file_modified_ns
    path = Path(path_text)
    data = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = ["분기", "실제가"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "gold_price.csv에 필요한 컬럼이 없습니다: "
            + ", ".join(missing_columns)
        )
    has_forecast_column = "예측가" in data.columns
    selected_columns = required_columns + (["예측가"] if has_forecast_column else [])
    data = data[selected_columns].copy()
    data["분기"] = (
        data["분기"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^(\d{4})Q([1-4])$", r"\1-Q\2", regex=True)
    )
    data = data[data["분기"].str.match(r"^\d{4}-Q[1-4]$")].copy()
    data["실제가"] = pd.to_numeric(data["실제가"], errors="coerce")
    data["예측가"] = (
        pd.to_numeric(data["예측가"], errors="coerce")
        if has_forecast_column
        else pd.NA
    )
    return (
        data.dropna(subset=["분기"])
        .drop_duplicates(subset=["분기"], keep="last")
        .sort_values("분기", key=lambda series: series.map(quarter_key))
        .reset_index(drop=True)
    )


@st.cache_data(max_entries=12)
def load_item_actual_data(path_text: str, file_modified_ns: int) -> pd.DataFrame:
    path = Path(path_text)
    data = pd.read_csv(path, encoding="utf-8-sig")
    if "quarter" not in data.columns:
        raise ValueError(f"{path.name}에 quarter 컬럼이 없습니다.")

    variable_columns = [
        column for column in data.columns if column not in {"quarter", "scenario"}
    ]
    if not variable_columns:
        raise ValueError(f"{path.name}에 주요 변수 컬럼이 없습니다.")

    data = data[["quarter", *variable_columns]].copy().dropna(subset=["quarter"])
    data["quarter"] = (
        data["quarter"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^(\d{4})Q([1-4])$", r"\1-Q\2", regex=True)
    )
    data = data[data["quarter"].str.match(r"^\d{4}-Q[1-4]$")].copy()
    for column in variable_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(
        "quarter", key=lambda series: series.map(quarter_key)
    ).reset_index(drop=True)


@st.cache_data(max_entries=12)
def load_item_forecast_data(
    path_text: str,
    file_modified_ns: int,
    variable_columns: tuple[str, ...],
) -> pd.DataFrame:
    path = Path(path_text)
    data = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = ["quarter", "scenario", *variable_columns]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(
            f"{path.name}에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
        )
    extra_variables = [
        column
        for column in data.columns
        if column not in {"quarter", "scenario", *variable_columns}
    ]
    if extra_variables:
        raise ValueError(
            f"{path.name}의 주요 변수 구성이 실제 데이터와 다릅니다: "
            + ", ".join(extra_variables)
        )

    data = data[required_columns].copy()
    data["quarter"] = (
        data["quarter"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^(\d{4})Q([1-4])$", r"\1-Q\2", regex=True)
    )
    data["scenario"] = data["scenario"].astype(str).str.strip().str.lower()
    data = data[data["scenario"].isin(("high", "base", "low"))].copy()
    for column in variable_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.reset_index(drop=True)


def load_configured_item_data(
    item_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], dict]:
    config = ITEM_VARIABLE_CONFIG[item_code]
    data_root = ITEM_DATA_DIRS[item_code]
    actual_path = data_root / config["actual_csv"]
    forecast_path = data_root / config["forecast_csv"]
    if not actual_path.exists() or not forecast_path.exists():
        missing = [
            path.name for path in (actual_path, forecast_path) if not path.exists()
        ]
        raise FileNotFoundError("필요한 데이터 파일이 없습니다: " + ", ".join(missing))
    actual = load_item_actual_data(str(actual_path), actual_path.stat().st_mtime_ns)
    actual = actual.rename(columns=config.get("actual_column_aliases", {}))
    variables = tuple(column for column in actual.columns if column != "quarter")
    forecast = load_item_forecast_data(
        str(forecast_path),
        forecast_path.stat().st_mtime_ns,
        variables,
    )
    return actual, forecast, variables, config


def quarter_key(label: str) -> tuple[int, int]:
    normalized = str(label).replace("/", "-").upper()
    year_text, quarter_text = normalized.split("-")
    return int(year_text), int(quarter_text.replace("Q", ""))


def quarter_label(label: str) -> str:
    year, quarter = quarter_key(label)
    return f"{year}-Q{quarter}"


def date_to_quarter(date_text: str) -> str:
    date = pd.Timestamp(date_text)
    return f"{date.year}-Q{((date.month - 1) // 3) + 1}"


def five_quarters_from(start_quarter: str) -> list[str]:
    year, quarter = quarter_key(start_quarter)
    quarters = []
    for offset in range(5):
        zero_based = (quarter - 1) + offset
        quarters.append(f"{year + zero_based // 4}-Q{zero_based % 4 + 1}")
    return quarters


def estimate_pill_row_count(
    labels: list[str],
    available_width: int = 310,
) -> int:
    """Estimate wrapped pill rows for the result-card selector."""
    if not labels:
        return 1

    rows = 1
    used_width = 0
    gap = 6
    for label in labels:
        text_width = sum(
            12
            if unicodedata.east_asian_width(character) in {"W", "F", "A"}
            else 7
            for character in str(label)
        )
        pill_width = text_width + 28
        required_width = pill_width if used_width == 0 else gap + pill_width
        if used_width and used_width + required_width > available_width:
            rows += 1
            used_width = pill_width
        else:
            used_width += required_width
    return rows


def shift_quarter(quarter_label_text: str, amount: int) -> str:
    year, quarter = quarter_key(quarter_label_text)
    zero_based = (quarter - 1) + amount
    return f"{year + zero_based // 4}-Q{zero_based % 4 + 1}"


def quarters_between(start_quarter: str, end_quarter: str) -> list[str]:
    """Return an inclusive, chronologically ordered quarter range."""
    start_year, start_number = quarter_key(start_quarter)
    end_year, end_number = quarter_key(end_quarter)
    start_index = start_year * 4 + start_number - 1
    end_index = end_year * 4 + end_number - 1
    return [
        f"{index // 4}-Q{index % 4 + 1}"
        for index in range(start_index, end_index + 1)
    ]


def gold_value_for(
    data: pd.DataFrame,
    quarter: str,
    variable: str,
) -> float | None:
    if variable not in data.columns:
        return None
    matched = data.loc[data["quarter"] == quarter, variable]
    if matched.empty or pd.isna(matched.iloc[0]):
        return None
    return float(matched.iloc[0])


def gold_forecast_value_for(
    data: pd.DataFrame,
    quarter: str,
    scenario: str,
    variable: str,
) -> float | None:
    if variable not in data.columns:
        return None
    matched = data.loc[
        (data["quarter"] == quarter) & (data["scenario"] == scenario),
        variable,
    ]
    if matched.empty or pd.isna(matched.iloc[0]):
        return None
    return float(matched.iloc[0])


def format_gold_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    numeric_value = float(value)
    absolute = abs(numeric_value)
    integer_digits = len(str(int(absolute))) if absolute >= 1 else 1
    decimal_places = max(0, 6 - integer_digits)
    formatted = f"{numeric_value:,.{decimal_places}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def mark_gold_prediction_pending() -> None:
    st.session_state.gold_prediction_confirmed = False
    st.session_state.item_price_stage_visible = False
    st.session_state.item_price_stage_warning = None


def gold_context_key(variable: str, quarter: str | None = None) -> str:
    applied_quarter = quarter or st.session_state.gold_applied_quarter
    item_code = st.session_state.get("selected_item_code", "gold")
    return f"{item_code}|{variable}|{applied_quarter}"


def apply_gold_prediction_date() -> None:
    prediction_date = st.session_state.gold_prediction_candidate
    st.session_state.gold_applied_prediction_date = prediction_date
    st.session_state.gold_applied_quarter = date_to_quarter(prediction_date)
    st.session_state.gold_prediction_confirmed = True
    item_code = st.session_state.get("selected_item_code", "gold")
    _, gold_forecast_data, variables, _ = load_configured_item_data(item_code)
    quarters = five_quarters_from(st.session_state.gold_applied_quarter)
    for variable in variables:
        context_key = gold_context_key(variable)
        values = {
            quarter: gold_forecast_value_for(
                gold_forecast_data,
                quarter,
                "base",
                variable,
            )
            for quarter in quarters
        }
        available = {
            quarter: "base" for quarter, value in values.items() if value is not None
        }
        st.session_state.gold_selected_cells[context_key] = available
        st.session_state.gold_final_values[context_key] = dict(values)
        st.session_state.gold_final_drafts[context_key] = dict(values)
        st.session_state.gold_imported_selections[context_key] = dict(available)
        st.session_state.gold_chart_focus[context_key] = "final"
        st.session_state.gold_final_input_versions[context_key] = (
            st.session_state.gold_final_input_versions.get(context_key, 0) + 1
        )
    st.session_state.gold_active_final_input_context = None
    st.session_state.gold_table_warning = None
    st.session_state.item_price_stage_visible = False
    st.session_state.item_price_stage_warning = None


def prepare_gold_one_click_test() -> None:
    """Randomly commit every variable so the price-prediction screen can be tested quickly."""
    item_code = st.session_state.get("selected_item_code", "gold")
    config = ITEM_VARIABLE_CONFIG[item_code]
    prediction_date = random.choice(config.get("prediction_dates", GOLD_PREDICTION_DATES))
    st.session_state.gold_prediction_candidate = prediction_date
    st.session_state.gold_applied_prediction_date = prediction_date
    st.session_state.gold_applied_quarter = date_to_quarter(prediction_date)
    st.session_state.gold_prediction_confirmed = True

    _, gold_forecast_data, variables, _ = load_configured_item_data(item_code)
    quarters = five_quarters_from(st.session_state.gold_applied_quarter)

    for variable in variables:
        context_key = gold_context_key(variable)
        selected: dict[str, str] = {}
        committed: dict[str, float] = {}
        for quarter in quarters:
            available_scenarios = [
                scenario_name
                for scenario_name in ("high", "base", "low")
                if gold_forecast_value_for(
                    gold_forecast_data, quarter, scenario_name, variable
                )
                is not None
            ]
            if not available_scenarios:
                continue
            scenario_name = random.choice(available_scenarios)
            selected[quarter] = scenario_name
            committed[quarter] = gold_forecast_value_for(
                gold_forecast_data, quarter, scenario_name, variable
            )

        st.session_state.gold_selected_cells[context_key] = selected
        st.session_state.gold_final_values[context_key] = dict(committed)
        st.session_state.gold_final_drafts[context_key] = dict(committed)
        st.session_state.gold_imported_selections[context_key] = dict(selected)
        st.session_state.gold_chart_focus[context_key] = "final"
        st.session_state.gold_final_input_versions[context_key] = (
            st.session_state.gold_final_input_versions.get(context_key, 0) + 1
        )

    st.session_state.gold_active_final_input_context = None
    st.session_state.gold_table_warning = None
    saved_slots = st.session_state.get("saved_forecast_datasets", {})
    if not any(saved_slots.get(slot) for slot in ("데이터셋1", "데이터셋2")):
        st.session_state.item_price_stage_visible = False
    st.session_state.item_price_stage_warning = None
    if variables:
        selected_variable = random.choice(variables)
        st.session_state.gold_variable_selector = selected_variable


def gold_final_input_key(context_key: str, quarter: str) -> str:
    version = st.session_state.gold_final_input_versions.get(context_key, 0)
    return f"gold_final_input|{version}|{context_key}|{quarter}"


def normalize_numeric_input_text(value: object) -> str:
    """Normalize display-only numeric text while preserving nonzero input."""
    normalized = str(value).strip().replace(",", "")
    sign = ""
    unsigned = normalized
    if unsigned.startswith(("+", "-")):
        sign, unsigned = unsigned[0], unsigned[1:]

    integer_part, separator, decimal_part = unsigned.partition(".")
    is_plain_number = (
        integer_part.isdigit()
        and (not separator or decimal_part.isdigit())
    )
    if is_plain_number:
        integer_part = integer_part.lstrip("0") or "0"
        return f"{sign}{integer_part}{separator}{decimal_part}"
    return normalized


def set_gold_chart_focus(context_key: str, series_name: str) -> None:
    current_focus = st.session_state.gold_chart_focus.get(context_key, "final")
    st.session_state.gold_chart_focus[context_key] = (
        "final" if current_focus == series_name else series_name
    )


def select_gold_scenario_cell(
    context_key: str,
    quarter: str,
    scenario_name: str,
    value: float | None,
) -> None:
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.gold_table_warning = "예측시점을 먼저 조회해 주세요."
        return
    if value is None or pd.isna(value):
        st.session_state.gold_table_warning = f"현재 {scenario_name} 데이터는 준비 중입니다."
        return
    st.session_state.gold_selected_cells.setdefault(context_key, {})[quarter] = scenario_name
    st.session_state.gold_table_warning = None


def select_gold_scenario_row(
    context_key: str,
    scenario_name: str,
    quarters: tuple[str, ...],
    values: tuple[float | None, ...],
) -> None:
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.gold_table_warning = "예측시점을 먼저 조회해 주세요."
        return
    available = [
        (quarter, value)
        for quarter, value in zip(quarters, values)
        if value is not None and not pd.isna(value)
    ]
    if not available:
        st.session_state.gold_table_warning = f"현재 {scenario_name} 데이터는 준비 중입니다."
        return
    selected = st.session_state.gold_selected_cells.setdefault(context_key, {})
    for quarter, _ in available:
        selected[quarter] = scenario_name
    set_gold_chart_focus(context_key, scenario_name)
    st.session_state.gold_table_warning = None


def import_gold_selected_values(
    context_key: str,
    quarters: tuple[str, ...],
    scenario_values: dict[str, tuple[float | None, ...]],
) -> None:
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.gold_table_warning = "예측시점을 먼저 조회해 주세요."
        return
    selected = st.session_state.gold_selected_cells.setdefault(context_key, {})
    if any(quarter not in selected for quarter in quarters):
        st.session_state.gold_table_warning = "각 분기에서 값을 하나씩 선택해 주세요."
        return
    values_by_scenario = {
        scenario_name: dict(zip(quarters, values))
        for scenario_name, values in scenario_values.items()
    }
    draft = st.session_state.gold_final_drafts.setdefault(context_key, {})
    imported = st.session_state.gold_imported_selections.setdefault(context_key, {})
    for quarter in quarters:
        scenario_name = selected[quarter]
        value = values_by_scenario.get(scenario_name, {}).get(quarter)
        if value is None or pd.isna(value):
            st.session_state.gold_table_warning = "선택한 데이터 값을 확인해 주세요."
            return
        draft[quarter] = float(value)
        imported[quarter] = scenario_name
        st.session_state[gold_final_input_key(context_key, quarter)] = (
            format_gold_value(value)
        )
    st.session_state.gold_table_warning = "선택한 값을 확인한 뒤 commit해 주세요."


def update_gold_final_draft(context_key: str, quarter: str) -> None:
    input_key = gold_final_input_key(context_key, quarter)
    raw_value = normalize_numeric_input_text(st.session_state.get(input_key, ""))
    st.session_state[input_key] = raw_value
    draft = st.session_state.gold_final_drafts.setdefault(context_key, {})
    st.session_state.item_price_stage_visible = False
    if not raw_value:
        draft[quarter] = None
        st.session_state.gold_table_warning = "final 값을 모두 입력해 주세요."
        return
    try:
        draft[quarter] = float(raw_value)
        st.session_state.gold_table_warning = "변경 내용을 commit해 주세요."
    except ValueError:
        draft[quarter] = None
        st.session_state.gold_table_warning = "숫자 형식으로 입력해 주세요."


def commit_gold_final_draft(context_key: str, quarters: tuple[str, ...]) -> None:
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.gold_table_warning = "예측시점을 먼저 조회해 주세요."
        return
    draft = st.session_state.gold_final_drafts.setdefault(context_key, {})
    if any(draft.get(quarter) is None for quarter in quarters):
        st.session_state.gold_table_warning = "final 값을 모두 입력해 주세요."
        return
    st.session_state.gold_final_values[context_key] = {
        quarter: float(draft[quarter]) for quarter in quarters
    }
    st.session_state.gold_chart_focus[context_key] = "final"
    st.session_state.gold_table_warning = None
    st.session_state.gold_final_commit_notice = True


def incomplete_item_variables(
    variables: tuple[str, ...],
    applied_quarter: str,
) -> list[str]:
    """Return variables whose five final values are missing or not committed."""
    quarters = five_quarters_from(applied_quarter)
    incomplete: list[str] = []
    for variable in variables:
        context_key = gold_context_key(variable, applied_quarter)
        committed = st.session_state.gold_final_values.get(context_key, {})
        draft = st.session_state.gold_final_drafts.get(context_key, {})
        if any(
            committed.get(quarter) is None
            or pd.isna(committed.get(quarter))
            or draft.get(quarter) is None
            or pd.isna(draft.get(quarter))
            or not values_match(draft.get(quarter), committed.get(quarter))
            for quarter in quarters
        ):
            incomplete.append(variable)
    return incomplete


def proceed_to_item_price_stage() -> None:
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.item_price_stage_warning = (
            "예측시점을 먼저 조회해 주세요."
        )
        return

    variables = tuple(st.session_state.get("active_item_variables", ()))
    incomplete = incomplete_item_variables(
        variables,
        st.session_state.gold_applied_quarter,
    )
    if incomplete:
        labels = ITEM_VARIABLE_CONFIG[
            st.session_state.selected_item_code
        ].get("labels", {})
        st.session_state.gold_variable_selector = incomplete[0]
        incomplete_text = ", ".join(
            labels.get(variable, variable) for variable in incomplete
        )
        st.session_state.item_price_stage_warning = (
            f"다음 주요 변수를 commit해 주세요: {incomplete_text}"
        )
        st.session_state.item_price_stage_visible = False
        return

    st.session_state.item_price_stage_warning = None
    st.session_state.item_price_stage_visible = True
    st.session_state.item_price_stage_scroll = True

    # 두 저장 슬롯 중 하나만 비어 있으면 새 예측 결과의 기본 저장 위치로 지정한다.
    saved_slots = st.session_state.get("saved_forecast_datasets", {})
    empty_slots = [
        slot
        for slot in ("데이터셋1", "데이터셋2")
        if saved_slots.get(slot) is None
    ]
    if len(empty_slots) == 1:
        empty_slot = empty_slots[0]
        if st.session_state.get("dataset_save_target") != empty_slot:
            st.session_state.dataset_save_target = empty_slot
            st.session_state.dataset_save_widget_version = (
                st.session_state.get("dataset_save_widget_version", 0) + 1
            )


def enforce_gold_variable_selection() -> None:
    selected = st.session_state.get("gold_variable_selector")
    variables = tuple(st.session_state.get("active_item_variables", ()))
    if variables and selected not in variables:
        st.session_state.gold_variable_selector = variables[0]


def padded_y_domain(
    *value_groups: list[float | None] | tuple[float | None, ...],
) -> tuple[float, float]:
    """Build one stable Y-axis domain for all compared series."""
    values = [
        float(value)
        for group in value_groups
        for value in group
        if value is not None and not pd.isna(value)
    ]
    if not values:
        return ()
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    padding = span * 0.08 if span else max(abs(maximum) * 0.08, 1)
    return minimum - padding, maximum + padding


def values_for_quarters(
    frame: pd.DataFrame,
    variable: str,
    quarters: list[str],
) -> list[float | None]:
    if frame.empty or variable not in frame.columns:
        return [None] * len(quarters)

    prepared = frame.copy()
    prepared["_quarter"] = prepared["Date"].astype(str).map(quarter_label)
    return [
        prepared.loc[prepared["_quarter"] == quarter, variable].iloc[0]
        if (prepared["_quarter"] == quarter).any()
        else None
        for quarter in quarters
    ]


@st.cache_data(show_spinner=False)
def create_trend_chart(
    variable: str,
    historical_quarters: tuple[str, ...],
    historical_values: tuple[float | None, ...],
    forecast_quarters: tuple[str, ...],
    base_values: tuple[float | None, ...],
    high_values: tuple[float | None, ...],
    low_values: tuple[float | None, ...],
    final_values: tuple[float | None, ...],
    chart_focus: str,
    show_title: bool = True,
    chart_height: int = 330,
    saved_final_values: tuple[float | None, ...] = (),
    dataset_focus: tuple[str, ...] = (),
    saved_dataset2_values: tuple[float | None, ...] = (),
    saved_forecast_quarters: tuple[str, ...] = (),
    saved_dataset2_quarters: tuple[str, ...] = (),
    fixed_axis_quarters: tuple[str, ...] = (),
    fixed_y_domain: tuple[float, float] = (),
    muted_historical_quarters: tuple[str, ...] = (),
    muted_historical_values: tuple[float | None, ...] = (),
    show_forecast_marker: bool = True,
    use_dataset_focus: bool = False,
) -> alt.Chart:
    focused_datasets = set(dataset_focus)
    saved_forecast_quarters = saved_forecast_quarters or forecast_quarters
    saved_dataset2_quarters = saved_dataset2_quarters or forecast_quarters
    axis_quarters = list(fixed_axis_quarters) or sorted(
        set(
            historical_quarters
            + forecast_quarters
            + saved_forecast_quarters
            + saved_dataset2_quarters
        ),
        key=quarter_key,
    )
    axis_label_quarters = (
        axis_quarters[::2]
        if len(historical_quarters) >= 8
        else axis_quarters
    )
    historical_data = pd.DataFrame(
        {"분기": historical_quarters, "값": historical_values}
    ).dropna(subset=["값"])
    muted_historical_data = pd.DataFrame(
        {
            "분기": muted_historical_quarters,
            "값": muted_historical_values,
        }
    ).dropna(subset=["값"])
    connector_history = (
        pd.concat(
            [historical_data, muted_historical_data],
            ignore_index=True,
        )
        .drop_duplicates(subset=["분기"], keep="first")
    )
    final_data = pd.DataFrame(
        {"분기": forecast_quarters, "값": final_values}
    ).dropna(subset=["값"])
    saved_final_data = pd.DataFrame(
        {
            "분기": saved_forecast_quarters,
            "값": (
                saved_final_values
                if saved_final_values
                else tuple(None for _ in saved_forecast_quarters)
            ),
        }
    ).dropna(subset=["값"])
    saved_dataset2_data = pd.DataFrame(
        {
            "분기": saved_dataset2_quarters,
            "값": (
                saved_dataset2_values
                if saved_dataset2_values
                else tuple(None for _ in saved_dataset2_quarters)
            ),
        }
    ).dropna(subset=["값"])
    comparison_data = pd.concat(
        [
            pd.DataFrame(
                {"분기": forecast_quarters, "값": values, "시나리오": scenario_name}
            )
            for scenario_name, values in (
                ("base", base_values),
                ("high", high_values),
                ("low", low_values),
            )
        ],
        ignore_index=True,
    ).dropna(subset=["값"])
    if not historical_data.empty and not comparison_data.empty:
        last_historical = historical_data.iloc[-1]
        connector_rows = pd.DataFrame(
            [
                {
                    "분기": last_historical["분기"],
                    "값": last_historical["값"],
                    "시나리오": scenario_name,
                }
                for scenario_name in ("base", "high", "low")
            ]
        )
        comparison_data = pd.concat(
            [connector_rows, comparison_data],
            ignore_index=True,
        )

    x_encoding = alt.X(
        "분기:N",
        title=None,
        sort=axis_quarters,
        scale=alt.Scale(domain=axis_quarters),
        axis=alt.Axis(
            values=axis_label_quarters,
            labelAngle=-30,
            labelOverlap=False,
            grid=False,
        ),
    )
    y_encoding = alt.Y(
        "값:Q",
        title=None,
        scale=(
            alt.Scale(domain=list(fixed_y_domain), zero=False)
            if fixed_y_domain
            else alt.Scale(zero=False, padding=25)
        ),
        axis=alt.Axis(
            format=",.0f",
            grid=True,
            gridDash=[2, 3],
        ),
    )
    historical_line = (
        alt.Chart(historical_data)
        .mark_line(
            color="#263238",
            strokeWidth=2.5,
        )
        .encode(
            x=x_encoding,
            y=y_encoding,
            tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
        )
    )
    chart_layers = []
    if not muted_historical_data.empty:
        chart_layers.append(
            alt.Chart(muted_historical_data)
            .mark_line(
                color="#263238",
                strokeWidth=2.2,
                opacity=.18,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
            )
        )
    if not comparison_data.empty:
        chart_layers.append(
            alt.Chart(comparison_data)
            .mark_line(
                point=alt.OverlayMarkDef(size=42),
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                color=alt.Color(
                    "시나리오:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["base", "high", "low"],
                        range=["#2563eb", "#c026d3", "#eab308"],
                    ),
                    legend=None,
                ),
                detail="시나리오:N",
                opacity=alt.condition(
                    alt.datum["시나리오"] == chart_focus,
                    alt.value(.95),
                    alt.value(.25),
                ),
                strokeWidth=alt.condition(
                    alt.datum["시나리오"] == chart_focus,
                    alt.value(3),
                    alt.value(1.6),
                ),
                tooltip=[
                    "시나리오:N",
                    "분기:N",
                    alt.Tooltip("값:Q", format=","),
                ],
            )
        )
    chart_layers.append(historical_line)
    if not saved_final_data.empty:
        saved_connector_data = saved_final_data.copy()
        if not connector_history.empty:
            connector = connector_history[
                connector_history["분기"]
                == shift_quarter(saved_forecast_quarters[0], -1)
            ]
            if not connector.empty:
                saved_connector_data = pd.concat(
                    [connector.tail(1), saved_connector_data],
                    ignore_index=True,
                )
        saved_is_focused = "데이터셋1" in focused_datasets
        chart_layers.extend(
            [
                alt.Chart(saved_connector_data)
                .mark_line(
                    color="#2563EB",
                    strokeWidth=3 if saved_is_focused else 1.7,
                    strokeDash=[4, 4],
                    opacity=.95 if saved_is_focused else .23,
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
                alt.Chart(saved_final_data)
                .mark_circle(
                    color="#2563EB",
                    fill="#2563EB",
                    stroke="white",
                    size=82 if saved_is_focused else 42,
                    strokeWidth=1.3,
                    opacity=.95 if saved_is_focused else .25,
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
            ]
        )
    if not saved_dataset2_data.empty:
        dataset2_connector_data = saved_dataset2_data.copy()
        if not connector_history.empty:
            connector = connector_history[
                connector_history["분기"]
                == shift_quarter(saved_dataset2_quarters[0], -1)
            ]
            if not connector.empty:
                dataset2_connector_data = pd.concat(
                    [connector.tail(1), dataset2_connector_data],
                    ignore_index=True,
                )
        dataset2_is_focused = "데이터셋2" in focused_datasets
        chart_layers.extend(
            [
                alt.Chart(dataset2_connector_data)
                .mark_line(
                    color="#CA8A04",
                    strokeWidth=3 if dataset2_is_focused else 1.7,
                    strokeDash=[2, 4],
                    opacity=.95 if dataset2_is_focused else .23,
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
                alt.Chart(saved_dataset2_data)
                .mark_circle(
                    color="#CA8A04",
                    fill="#CA8A04",
                    stroke="white",
                    size=82 if dataset2_is_focused else 42,
                    strokeWidth=1.3,
                    opacity=.95 if dataset2_is_focused else .25,
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
            ]
        )
    if not final_data.empty:
        connector_data = final_data.copy()
        if not connector_history.empty:
            connector = connector_history[
                connector_history["분기"]
                == shift_quarter(forecast_quarters[0], -1)
            ]
            if not connector.empty:
                connector_data = pd.concat(
                    [connector.tail(1), connector_data],
                    ignore_index=True,
                )
        current_is_focused = (
            "현재 그래프" in focused_datasets
            if use_dataset_focus
            else True
        )
        chart_layers.extend(
            [
                alt.Chart(connector_data)
                .mark_line(
                    color="#ef6461",
                    strokeWidth=(
                        3
                        if chart_focus == "final" and current_is_focused
                        else 1.7
                    ),
                    strokeDash=[7, 5],
                    opacity=(
                        1
                        if chart_focus == "final" and current_is_focused
                        else .25
                    ),
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
                alt.Chart(final_data)
                .mark_circle(
                    color="#ef6461",
                    fill="#ef6461",
                    stroke="white",
                    size=(
                        85
                        if chart_focus == "final" and current_is_focused
                        else 45
                    ),
                    strokeWidth=1.5,
                    opacity=(
                        1
                        if chart_focus == "final" and current_is_focused
                        else .3
                    ),
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
            ]
        )

    saved_origins = []
    if not saved_final_data.empty:
        saved_origins.append(
            {"분기": saved_forecast_quarters[0], "데이터셋": "데이터셋1"}
        )
    if not saved_dataset2_data.empty:
        saved_origins.append(
            {"분기": saved_dataset2_quarters[0], "데이터셋": "데이터셋2"}
        )
    if saved_origins:
        origin_data = pd.DataFrame(saved_origins)
        origin_color = alt.Color(
            "데이터셋:N",
            scale=alt.Scale(
                domain=["데이터셋1", "데이터셋2"],
                range=["#2563EB", "#CA8A04"],
            ),
            legend=None,
        )
        chart_layers.extend(
            [
                alt.Chart(origin_data)
                .mark_rule(
                    strokeDash=[2, 4],
                    opacity=.48 if show_forecast_marker else 0,
                )
                .encode(x=x_encoding, color=origin_color),
                alt.Chart(origin_data)
                .mark_text(
                    fontSize=9,
                    baseline="top",
                    align="left",
                    dx=5,
                    opacity=1 if show_forecast_marker else 0,
                )
                .encode(
                    x=x_encoding,
                    y=alt.value(17),
                    text="데이터셋:N",
                    color=origin_color,
                ),
            ]
        )

    chart_layers.append(
        alt.Chart(pd.DataFrame({"분기": [forecast_quarters[0]]}))
        .mark_rule(
            color="#ef6461",
            strokeDash=[3, 3],
            opacity=.7 if show_forecast_marker else 0,
        )
        .encode(x=x_encoding)
    )
    chart_layers.append(
        alt.Chart(
            pd.DataFrame(
                {"분기": [forecast_quarters[0]], "라벨": ["예측시점"]}
            )
        )
        .mark_text(
            color="#ef6461",
            fontSize=10,
            fontWeight=600,
            baseline="top",
            align="left",
            dx=7,
            opacity=1 if show_forecast_marker else 0,
        )
        .encode(
            x=x_encoding,
            y=alt.value(5),
            text="라벨:N",
        )
    )
    chart = alt.layer(*chart_layers).properties(height=chart_height)
    if show_title:
        chart = chart.properties(
            title=alt.TitleParams(
                text=f"{variable} 추이",
                anchor="middle",
                frame="bounds",
            )
        )
    return (
        chart
        .configure_view(strokeWidth=0)
        .configure_title(anchor="middle", fontSize=17, color="#172033")
    )


@st.cache_data(show_spinner=False)
def create_price_forecast_chart(
    item_name: str,
    historical_quarters: tuple[str, ...],
    historical_values: tuple[float | None, ...],
    forecast_quarters: tuple[str, ...],
    forecast_values: tuple[float | None, ...],
    show_title: bool = True,
    chart_height: int = 330,
    saved_forecast_values: tuple[float | None, ...] = (),
    dataset_focus: tuple[str, ...] = (),
    saved_dataset2_values: tuple[float | None, ...] = (),
    saved_forecast_quarters: tuple[str, ...] = (),
    saved_dataset2_quarters: tuple[str, ...] = (),
    fixed_axis_quarters: tuple[str, ...] = (),
    fixed_y_domain: tuple[float, float] = (),
    muted_historical_quarters: tuple[str, ...] = (),
    muted_historical_values: tuple[float | None, ...] = (),
    show_forecast_marker: bool = True,
) -> alt.Chart:
    focused_datasets = set(dataset_focus)
    saved_forecast_quarters = saved_forecast_quarters or forecast_quarters
    saved_dataset2_quarters = saved_dataset2_quarters or forecast_quarters
    axis_quarters = list(fixed_axis_quarters) or sorted(
        set(
            historical_quarters
            + forecast_quarters
            + saved_forecast_quarters
            + saved_dataset2_quarters
        ),
        key=quarter_key,
    )
    axis_label_quarters = (
        axis_quarters[::2]
        if len(historical_quarters) >= 8
        else axis_quarters
    )
    historical_data = pd.DataFrame(
        {"분기": historical_quarters, "값": historical_values}
    ).dropna(subset=["값"])
    muted_historical_data = pd.DataFrame(
        {
            "분기": muted_historical_quarters,
            "값": muted_historical_values,
        }
    ).dropna(subset=["값"])
    connector_history = (
        pd.concat(
            [historical_data, muted_historical_data],
            ignore_index=True,
        )
        .drop_duplicates(subset=["분기"], keep="first")
    )
    forecast_data = pd.DataFrame(
        {"분기": forecast_quarters, "값": forecast_values}
    ).dropna(subset=["값"])
    saved_forecast_data = pd.DataFrame(
        {
            "분기": saved_forecast_quarters,
            "값": (
                saved_forecast_values
                if saved_forecast_values
                else tuple(None for _ in saved_forecast_quarters)
            ),
        }
    ).dropna(subset=["값"])
    saved_dataset2_data = pd.DataFrame(
        {
            "분기": saved_dataset2_quarters,
            "값": (
                saved_dataset2_values
                if saved_dataset2_values
                else tuple(None for _ in saved_dataset2_quarters)
            ),
        }
    ).dropna(subset=["값"])
    connector_data = forecast_data.copy()
    if not connector_history.empty:
        connector = connector_history[
            connector_history["분기"]
            == shift_quarter(forecast_quarters[0], -1)
        ]
        if not connector.empty:
            connector_data = pd.concat(
                [connector.tail(1), forecast_data],
                ignore_index=True,
            )

    x_encoding = alt.X(
        "분기:N",
        title=None,
        sort=axis_quarters,
        scale=alt.Scale(domain=axis_quarters),
        axis=alt.Axis(
            values=axis_label_quarters,
            labelAngle=-30,
            labelOverlap=False,
            grid=False,
        ),
    )
    y_encoding = alt.Y(
        "값:Q",
        title=None,
        scale=(
            alt.Scale(domain=list(fixed_y_domain), zero=False)
            if fixed_y_domain
            else alt.Scale(zero=False, padding=25)
        ),
        axis=alt.Axis(format=",.0f", grid=True, gridDash=[2, 3]),
    )
    current_is_focused = "현재 그래프" in focused_datasets
    layers = []
    if not muted_historical_data.empty:
        layers.append(
            alt.Chart(muted_historical_data)
            .mark_line(
                color="#263238",
                strokeWidth=2.2,
                opacity=.18,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
            )
        )
    layers.extend([
        alt.Chart(historical_data)
        .mark_line(
            color="#263238",
            strokeWidth=2.5,
        )
        .encode(
            x=x_encoding,
            y=y_encoding,
            tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
        ),
        alt.Chart(connector_data)
        .mark_line(
            color="#ef6461",
            strokeWidth=3 if current_is_focused else 1.7,
            strokeDash=[7, 5],
            opacity=1 if current_is_focused else .25,
        )
        .encode(
            x=x_encoding,
            y=y_encoding,
            tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
        ),
        alt.Chart(forecast_data)
        .mark_circle(
            color="#ef6461",
            fill="#ef6461",
            stroke="white",
            strokeWidth=1.5,
            size=85 if current_is_focused else 45,
            opacity=1 if current_is_focused else .3,
        )
        .encode(
            x=x_encoding,
            y=y_encoding,
            tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
        ),
        alt.Chart(pd.DataFrame({"분기": [forecast_quarters[0]]}))
        .mark_rule(
            color="#ef6461",
            strokeDash=[3, 3],
            opacity=.7 if show_forecast_marker else 0,
        )
        .encode(x=x_encoding),
        alt.Chart(
            pd.DataFrame(
                {"분기": [forecast_quarters[0]], "라벨": ["예측시점"]}
            )
        )
        .mark_text(
            color="#ef6461",
            fontSize=10,
            fontWeight=600,
            baseline="top",
            align="left",
            dx=7,
            opacity=1 if show_forecast_marker else 0,
        )
        .encode(
            x=x_encoding,
            y=alt.value(5),
            text="라벨:N",
        ),
    ])
    saved_origins = []
    if not saved_forecast_data.empty:
        saved_origins.append(
            {"분기": saved_forecast_quarters[0], "데이터셋": "데이터셋1"}
        )
    if not saved_dataset2_data.empty:
        saved_origins.append(
            {"분기": saved_dataset2_quarters[0], "데이터셋": "데이터셋2"}
        )
    if saved_origins:
        origin_data = pd.DataFrame(saved_origins)
        origin_color = alt.Color(
            "데이터셋:N",
            scale=alt.Scale(
                domain=["데이터셋1", "데이터셋2"],
                range=["#2563EB", "#CA8A04"],
            ),
            legend=None,
        )
        layers.extend(
            [
                alt.Chart(origin_data)
                .mark_rule(
                    strokeDash=[2, 4],
                    opacity=.48 if show_forecast_marker else 0,
                )
                .encode(x=x_encoding, color=origin_color),
                alt.Chart(origin_data)
                .mark_text(
                    fontSize=9,
                    baseline="top",
                    align="left",
                    dx=5,
                    opacity=1 if show_forecast_marker else 0,
                )
                .encode(
                    x=x_encoding,
                    y=alt.value(17),
                    text="데이터셋:N",
                    color=origin_color,
                ),
            ]
        )
    if not saved_forecast_data.empty:
        saved_connector_data = saved_forecast_data.copy()
        if not connector_history.empty:
            connector = connector_history[
                connector_history["분기"]
                == shift_quarter(saved_forecast_quarters[0], -1)
            ]
            if not connector.empty:
                saved_connector_data = pd.concat(
                    [connector.tail(1), saved_connector_data],
                    ignore_index=True,
                )
        saved_is_focused = "데이터셋1" in focused_datasets
        layers[1:1] = [
            alt.Chart(saved_connector_data)
            .mark_line(
                color="#2563EB",
                strokeWidth=3 if saved_is_focused else 1.7,
                strokeDash=[4, 4],
                opacity=.95 if saved_is_focused else .23,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
            ),
            alt.Chart(saved_forecast_data)
            .mark_circle(
                color="#2563EB",
                fill="#2563EB",
                stroke="white",
                strokeWidth=1.3,
                size=82 if saved_is_focused else 42,
                opacity=.95 if saved_is_focused else .25,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
            ),
        ]
    if not saved_dataset2_data.empty:
        dataset2_connector_data = saved_dataset2_data.copy()
        if not connector_history.empty:
            connector = connector_history[
                connector_history["분기"]
                == shift_quarter(saved_dataset2_quarters[0], -1)
            ]
            if not connector.empty:
                dataset2_connector_data = pd.concat(
                    [connector.tail(1), dataset2_connector_data],
                    ignore_index=True,
                )
        dataset2_is_focused = "데이터셋2" in focused_datasets
        layers[1:1] = [
            alt.Chart(dataset2_connector_data)
            .mark_line(
                color="#CA8A04",
                strokeWidth=3 if dataset2_is_focused else 1.7,
                strokeDash=[2, 4],
                opacity=.95 if dataset2_is_focused else .23,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
            ),
            alt.Chart(saved_dataset2_data)
            .mark_circle(
                color="#CA8A04",
                fill="#CA8A04",
                stroke="white",
                strokeWidth=1.3,
                size=82 if dataset2_is_focused else 42,
                opacity=.95 if dataset2_is_focused else .25,
            )
            .encode(
                x=x_encoding,
                y=y_encoding,
                tooltip=["분기:N", alt.Tooltip("값:Q", format=",.2f")],
            ),
        ]
    chart = alt.layer(*layers).properties(height=chart_height)
    if show_title:
        chart = chart.properties(
            title=alt.TitleParams(
                text=f"{item_name} 가격 예측",
                anchor="middle",
                frame="bounds",
            )
        )
    return (
        chart
        .configure_view(strokeWidth=0)
        .configure_title(anchor="middle", fontSize=17, color="#172033")
    )


def update_result_visibility_after_edit() -> None:
    """Keep comparison results visible once a baseline dataset is saved."""
    saved_slots = st.session_state.get("saved_forecast_datasets", {})
    has_saved_dataset = any(
        dataset is not None for dataset in saved_slots.values()
    ) or st.session_state.get("saved_forecast_dataset") is not None
    st.session_state.show_next_stage = has_saved_dataset
    if has_saved_dataset:
        st.session_state.scroll_to_next_stage = False


def select_scenario_cell(context_key: str, quarter: str, scenario_name: str) -> None:
    if not st.session_state.prediction_confirmed:
        st.session_state.table_warning = "예측시점을 먼저 조회해 주세요."
        return

    selected = st.session_state.selected_cells.setdefault(context_key, {})
    selected[quarter] = scenario_name
    st.session_state.table_warning = None
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None
    st.session_state.acknowledged_selection_warning = None


def set_chart_focus(context_key: str, series_name: str) -> None:
    current_focus = st.session_state.chart_focus.get(context_key, "final")
    st.session_state.chart_focus[context_key] = (
        "final" if current_focus == series_name else series_name
    )


def select_scenario_row(
    context_key: str,
    scenario_name: str,
    quarters: tuple[str, ...],
) -> None:
    if not st.session_state.prediction_confirmed:
        st.session_state.table_warning = "예측시점을 먼저 조회해 주세요."
        return

    selected = st.session_state.selected_cells.setdefault(context_key, {})
    for quarter in quarters:
        selected[quarter] = scenario_name
    st.session_state.table_warning = None
    set_chart_focus(context_key, scenario_name)
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None
    st.session_state.acknowledged_selection_warning = None


def final_input_key(context_key: str, quarter: str) -> str:
    input_version = st.session_state.get("final_input_versions", {}).get(
        context_key,
        0,
    )
    return f"final_input|{input_version}|{context_key}|{quarter}"


def warn_prediction_required() -> None:
    st.session_state.table_warning = "예측시점을 먼저 조회해 주세요."


def values_match(left: float | None, right: float | None) -> bool:
    left_missing = left is None or pd.isna(left)
    right_missing = right is None or pd.isna(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return float(left) == float(right)


def dataset_number_for(variable: str) -> int:
    return st.session_state.variable_dataset_choices.get(variable, 1)


def context_key_for(variable: str) -> str:
    return (
        f"{st.session_state.selected_item}|"
        f"{variable}|"
        f"세트{dataset_number_for(variable)}|"
        f"{st.session_state.applied_quarter}"
    )


def snapshot_final_values(
    snapshot: dict | None,
    variable: str,
    quarters: list[str],
) -> tuple[float | None, ...]:
    if snapshot is None:
        return tuple(None for _ in quarters)
    dataset_number = snapshot.get("variable_choices", {}).get(variable, 1)
    snapshot_context_key = (
        f"{snapshot.get('item', '전기동')}|"
        f"{variable}|"
        f"세트{dataset_number}|"
        f"{snapshot.get('quarter', '')}"
    )
    committed = snapshot.get("final_values", {}).get(
        snapshot_context_key,
        {},
    )
    return tuple(committed.get(quarter) for quarter in quarters)


def dataset_display_name(slot: str) -> str:
    return st.session_state.dataset_display_names.get(slot, slot)


def sync_dataset_save_target(widget_key: str) -> None:
    st.session_state.dataset_save_target = st.session_state[widget_key]


def sync_comparison_focus() -> None:
    st.session_state.comparison_dataset_toggle = [
        *st.session_state.get("comparison_current_focus", []),
        *st.session_state.get("comparison_saved_focus", []),
    ]


def set_comparison_focus(focuses: list[str]) -> None:
    st.session_state.comparison_dataset_toggle = focuses
    st.session_state.comparison_current_focus = [
        focus for focus in focuses if focus == "현재 그래프"
    ]
    st.session_state.comparison_saved_focus = [
        focus
        for focus in focuses
        if focus in ("데이터셋1", "데이터셋2")
    ]


def save_forecast_dataset(snapshot: dict) -> None:
    save_target = st.session_state.dataset_save_target
    st.session_state.saved_forecast_datasets[save_target] = deepcopy(snapshot)
    if save_target == "데이터셋1":
        st.session_state.saved_forecast_dataset = deepcopy(snapshot)
    set_comparison_focus(["현재 그래프"])
    st.session_state.dataset_save_notice = save_target


def apply_item_dataset_names() -> None:
    name1 = st.session_state.dataset_name_input_1.strip()
    name2 = st.session_state.dataset_name_input_2.strip()
    if not name1 or not name2:
        st.session_state.dataset_manage_notice = (
            "데이터셋 이름을 모두 입력해 주세요."
        )
    elif name1 == name2:
        st.session_state.dataset_manage_notice = (
            "서로 다른 데이터셋 이름을 입력해 주세요."
        )
    else:
        st.session_state.dataset_display_names = {
            "데이터셋1": name1,
            "데이터셋2": name2,
        }
        st.session_state.dataset_save_widget_version += 1
        st.session_state.dataset_manage_notice = (
            "데이터셋 이름을 변경했습니다."
        )


def delete_item_result_dataset(slot: str) -> None:
    st.session_state.saved_forecast_datasets[slot] = None
    if slot == "데이터셋1":
        st.session_state.saved_forecast_dataset = None
    st.session_state.dataset_save_notice = False
    set_comparison_focus(
        [
            focus
            for focus in st.session_state.comparison_dataset_toggle
            if focus != slot
        ]
    )
    st.session_state.dataset_manage_notice = (
        f"{dataset_display_name(slot)} 내용을 삭제했습니다."
    )


def import_selected_values(
    context_key: str,
    quarters: list[str],
    scenario_table: pd.DataFrame,
) -> None:
    if not st.session_state.prediction_confirmed:
        warn_prediction_required()
        return

    selected = st.session_state.selected_cells.setdefault(context_key, {})
    selected_quarters = [quarter for quarter in quarters if quarter in selected]
    if not selected_quarters:
        st.session_state.table_warning = "가져올 값을 선택해 주세요."
        return

    draft = st.session_state.final_drafts.setdefault(context_key, {})
    imported_selections = st.session_state.imported_selections.setdefault(context_key, {})
    imported_values = st.session_state.imported_values.setdefault(context_key, {})
    for quarter in selected_quarters:
        scenario_name = selected[quarter]
        value = float(scenario_table.loc[scenario_name, quarter])
        draft[quarter] = value
        imported_selections[quarter] = scenario_name
        imported_values[quarter] = value
        st.session_state[final_input_key(context_key, quarter)] = f"{value:.0f}"
    st.session_state.table_warning = "선택한 값을 확인한 뒤 commit해 주세요."
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None


def update_final_draft(context_key: str, quarter: str) -> None:
    input_key = final_input_key(context_key, quarter)
    normalized_value = normalize_numeric_input_text(st.session_state[input_key])
    st.session_state[input_key] = normalized_value
    is_empty = normalized_value == ""
    is_invalid = False

    if is_empty:
        current_value = None
    else:
        try:
            current_value = float(normalized_value)
        except ValueError:
            current_value = None
            is_invalid = True

    if is_empty:
        st.session_state[input_key] = ""

    st.session_state.final_drafts.setdefault(context_key, {})[quarter] = current_value

    imported_value = st.session_state.imported_values.get(context_key, {}).get(quarter)
    imported_scenario = st.session_state.imported_selections.get(context_key, {}).get(quarter)
    selected = st.session_state.selected_cells.setdefault(context_key, {})
    if imported_scenario is not None and values_match(current_value, imported_value):
        selected[quarter] = imported_scenario
    else:
        selected.pop(quarter, None)

    if is_invalid:
        st.session_state.table_warning = "숫자를 입력해 주세요."
    elif is_empty:
        st.session_state.table_warning = "선택한 값을 확인한 뒤 commit해 주세요."
    else:
        st.session_state.table_warning = "변경 내용을 commit해 주세요."
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None


def commit_final_draft(context_key: str, quarters: list[str]) -> None:
    if not st.session_state.prediction_confirmed:
        warn_prediction_required()
        return

    draft = st.session_state.final_drafts.setdefault(context_key, {})
    values = {quarter: draft.get(quarter) for quarter in quarters}
    if not all(
        value is not None and not pd.isna(value)
        for value in values.values()
    ):
        st.session_state.table_warning = "final 값을 모두 입력해 주세요."
        return

    st.session_state.final_values[context_key] = values.copy()
    st.session_state.chart_focus[context_key] = "final"
    st.session_state.table_warning = None
    st.session_state.final_commit_notice = True
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None


def build_scenario_table(
    scenarios: dict[str, pd.DataFrame],
    variable: str,
    start_quarter: str,
) -> pd.DataFrame:
    selected_labels = five_quarters_from(start_quarter)

    rows = []
    for scenario_name in ("base", "high", "low"):
        frame = scenarios[scenario_name].copy()
        if frame.empty or variable not in frame.columns:
            values = [None] * len(selected_labels)
        else:
            frame["_quarter"] = frame["Date"].astype(str).map(quarter_label)
            values = [
                frame.loc[frame["_quarter"] == label, variable].iloc[0]
                if (frame["_quarter"] == label).any()
                else None
                for label in selected_labels
            ]
        rows.append([scenario_name, *values])

    return pd.DataFrame(rows, columns=["시나리오", *selected_labels]).set_index("시나리오")


def commit_base_defaults_for_all_variables() -> None:
    forecast_quarters = five_quarters_from(st.session_state.applied_quarter)

    for variable in VARIABLES:
        dataset_number = dataset_number_for(variable)
        context_key = context_key_for(variable)
        scenario_table = build_scenario_table(
            scenario_sets[dataset_number],
            variable,
            st.session_state.applied_quarter,
        )
        selected: dict[str, str] = {}
        committed: dict[str, float] = {}

        for quarter in forecast_quarters:
            value = scenario_table.loc["base", quarter]
            if pd.isna(value):
                continue
            numeric_value = float(value)
            selected[quarter] = "base"
            committed[quarter] = numeric_value

        st.session_state.final_input_versions[context_key] = (
            st.session_state.final_input_versions.get(context_key, 0) + 1
        )
        for quarter, value in committed.items():
            st.session_state[final_input_key(context_key, quarter)] = f"{value:.0f}"

        st.session_state.selected_cells[context_key] = selected.copy()
        st.session_state.imported_selections[context_key] = selected.copy()
        st.session_state.imported_values[context_key] = committed.copy()
        st.session_state.final_drafts[context_key] = committed.copy()
        st.session_state.final_values[context_key] = committed.copy()
        st.session_state.chart_focus[context_key] = "final"

    st.session_state.active_final_input_context = None
    st.session_state.table_warning = None
    st.session_state.next_stage_warning = None
    st.session_state.acknowledged_selection_warning = None
    st.session_state.final_commit_notice = False
    st.session_state.forecast_column = None


def apply_prediction_date() -> None:
    prediction_date = st.session_state.prediction_candidate
    st.session_state.applied_prediction_date = prediction_date
    st.session_state.applied_quarter = date_to_quarter(prediction_date)
    st.session_state.prediction_confirmed = True
    commit_base_defaults_for_all_variables()

    default_variable = VARIABLES[0]
    st.session_state.selected_variable = default_variable
    st.session_state.variable_selector = default_variable
    st.session_state.dataset_selector = (
        f"{default_variable}{dataset_number_for(default_variable)}"
    )


def variable_commit_is_complete(variable: str) -> bool:
    """Return whether final values are fully committed, ignoring table-only selections."""
    context_key = context_key_for(variable)
    quarters = five_quarters_from(st.session_state.applied_quarter)
    committed = st.session_state.final_values.get(context_key, {})
    if not all(
        quarter in committed and not pd.isna(committed[quarter])
        for quarter in quarters
    ):
        return False

    draft = st.session_state.final_drafts.get(context_key, {})
    return not draft or all(
        quarter in draft
        and values_match(draft.get(quarter), committed.get(quarter))
        for quarter in quarters
    )


def selection_mismatch_signature() -> tuple[tuple[str, str, str], ...]:
    """Describe scenario selections that differ from already committed final values."""
    mismatches: list[tuple[str, str, str]] = []
    quarters = five_quarters_from(st.session_state.applied_quarter)

    for variable in VARIABLES:
        context_key = context_key_for(variable)
        committed = st.session_state.final_values.get(context_key, {})
        selected = st.session_state.selected_cells.get(context_key, {})
        if not selected or not variable_commit_is_complete(variable):
            continue

        scenario_table = build_scenario_table(
            scenario_sets[dataset_number_for(variable)],
            variable,
            st.session_state.applied_quarter,
        )
        for quarter, scenario_name in selected.items():
            if (
                quarter in quarters
                and scenario_name in scenario_table.index
                and not values_match(
                    scenario_table.loc[scenario_name, quarter],
                    committed.get(quarter),
                )
            ):
                mismatches.append((variable, quarter, scenario_name))

    return tuple(mismatches)


def uncommitted_variable_labels() -> list[tuple[str, str]]:
    return [
        (variable, f"{variable}{dataset_number_for(variable)}")
        for variable in VARIABLES
        if not variable_commit_is_complete(variable)
    ]


def proceed_to_next_stage() -> None:
    incomplete = uncommitted_variable_labels()
    if incomplete:
        st.session_state.acknowledged_selection_warning = None
        first_variable = incomplete[0][0]
        st.session_state.selected_variable = first_variable
        st.session_state.variable_selector = first_variable
        st.session_state.dataset_selector = (
            f"{first_variable}{dataset_number_for(first_variable)}"
        )
        labels = ", ".join(label for _, label in incomplete)
        st.session_state.next_stage_warning = (
            f"다음 주요 변수를 확인하고 commit해 주세요: {labels}"
        )
        update_result_visibility_after_edit()
        st.session_state.scroll_to_next_stage = False
        st.session_state.scroll_to_variable_settings = True
        return

    mismatch_signature = selection_mismatch_signature()
    if (
        mismatch_signature
        and st.session_state.acknowledged_selection_warning != mismatch_signature
    ):
        mismatch_variables = [
            variable
            for variable in VARIABLES
            if any(item[0] == variable for item in mismatch_signature)
        ]
        first_variable = mismatch_variables[0]
        st.session_state.selected_variable = first_variable
        st.session_state.variable_selector = first_variable
        st.session_state.dataset_selector = (
            f"{first_variable}{dataset_number_for(first_variable)}"
        )
        labels = ", ".join(
            f"{variable}{dataset_number_for(variable)}"
            for variable in mismatch_variables
        )
        st.session_state.next_stage_warning = (
            f"선택값이 final과 다른 주요 변수를 확인해 주세요: {labels}"
        )
        st.session_state.acknowledged_selection_warning = mismatch_signature
        update_result_visibility_after_edit()
        st.session_state.scroll_to_next_stage = False
        st.session_state.scroll_to_variable_settings = True
        return

    st.session_state.acknowledged_selection_warning = None
    st.session_state.scroll_to_variable_settings = False
    st.session_state.next_stage_warning = None
    st.session_state.forecast_column = random.choice(FORECAST_COLUMNS)

    # 두 저장 슬롯 중 하나만 비어 있으면 새 예측 결과의 기본 저장 위치로 지정한다.
    saved_slots = st.session_state.get("saved_forecast_datasets", {})
    empty_slots = [
        slot
        for slot in ("데이터셋1", "데이터셋2")
        if saved_slots.get(slot) is None
    ]
    if len(empty_slots) == 1:
        empty_slot = empty_slots[0]
        if st.session_state.get("dataset_save_target") != empty_slot:
            st.session_state.dataset_save_target = empty_slot
            st.session_state.dataset_save_widget_version = (
                st.session_state.get("dataset_save_widget_version", 0) + 1
            )

    st.session_state.show_next_stage = True
    st.session_state.scroll_to_next_stage = True
    st.session_state.next_stage_scroll_nonce += 1


def prepare_one_click_test() -> None:
    prediction_date = random.choice(PREDICTION_DATES)
    st.session_state.prediction_candidate = prediction_date
    st.session_state.applied_prediction_date = prediction_date
    st.session_state.applied_quarter = date_to_quarter(prediction_date)
    st.session_state.prediction_confirmed = True
    forecast_quarters = five_quarters_from(st.session_state.applied_quarter)

    for variable in VARIABLES:
        dataset_number = random.choice(DATASET_NUMBERS)
        st.session_state.variable_dataset_choices[variable] = dataset_number
        context_key = context_key_for(variable)
        st.session_state.final_input_versions[context_key] = (
            st.session_state.final_input_versions.get(context_key, 0) + 1
        )
        scenario_table = build_scenario_table(
            scenario_sets[dataset_number],
            variable,
            st.session_state.applied_quarter,
        )

        selected = {}
        committed = {}
        imported = {}
        for quarter in forecast_quarters:
            available_scenarios = [
                scenario_name
                for scenario_name in scenario_table.index
                if not pd.isna(scenario_table.loc[scenario_name, quarter])
            ]
            scenario_name = random.choice(available_scenarios)
            value = float(scenario_table.loc[scenario_name, quarter])
            selected[quarter] = scenario_name
            committed[quarter] = value
            imported[quarter] = value
            st.session_state[final_input_key(context_key, quarter)] = f"{value:.0f}"

        st.session_state.selected_cells[context_key] = selected
        st.session_state.imported_selections[context_key] = selected.copy()
        st.session_state.imported_values[context_key] = imported.copy()
        st.session_state.final_drafts[context_key] = committed.copy()
        st.session_state.final_values[context_key] = committed.copy()
        st.session_state.chart_focus[context_key] = "final"

    selected_variable = random.choice(VARIABLES)
    st.session_state.selected_variable = selected_variable
    st.session_state.variable_selector = selected_variable
    st.session_state.dataset_selector = (
        f"{selected_variable}{dataset_number_for(selected_variable)}"
    )
    st.session_state.table_warning = None
    st.session_state.next_stage_warning = None
    update_result_visibility_after_edit()
    st.session_state.scroll_to_next_stage = False
    st.session_state.scroll_to_variable_settings = False
    st.session_state.scroll_after_one_click = True
    st.session_state.one_click_scroll_nonce += 1
    st.session_state.forecast_column = None
    st.session_state.acknowledged_selection_warning = None
    st.session_state.final_commit_notice = False
    st.session_state.active_final_input_context = None
    st.session_state.one_click_notice = True


def advance_dataset() -> None:
    variable = st.session_state.selected_variable
    current_number = dataset_number_for(variable)
    current_index = DATASET_NUMBERS.index(current_number)
    next_number = DATASET_NUMBERS[(current_index + 1) % len(DATASET_NUMBERS)]
    st.session_state.variable_dataset_choices[variable] = next_number
    st.session_state.dataset_selector = f"{variable}{next_number}"
    st.session_state.table_warning = None
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None


def enforce_variable_selection() -> None:
    selected_variable = st.session_state.get("variable_selector")
    if selected_variable not in VARIABLES:
        selected_variable = st.session_state.get("selected_variable")
    if selected_variable not in VARIABLES:
        selected_variable = VARIABLES[0]
    st.session_state.variable_selector = selected_variable
    st.session_state.selected_variable = selected_variable
    st.session_state.table_warning = None


def mark_prediction_pending() -> None:
    st.session_state.prediction_confirmed = False
    st.session_state.table_warning = None
    update_result_visibility_after_edit()
    st.session_state.next_stage_warning = None


def render_item_variable_dashboard() -> None:
    try:
        item_code = st.session_state.get("selected_item_code", "gold")
        gold_data, gold_forecast_data, item_variables, item_config = (
            load_configured_item_data(item_code)
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"{SELECTED_ITEM_LABELS.get(item_code, item_code)} 주요 변수 데이터를 불러오지 못했습니다: {error}")
        return

    if gold_data.empty:
        st.warning(f"{item_config['actual_csv']}에 표시할 데이터가 없습니다.")
        return
    if gold_forecast_data.empty:
        st.warning(f"{item_config['forecast_csv']}에 예측 데이터가 없습니다.")
        return

    st.session_state.active_item_variables = item_variables
    variable_labels = item_config.get("labels", {})
    full_labels = item_config.get("full_labels", variable_labels)
    prediction_dates = list(item_config.get("prediction_dates", GOLD_PREDICTION_DATES))

    if st.session_state.gold_variable_selector not in item_variables:
        st.session_state.gold_variable_selector = item_variables[0]
    if st.session_state.gold_prediction_candidate not in prediction_dates:
        st.session_state.gold_prediction_candidate = prediction_dates[-1]
    if st.session_state.get("gold_history_period") not in GOLD_HISTORY_PERIOD_LABELS:
        st.session_state.gold_history_period = "1y"

    with st.container(border=True, key="gold_analysis_workspace"):
        header_layout = st.container(key="gold_workspace_header")
        variable_controls, prediction_controls = header_layout.columns(
            [1.45, 1],
            gap="medium",
        )

        with variable_controls:
            st.markdown(
                '<div class="section-title">주요 변수 설정</div>',
                unsafe_allow_html=True,
            )
            selected_variable = st.segmented_control(
                "주요 변수",
                item_variables,
                key="gold_variable_selector",
                format_func=lambda variable: variable_labels.get(variable, variable),
                label_visibility="collapsed",
                on_change=enforce_gold_variable_selection,
            )
            with st.container(
                key="gold_variable_navigation",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                st.selectbox(
                    "금 데이터 종류",
                    ["1"],
                    key="gold_dataset_selector",
                    label_visibility="collapsed",
                    width=360,
                )
                st.button(
                    "다음",
                    key="gold_dataset_next",
                    width=90,
                    disabled=True,
                    help="현재 주요 변수 데이터는 항목별 1개만 제공됩니다.",
                )

        with prediction_controls:
            st.markdown(
                '<div class="section-title">예측시점</div>',
                unsafe_allow_html=True,
            )
            with st.container(
                key="gold_prediction_navigation",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                st.selectbox(
                    "금 예측시점",
                    prediction_dates,
                    key="gold_prediction_candidate",
                    label_visibility="collapsed",
                    width=170,
                    on_change=mark_gold_prediction_pending,
                )
                st.button(
                    "조회하기",
                    key="gold_prediction_apply",
                    type="primary",
                    width=110,
                    on_click=apply_gold_prediction_date,
                )
            if not st.session_state.gold_prediction_confirmed:
                st.caption("조회하기를 눌러 예측시점을 확정해 주세요.")

        st.markdown(
            '<div class="section-title">그래프 및 주요 변수 데이터</div>',
            unsafe_allow_html=True,
        )
        content_layout = st.container(key="gold_content_layout")
        graph_column, data_column = content_layout.columns(
            [1.35, 1],
            gap="medium",
        )

        active_variable = (
            selected_variable
            if selected_variable in item_variables
            else item_variables[0]
        )
        active_label = full_labels.get(active_variable, active_variable)
        applied_quarter = st.session_state.gold_applied_quarter
        forecast_quarters = five_quarters_from(applied_quarter)
        context_key = gold_context_key(active_variable, applied_quarter)
        chart_focus = st.session_state.gold_chart_focus.setdefault(
            context_key, "final"
        )
        committed_values = st.session_state.gold_final_values.get(context_key, {})

        with graph_column:
            if not st.session_state.gold_prediction_confirmed:
                st.markdown(
                    """
                    <div class="chart-placeholder">
                        <strong>그래프 영역</strong>
                        <span>예측시점을 조회하면 주요 변수 그래프가 표시됩니다.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                active_history_period = (
                    st.session_state.get("gold_history_period") or "1y"
                )
                history_years = int(active_history_period[0])
                historical_quarters = [
                    shift_quarter(applied_quarter, offset)
                    for offset in range(-(history_years * 4), 0)
                ]
                axis_quarters = [*historical_quarters, *forecast_quarters]
                historical_rows = [
                    {
                        "분기": quarter,
                        "값": gold_value_for(gold_data, quarter, active_variable),
                    }
                    for quarter in historical_quarters
                ]
                scenario_rows = {
                    scenario_name: [
                        {
                            "분기": historical_quarters[-1],
                            "값": gold_value_for(
                                gold_data,
                                historical_quarters[-1],
                                active_variable,
                            ),
                        },
                        *[
                            {
                                "분기": quarter,
                                "값": gold_forecast_value_for(
                                    gold_forecast_data,
                                    quarter,
                                    scenario_name,
                                    active_variable,
                                ),
                            }
                            for quarter in forecast_quarters
                        ],
                    ]
                    for scenario_name in ("high", "base", "low")
                }
                historical_frame = pd.DataFrame(historical_rows).dropna(
                    subset=["값"]
                )
                scenario_frames = {
                    scenario_name: pd.DataFrame(rows).dropna(subset=["값"])
                    for scenario_name, rows in scenario_rows.items()
                }
                final_rows = [
                    {
                        "분기": historical_quarters[-1],
                        "값": gold_value_for(
                            gold_data,
                            historical_quarters[-1],
                            active_variable,
                        ),
                    },
                    *[
                        {"분기": quarter, "값": committed_values.get(quarter)}
                        for quarter in forecast_quarters
                    ],
                ]
                final_frame = pd.DataFrame(final_rows).dropna(subset=["값"])
                x_axis = alt.X(
                    "분기:N",
                    title=None,
                    sort=axis_quarters,
                    scale=alt.Scale(domain=axis_quarters),
                    axis=alt.Axis(labelAngle=-30, grid=False),
                )
                y_axis = alt.Y(
                    "값:Q",
                    title=None,
                    scale=alt.Scale(zero=False, padding=25),
                    axis=alt.Axis(grid=True, gridDash=[2, 3]),
                )
                historical_line = (
                    alt.Chart(historical_frame)
                    .mark_line(color="#263238", strokeWidth=2.5, point=True)
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                base_line = (
                    alt.Chart(scenario_frames["base"])
                    .mark_line(
                        color="#2563eb",
                        strokeWidth=2.7,
                        point=alt.OverlayMarkDef(size=45),
                        opacity=1 if chart_focus == "base" else 0.28,
                    )
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                high_line = (
                    alt.Chart(scenario_frames["high"])
                    .mark_line(
                        color="#c026d3",
                        strokeWidth=2.7,
                        point=alt.OverlayMarkDef(size=45),
                        opacity=1 if chart_focus == "high" else 0.28,
                    )
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                low_line = (
                    alt.Chart(scenario_frames["low"])
                    .mark_line(
                        color="#eab308",
                        strokeWidth=2.7,
                        point=alt.OverlayMarkDef(size=45),
                        opacity=1 if chart_focus == "low" else 0.28,
                    )
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                final_line = (
                    alt.Chart(final_frame)
                    .mark_line(
                        color="#ef6461",
                        strokeWidth=3,
                        strokeDash=[6, 4],
                        point=alt.OverlayMarkDef(size=55),
                        opacity=1 if chart_focus == "final" else 0.28,
                    )
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                boundary = pd.DataFrame({"분기": [applied_quarter]})
                boundary_rule = (
                    alt.Chart(boundary)
                    .mark_rule(color="#ff6b6b", strokeDash=[3, 3])
                    .encode(x=alt.X("분기:N", sort=axis_quarters))
                )
                boundary_label = (
                    alt.Chart(boundary)
                    .mark_text(
                        text="예측시점",
                        color="#ff4d4f",
                        align="left",
                        baseline="top",
                        dx=7,
                        dy=6,
                        fontSize=11,
                    )
                    .encode(x=alt.X("분기:N", sort=axis_quarters), y=alt.value(0))
                )
                chart = (
                    alt.layer(
                        historical_line,
                        base_line,
                        high_line,
                        low_line,
                        final_line,
                        boundary_rule,
                        boundary_label,
                    )
                    .properties(
                        title=alt.Title(
                            f"{active_label} 추이",
                            anchor="middle",
                        ),
                        height=350,
                    )
                    .configure_view(stroke=None)
                )
                st.altair_chart(chart, width="stretch")
                st.markdown(
                    """
                    <div class="chart-legend">
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#2563eb"></span>base
                        </span>
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#c026d3"></span>high
                        </span>
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#eab308"></span>low
                        </span>
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#ef6461"></span>final
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.segmented_control(
                "기간",
                list(GOLD_HISTORY_PERIOD_LABELS),
                key="gold_history_period",
                format_func=lambda period: GOLD_HISTORY_PERIOD_LABELS[period],
            )

        with data_column.container(key="gold_data_panel"):
            st.caption(
                f"{active_label} · 예측시점 "
                f"{st.session_state.gold_applied_prediction_date} · "
                f"{applied_quarter}부터 5개 분기"
            )
            scenario_table_values = {
                scenario_name: [
                    gold_forecast_value_for(
                        gold_forecast_data,
                        quarter,
                        scenario_name,
                        active_variable,
                    )
                    if st.session_state.gold_prediction_confirmed
                    else None
                    for quarter in forecast_quarters
                ]
                for scenario_name in ("high", "base", "low")
            }
            selected_for_context = st.session_state.gold_selected_cells.setdefault(
                context_key, {}
            )
            draft_values = st.session_state.gold_final_drafts.setdefault(
                context_key, {}
            )
            with st.container(key="scenario_grid"):
                header_columns = st.columns([.82, 1, 1, 1, 1, 1], gap=None)
                header_columns[0].empty()
                for column, quarter in zip(header_columns[1:], forecast_quarters):
                    column.markdown(
                        f'<div class="grid-header">{quarter}</div>',
                        unsafe_allow_html=True,
                    )

                for scenario_name in ("high", "base", "low"):
                    scenario_values = scenario_table_values[scenario_name]
                    row_columns = st.columns([.82, 1, 1, 1, 1, 1], gap=None)
                    row_columns[0].button(
                        scenario_name,
                        key=f"gold_{scenario_name}_row",
                        type="primary" if chart_focus == scenario_name else "secondary",
                        width="stretch",
                        on_click=select_gold_scenario_row,
                        args=(
                            context_key,
                            scenario_name,
                            tuple(forecast_quarters),
                            tuple(scenario_values),
                        ),
                    )
                    for index, (column, quarter, value) in enumerate(
                        zip(row_columns[1:], forecast_quarters, scenario_values)
                    ):
                        is_selected = (
                            selected_for_context.get(quarter) == scenario_name
                        )
                        column.button(
                            format_gold_value(value),
                            key=(
                                f"gold_{scenario_name}_value_"
                                f"{active_variable}_{index}"
                            ),
                            type="primary" if is_selected else "secondary",
                            width="stretch",
                            on_click=select_gold_scenario_cell,
                            args=(context_key, quarter, scenario_name, value),
                        )

            with st.container(key="commit_row"):
                warning_space, import_button = st.columns([4.4, 1], gap="small")
                with warning_space:
                    if st.session_state.gold_table_warning and not draft_values:
                        st.markdown(
                            f'<div class="commit-warning">{st.session_state.gold_table_warning}</div>',
                            unsafe_allow_html=True,
                        )
                with import_button:
                    st.button(
                        "가져오기",
                        type="primary",
                        width="stretch",
                        on_click=import_gold_selected_values,
                        args=(
                            context_key,
                            tuple(forecast_quarters),
                            {
                                scenario_name: tuple(values)
                                for scenario_name, values in scenario_table_values.items()
                            },
                        ),
                    )

            committed_values = st.session_state.gold_final_values.get(context_key, {})
            if (
                st.session_state.gold_prediction_confirmed
                and st.session_state.gold_active_final_input_context != context_key
            ):
                for quarter in forecast_quarters:
                    restored_value = draft_values.get(
                        quarter, committed_values.get(quarter)
                    )
                    st.session_state[gold_final_input_key(context_key, quarter)] = (
                        "" if restored_value is None else format_gold_value(restored_value)
                    )
                st.session_state.gold_active_final_input_context = context_key

            st.markdown('<div class="final-wrap"></div>', unsafe_allow_html=True)
            final_grid = st.container(key="final_grid")
            final_columns = final_grid.columns([.82, 1, 1, 1, 1, 1], gap=None)
            final_columns[0].button(
                "final",
                key="gold_final_row",
                type="primary" if chart_focus == "final" else "secondary",
                width="stretch",
                disabled=not st.session_state.gold_prediction_confirmed,
                on_click=set_gold_chart_focus,
                args=(context_key, "final"),
            )
            if not st.session_state.gold_prediction_confirmed:
                for index, column in enumerate(final_columns[1:]):
                    column.button(
                        "—",
                        key=f"gold_final_locked_{index}",
                        width="stretch",
                    )
            else:
                for index, (column, quarter) in enumerate(
                    zip(final_columns[1:], forecast_quarters)
                ):
                    input_key = gold_final_input_key(context_key, quarter)
                    if input_key not in st.session_state:
                        initial_value = draft_values.get(
                            quarter, committed_values.get(quarter)
                        )
                        st.session_state[input_key] = (
                            "" if initial_value is None else format_gold_value(initial_value)
                        )
                    current_value = draft_values.get(quarter)
                    committed_value = committed_values.get(quarter)
                    is_dirty = (
                        current_value is not None
                        and (
                            committed_value is None
                            or not values_match(current_value, committed_value)
                        )
                    )
                    cell_state = "dirty" if is_dirty else "clean"
                    with column:
                        with st.container(key=f"final_{cell_state}_gold_{index}"):
                            st.text_input(
                                quarter,
                                key=input_key,
                                placeholder="—",
                                label_visibility="collapsed",
                                on_change=update_gold_final_draft,
                                args=(context_key, quarter),
                            )

            with st.container(key="final_commit_row"):
                warning_space, commit_button = st.columns([4.4, 1], gap="small")
                with warning_space:
                    if st.session_state.gold_table_warning:
                        st.markdown(
                            f'<div class="commit-warning">{st.session_state.gold_table_warning}</div>',
                            unsafe_allow_html=True,
                        )
                with commit_button:
                    st.button(
                        "commit",
                        type="primary",
                        width="stretch",
                        on_click=commit_gold_final_draft,
                        args=(context_key, tuple(forecast_quarters)),
                    )

            if st.session_state.gold_final_commit_notice:
                st.toast("final 값이 반영되었습니다.")
                st.session_state.gold_final_commit_notice = False
    item_label = SELECTED_ITEM_LABELS.get(item_code, item_code)
    with st.container(key="item_price_stage_action"):
        warning_column, quick_column, button_column = st.columns(
            [3.85, 1.0, 1.25], gap="small"
        )
        with warning_column:
            if st.session_state.item_price_stage_warning:
                st.markdown(
                    f'<div class="commit-warning">'
                    f'{escape(st.session_state.item_price_stage_warning)}</div>',
                    unsafe_allow_html=True,
                )
        with quick_column:
            st.button(
                "원클릭 설정",
                key="item_price_stage_quick_button",
                width="stretch",
                on_click=prepare_gold_one_click_test,
                help="테스트용 임의 데이터로 모든 주요 변수의 commit을 완료합니다.",
            )
        with button_column:
            st.button(
                f"{item_label} 가격 예측하기",
                key="item_price_stage_button",
                type="primary",
                width="stretch",
                on_click=proceed_to_item_price_stage,
            )

    if st.session_state.item_price_stage_visible:
        st.html(
            """
            <div id="item-price-stage-anchor" style="scroll-margin-top:110px"></div>
            """
        )
        if st.session_state.item_price_stage_scroll:
            st.html(
                """
                <script>
                    window.setTimeout(() => {
                        const target = document.getElementById(
                            "item-price-stage-anchor"
                        );
                        if (target) {
                            target.scrollIntoView({
                                behavior: "smooth",
                                block: "start"
                            });
                        }
                    }, 120);
                </script>
                """,
                unsafe_allow_javascript=True,
            )
            st.session_state.item_price_stage_scroll = False

        with st.container(border=True, key="item_price_stage_interface"):
            st.markdown(
                '<div class="section-title">가격 예측 결과</div>',
                unsafe_allow_html=True,
            )

            result_variable_key = "item_result_variable_selector"
            if st.session_state.get(result_variable_key) not in item_variables:
                st.session_state[result_variable_key] = item_variables[0]
            if st.session_state.get("item_result_history_period") not in (
                "1년",
                "2년",
                "3년",
            ):
                st.session_state.item_result_history_period = "1년"
            result_period = st.session_state.get(
                "item_result_history_period", "1년"
            )
            history_years = int(result_period[0])
            result_variable = st.session_state[result_variable_key]
            result_quarters = five_quarters_from(applied_quarter)
            result_history_quarters = [
                shift_quarter(applied_quarter, offset)
                for offset in range(-(history_years * 4), 0)
            ]
            result_context_key = gold_context_key(
                result_variable, applied_quarter
            )
            result_committed = st.session_state.gold_final_values.get(
                result_context_key, {}
            )
            result_axis = [*result_history_quarters, *result_quarters]
            result_history_frame = pd.DataFrame(
                {
                    "분기": result_history_quarters,
                    "값": [
                        gold_value_for(gold_data, quarter, result_variable)
                        for quarter in result_history_quarters
                    ],
                }
            ).dropna(subset=["값"])
            result_forecast_frame = pd.DataFrame(
                [
                    {
                        "분기": result_history_quarters[-1],
                        "값": gold_value_for(
                            gold_data,
                            result_history_quarters[-1],
                            result_variable,
                        ),
                    },
                    *[
                        {
                            "분기": quarter,
                            "값": result_committed.get(quarter),
                        }
                        for quarter in result_quarters
                    ],
                ]
            ).dropna(subset=["값"])
            result_x = alt.X(
                "분기:N",
                title=None,
                sort=result_axis,
                scale=alt.Scale(domain=result_axis),
                axis=alt.Axis(labelAngle=-30, grid=False),
            )
            result_y = alt.Y(
                "값:Q",
                title=None,
                scale=alt.Scale(zero=False, padding=45),
                axis=alt.Axis(grid=True, gridDash=[2, 3]),
            )

            saved_forecast_datasets = st.session_state.saved_forecast_datasets
            focused_datasets = set(st.session_state.comparison_dataset_toggle)
            current_is_focused = "현재 그래프" in focused_datasets

            def result_dataset_frame(slot: str) -> pd.DataFrame:
                snapshot = saved_forecast_datasets.get(slot)
                if not snapshot or snapshot.get("item_code") != item_code:
                    return pd.DataFrame(columns=["분기", "값"])
                variable_values = snapshot.get("variable_values", {}).get(
                    result_variable, {}
                )
                return pd.DataFrame(
                    [
                        {
                            "분기": result_history_quarters[-1],
                            "값": gold_value_for(
                                gold_data,
                                result_history_quarters[-1],
                                result_variable,
                            ),
                        },
                        *[
                            {"분기": quarter, "값": variable_values.get(quarter)}
                            for quarter in result_quarters
                        ],
                    ]
                ).dropna(subset=["값"])

            dataset1_forecast_frame = result_dataset_frame("데이터셋1")
            dataset2_forecast_frame = result_dataset_frame("데이터셋2")
            dataset1_is_focused = "데이터셋1" in focused_datasets
            dataset2_is_focused = "데이터셋2" in focused_datasets

            result_layers = [
                alt.Chart(result_history_frame)
                .mark_line(color="#263238", strokeWidth=2.5)
                .encode(
                    x=result_x,
                    y=result_y,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                )
            ]
            if not dataset2_forecast_frame.empty:
                result_layers.append(
                    alt.Chart(dataset2_forecast_frame)
                    .mark_line(
                        color="#CA8A04",
                        strokeWidth=3 if dataset2_is_focused else 1.6,
                        strokeDash=[2, 4],
                        opacity=0.95 if dataset2_is_focused else 0.25,
                        point=alt.OverlayMarkDef(
                            size=70 if dataset2_is_focused else 35,
                            filled=True,
                        ),
                    )
                    .encode(
                        x=result_x,
                        y=result_y,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
            if not dataset1_forecast_frame.empty:
                result_layers.append(
                    alt.Chart(dataset1_forecast_frame)
                    .mark_line(
                        color="#2563EB",
                        strokeWidth=3 if dataset1_is_focused else 1.6,
                        strokeDash=[4, 4],
                        opacity=0.95 if dataset1_is_focused else 0.25,
                        point=alt.OverlayMarkDef(
                            size=70 if dataset1_is_focused else 35,
                            filled=True,
                        ),
                    )
                    .encode(
                        x=result_x,
                        y=result_y,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
            result_layers.append(
                alt.Chart(result_forecast_frame)
                .mark_line(
                    color="#ef6461",
                    strokeWidth=3 if current_is_focused else 1.6,
                    strokeDash=[7, 5],
                    opacity=1 if current_is_focused else 0.3,
                    point=alt.OverlayMarkDef(
                        size=55 if current_is_focused else 35,
                        filled=True,
                    ),
                )
                .encode(
                    x=result_x,
                    y=result_y,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                )
            )
            result_chart = alt.layer(*result_layers).properties(height=285)

            price_actual_chart = None
            price_history_error = None
            price_forecast_available = False
            if item_code == "gold":
                gold_price_path = GOLD_PRICE_PATH
                try:
                    if not gold_price_path.exists():
                        raise FileNotFoundError("gold_price.csv 파일이 없습니다.")
                    gold_price_data = load_gold_price_history(
                        str(gold_price_path),
                        gold_price_path.stat().st_mtime_ns,
                    )
                    price_axis_quarters = [*result_history_quarters, *result_quarters]
                    price_full_frame = (
                        gold_price_data[
                            gold_price_data["분기"].isin(price_axis_quarters)
                        ]
                        .copy()
                    )
                    price_full_frame["분기"] = pd.Categorical(
                        price_full_frame["분기"],
                        categories=price_axis_quarters,
                        ordered=True,
                    )
                    price_full_frame = price_full_frame.sort_values("분기")
                    price_history_frame = price_full_frame.dropna(subset=["실제가"])

                    def price_value_at(quarter: str, column: str) -> float | None:
                        matched = gold_price_data.loc[
                            gold_price_data["분기"] == quarter, column
                        ]
                        if matched.empty or pd.isna(matched.iloc[0]):
                            return None
                        return float(matched.iloc[0])

                    last_actual_quarter = result_history_quarters[-1]
                    price_forecast_frame = pd.DataFrame(
                        [
                            {
                                "분기": last_actual_quarter,
                                "예측가": price_value_at(
                                    last_actual_quarter, "실제가"
                                ),
                            },
                            *[
                                {
                                    "분기": quarter,
                                    "예측가": price_value_at(quarter, "예측가"),
                                }
                                for quarter in result_quarters
                            ],
                        ]
                    ).dropna(subset=["예측가"])
                    price_forecast_available = not price_forecast_frame.empty
                    price_x = alt.X(
                        "분기:N",
                        title=None,
                        sort=price_axis_quarters,
                        scale=alt.Scale(
                            domain=price_axis_quarters
                        ),
                        axis=alt.Axis(
                            labelAngle=-30,
                            grid=False,
                        ),
                    )
                    price_y = alt.Y(
                        "실제가:Q",
                        title=None,
                        scale=alt.Scale(zero=False, padding=45),
                        axis=alt.Axis(
                            grid=True,
                            gridDash=[2, 3],
                            format=",.0f",
                            tickCount=6,
                        ),
                    )
                    price_actual_chart = (
                        alt.Chart(price_history_frame)
                        .mark_line(
                            color="#263238",
                            strokeWidth=2.6,
                            point=alt.OverlayMarkDef(
                                color="#0b6fc2",
                                filled=True,
                                size=42,
                            ),
                        )
                        .encode(
                            x=price_x,
                            y=price_y,
                            tooltip=[
                                "분기:N",
                                alt.Tooltip("실제가:Q", format=",.2f"),
                            ],
                        )
                    )
                    if price_forecast_available:
                        price_actual_chart = price_actual_chart + (
                            alt.Chart(price_forecast_frame)
                            .mark_line(
                                color="#ef6461",
                                strokeWidth=2.6,
                                strokeDash=[7, 5],
                                point=alt.OverlayMarkDef(size=48, filled=True),
                            )
                            .encode(
                                x=price_x,
                                y=alt.Y(
                                    "예측가:Q",
                                    title=None,
                                    scale=alt.Scale(zero=False, padding=45),
                                ),
                                tooltip=[
                                    "분기:N",
                                    alt.Tooltip("예측가:Q", format=",.2f"),
                                ],
                            )
                        )
                    price_actual_chart = price_actual_chart.properties(height=285)
                except (
                    OSError,
                    ValueError,
                    pd.errors.ParserError,
                ) as error:
                    price_history_error = str(error)

            current_result_snapshot = {
                "item": item_label,
                "item_code": item_code,
                "prediction_date": st.session_state.gold_applied_prediction_date,
                "quarter": applied_quarter,
                "variable_values": {
                    variable: deepcopy(
                        st.session_state.gold_final_values.get(
                            gold_context_key(variable, applied_quarter),
                            {},
                        )
                    )
                    for variable in item_variables
                },
                "price_values": None,
            }

            result_selector_labels = [
                variable_labels.get(variable, variable)
                for variable in item_variables
            ]
            result_selector_rows = estimate_pill_row_count(
                result_selector_labels
            )
            result_header_height = 38 + (result_selector_rows - 1) * 40
            st.html(
                f"""
                <style>
                    .st-key-variable_forecast_header,
                    .st-key-price_forecast_header {{
                        min-height:{result_header_height}px !important;
                    }}
                </style>
                """
            )

            comparison_layout = st.container(key="forecast_comparison_layout")
            graph_area, dataset_panel = comparison_layout.columns(
                [4.1, 1], gap="small"
            )
            graph_grid = graph_area.container(key="forecast_graph_grid")
            variable_graph, price_graph = graph_grid.columns(2, gap="small")

            with variable_graph:
                with st.container(
                    border=True,
                    key="variable_forecast_card",
                ):
                    with st.container(key="variable_forecast_header"):
                        title_column, selector_column = st.columns(
                            [1.05, 2.35],
                            gap="small",
                            vertical_alignment="top",
                        )
                    with title_column:
                        st.markdown(
                            '<div class="forecast-card-title">'
                            '• 주요 변수 전망</div>',
                            unsafe_allow_html=True,
                        )
                    with selector_column:
                        with st.container(key="item_result_variable_scroll"):
                            st.pills(
                                "주요 변수",
                                options=item_variables,
                                key=result_variable_key,
                                format_func=lambda variable: variable_labels.get(
                                    variable, variable
                                ),
                                selection_mode="single",
                                width="stretch",
                                label_visibility="collapsed",
                            )
                    st.altair_chart(
                        result_chart,
                        key="item_result_variable_chart",
                        width="stretch",
                    )
                    dataset1_legend_item = (
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" '
                        'style="background:#2563EB"></span>'
                        f'{escape(dataset_display_name("데이터셋1"))}</span>'
                        if not dataset1_forecast_frame.empty
                        else ""
                    )
                    dataset2_legend_item = (
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" '
                        'style="background:#CA8A04"></span>'
                        f'{escape(dataset_display_name("데이터셋2"))}</span>'
                        if not dataset2_forecast_frame.empty
                        else ""
                    )
                    st.markdown(
                        '<div class="chart-legend">'
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" '
                        'style="background:#263238"></span>과거 데이터</span>'
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" '
                        'style="background:#ef6461"></span>commit된 final</span>'
                        f'{dataset2_legend_item}'
                        f'{dataset1_legend_item}'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            with price_graph:
                with st.container(
                    border=True,
                    key="price_forecast_card",
                ):
                    with st.container(
                        key="price_forecast_header",
                    ):
                        st.markdown(
                            f'<div class="forecast-card-title">'
                            f'• {escape(item_label)} 가격 전망</div>',
                            unsafe_allow_html=True,
                        )
                    if price_actual_chart is not None:
                        st.altair_chart(
                            price_actual_chart,
                            key="gold_price_actual_chart",
                            width="stretch",
                        )
                        forecast_legend_item = (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" '
                            'style="background:#ef6461"></span>예측가</span>'
                            if price_forecast_available
                            else ""
                        )
                        st.markdown(
                            '<div class="chart-legend">'
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" '
                            'style="background:#263238"></span>실제가</span>'
                            f'{forecast_legend_item}'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        error_detail = (
                            escape(price_history_error)
                            if price_history_error
                            else "가격 데이터 연결 후 표시됩니다."
                        )
                        st.markdown(
                            f"""
                            <div class="chart-placeholder" style="height:285px">
                                <strong>가격 예측 그래프 영역</strong>
                                <span>{error_detail}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="chart-legend">'
                            '<span class="chart-legend-placeholder">실제가</span>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

            with dataset_panel:
                with st.container(
                    border=True,
                    key="dataset_comparison_panel",
                    height="stretch",
                ):
                    st.markdown(
                        '<div class="dataset-panel-title">• 데이터셋 비교</div>'
                        '<div class="dataset-panel-caption">'
                        '현재 결과를 저장할 슬롯과 강조할 데이터셋을 '
                        '선택합니다.</div>',
                        unsafe_allow_html=True,
                    )
                    save_target_widget_key = (
                        "dataset_save_target_widget_"
                        f"{st.session_state.dataset_save_widget_version}"
                    )
                    if save_target_widget_key not in st.session_state:
                        st.session_state[save_target_widget_key] = (
                            st.session_state.dataset_save_target
                        )
                    save_target = st.selectbox(
                        "저장 위치",
                        options=["데이터셋1", "데이터셋2"],
                        format_func=dataset_display_name,
                        key=save_target_widget_key,
                        on_change=sync_dataset_save_target,
                        args=(save_target_widget_key,),
                        width="stretch",
                    )
                    st.session_state.dataset_save_target = save_target
                    st.button(
                        "데이터셋 저장",
                        key="item_result_save_button",
                        icon=":material/save:",
                        width="stretch",
                        on_click=save_forecast_dataset,
                        args=(current_result_snapshot,),
                    )

                    with st.popover(
                        "데이터셋 관리",
                        icon=":material/edit:",
                        width="stretch",
                    ):
                        st.text_input(
                            "데이터셋1 이름",
                            key="dataset_name_input_1",
                            max_chars=16,
                        )
                        st.text_input(
                            "데이터셋2 이름",
                            key="dataset_name_input_2",
                            max_chars=16,
                        )
                        st.button(
                            "이름 적용",
                            key="item_result_name_apply",
                            width="stretch",
                            on_click=apply_item_dataset_names,
                        )
                        st.caption("삭제한 저장 내용은 복구할 수 없습니다.")
                        st.button(
                            f"{dataset_display_name('데이터셋1')} 내용 삭제",
                            key="item_result_delete_1",
                            width="stretch",
                            on_click=delete_item_result_dataset,
                            args=("데이터셋1",),
                        )
                        st.button(
                            f"{dataset_display_name('데이터셋2')} 내용 삭제",
                            key="item_result_delete_2",
                            width="stretch",
                            on_click=delete_item_result_dataset,
                            args=("데이터셋2",),
                        )

                    saved_slots = st.session_state.saved_forecast_datasets
                    slot_markup = ""
                    for slot in ("데이터셋1", "데이터셋2"):
                        state = "저장됨" if saved_slots.get(slot) else "비어 있음"
                        current_class = (
                            "dataset-slot current"
                            if save_target == slot
                            else "dataset-slot"
                        )
                        slot_markup += (
                            f'<div class="{current_class}">'
                            f'<span class="dataset-slot-name">'
                            f'{escape(dataset_display_name(slot))}</span>'
                            f'<span class="dataset-slot-state">{state}</span>'
                            '</div>'
                        )
                    st.markdown(slot_markup, unsafe_allow_html=True)
                    with st.container(
                        key="item_dataset_focus_section",
                        gap="small",
                    ):
                        st.pills(
                            "그래프 강조",
                            options=["현재 그래프"],
                            key="comparison_current_focus",
                            selection_mode="multi",
                            label_visibility="visible",
                            on_change=sync_comparison_focus,
                            width="stretch",
                        )
                        st.pills(
                            "저장 데이터셋 강조",
                            options=["데이터셋1", "데이터셋2"],
                            format_func=dataset_display_name,
                            key="comparison_saved_focus",
                            selection_mode="multi",
                            label_visibility="collapsed",
                            on_change=sync_comparison_focus,
                            width="stretch",
                        )
                    notice_slot = st.session_state.dataset_save_notice
                    notice = (
                        f"현재 결과를 {dataset_display_name(notice_slot)}에 "
                        "저장했습니다."
                        if notice_slot
                        else ""
                    )
                    st.markdown(
                        '<div class="dataset-save-notice">'
                        f'{escape(notice) or "&nbsp;"}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.session_state.dataset_manage_notice:
                        st.caption(st.session_state.dataset_manage_notice)

            with graph_area:
                with st.container(
                    key="forecast_period_control",
                    horizontal=True,
                    vertical_alignment="center",
                    gap="small",
                ):
                    st.markdown(
                        '<span class="period-inline-label">기간</span>',
                        unsafe_allow_html=True,
                    )
                    st.segmented_control(
                        "기간",
                        options=["1년", "2년", "3년"],
                        key="item_result_history_period",
                        selection_mode="single",
                        width="content",
                        label_visibility="collapsed",
                    )


def apply_item_selection() -> None:
    candidate = st.session_state.item_candidate
    st.session_state.selected_item = candidate
    st.session_state.selected_item_code = ITEM_CODES[candidate]
    st.session_state.item_confirmed = True
    st.session_state.selected_variable = VARIABLES[0]
    st.session_state.variable_selector = VARIABLES[0]
    st.session_state.dataset_selector = (
        f"{VARIABLES[0]}{dataset_number_for(VARIABLES[0])}"
    )
    st.session_state.prediction_confirmed = False
    st.session_state.table_warning = None
    st.session_state.show_next_stage = False
    st.session_state.next_stage_warning = None
    st.session_state.item_warning = None
    st.session_state.selector_open = False
    item_code = st.session_state.selected_item_code
    try:
        _, _, item_variables, _ = load_configured_item_data(item_code)
    except (OSError, ValueError, pd.errors.ParserError):
        item_variables = ()
    st.session_state.active_item_variables = item_variables
    if item_variables:
        st.session_state.gold_variable_selector = item_variables[0]
    st.session_state.gold_dataset_selector = "1"
    st.session_state.gold_prediction_confirmed = False
    st.session_state.gold_selected_cells = {}
    st.session_state.gold_final_values = {}
    st.session_state.gold_final_drafts = {}
    st.session_state.gold_imported_selections = {}
    st.session_state.gold_chart_focus = {}
    st.session_state.gold_table_warning = None
    st.session_state.gold_active_final_input_context = None
    st.session_state.item_price_stage_visible = False
    st.session_state.item_price_stage_warning = None
    st.session_state.item_price_stage_scroll = False


for key, default in {
    "selector_open": False,
    "selected_item": "전기동",
    "selected_item_code": "copper",
    "item_candidate": "전기동",
    "item_warning": None,
    "item_confirmed": False,
    "selected_variable": VARIABLES[0],
    "variable_selector": VARIABLES[0],
    "prediction_candidate": PREDICTION_DATES[0],
    "prediction_confirmed": False,
    "table_warning": None,
    "next_stage_warning": None,
    "show_next_stage": False,
    "scroll_to_next_stage": False,
    "scroll_to_variable_settings": False,
    "scroll_after_one_click": False,
    "one_click_scroll_nonce": 0,
    "next_stage_scroll_nonce": 0,
    "one_click_notice": False,
    "forecast_column": None,
    "acknowledged_selection_warning": None,
    "active_final_input_context": None,
    "gold_variable_selector": GOLD_VARIABLES[0],
    "gold_dataset_selector": "1",
    "gold_prediction_candidate": "2026-01-21",
    "gold_applied_prediction_date": "2026-01-21",
    "gold_applied_quarter": "2026-Q1",
    "gold_prediction_confirmed": False,
    "gold_history_period": "1y",
    "gold_selected_cells": {},
    "gold_final_values": {},
    "gold_final_drafts": {},
    "gold_final_input_versions": {},
    "gold_imported_selections": {},
    "gold_chart_focus": {},
    "gold_table_warning": None,
    "gold_active_final_input_context": None,
    "gold_final_commit_notice": False,
    "active_item_variables": GOLD_VARIABLES,
    "item_price_stage_visible": False,
    "item_price_stage_warning": None,
    "item_price_stage_scroll": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.selected_variable not in VARIABLES:
    st.session_state.selected_variable = VARIABLES[0]
if st.session_state.variable_selector not in VARIABLES:
    st.session_state.variable_selector = st.session_state.selected_variable

scenario_sets = {
    dataset_number: load_scenarios(dataset_number)
    for dataset_number in DATASET_NUMBERS
}
price_forecast_data = load_price_forecast_data()
scenarios = scenario_sets[1]
available_dates = scenarios["base"].get("Date", pd.Series(dtype=str)).astype(str).tolist()
quarter_options = [quarter_label(value) for value in sorted(available_dates, key=quarter_key)]
default_prediction_date = PREDICTION_DATES[0]
default_quarter = date_to_quarter(default_prediction_date)
if "applied_prediction_date" not in st.session_state:
    st.session_state.applied_prediction_date = default_prediction_date
if "applied_quarter" not in st.session_state:
    st.session_state.applied_quarter = default_quarter
if "selected_cells" not in st.session_state:
    st.session_state.selected_cells = {}
if "final_values" not in st.session_state:
    st.session_state.final_values = {}
if "final_drafts" not in st.session_state:
    st.session_state.final_drafts = {}
if "final_input_versions" not in st.session_state:
    st.session_state.final_input_versions = {}
if "imported_selections" not in st.session_state:
    st.session_state.imported_selections = {}
if "imported_values" not in st.session_state:
    st.session_state.imported_values = {}
if "final_commit_notice" not in st.session_state:
    st.session_state.final_commit_notice = False
if "chart_focus" not in st.session_state:
    st.session_state.chart_focus = {}
if "variable_dataset_choices" not in st.session_state:
    st.session_state.variable_dataset_choices = {
        variable: 1 for variable in VARIABLES
    }
if "saved_forecast_dataset" not in st.session_state:
    st.session_state.saved_forecast_dataset = None
if "saved_forecast_datasets" not in st.session_state:
    st.session_state.saved_forecast_datasets = {
        "데이터셋1": st.session_state.saved_forecast_dataset,
        "데이터셋2": None,
    }
elif (
    st.session_state.saved_forecast_datasets.get("데이터셋1") is None
    and st.session_state.saved_forecast_dataset is not None
):
    st.session_state.saved_forecast_datasets["데이터셋1"] = (
        st.session_state.saved_forecast_dataset
    )
if "dataset_save_target" not in st.session_state:
    st.session_state.dataset_save_target = "데이터셋1"
if "dataset_save_widget_version" not in st.session_state:
    st.session_state.dataset_save_widget_version = 0
if "dataset_display_names" not in st.session_state:
    st.session_state.dataset_display_names = {
        "데이터셋1": "데이터셋1",
        "데이터셋2": "데이터셋2",
    }
for slot in ("데이터셋1", "데이터셋2"):
    input_key = f"dataset_name_input_{slot[-1]}"
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state.dataset_display_names[slot]
if "dataset_manage_notice" not in st.session_state:
    st.session_state.dataset_manage_notice = None
if not st.session_state.get("comparison_focus_v2_initialized", False):
    st.session_state.comparison_dataset_toggle = ["현재 그래프"]
    st.session_state.comparison_focus_v2_initialized = True
elif not isinstance(
    st.session_state.get("comparison_dataset_toggle"),
    list,
):
    previous_focus = st.session_state.get("comparison_dataset_toggle")
    st.session_state.comparison_dataset_toggle = (
        [previous_focus]
        if previous_focus in ("데이터셋1", "데이터셋2")
        else ["현재 그래프"]
    )
if "comparison_current_focus" not in st.session_state:
    st.session_state.comparison_current_focus = [
        focus
        for focus in st.session_state.comparison_dataset_toggle
        if focus == "현재 그래프"
    ]
if "comparison_saved_focus" not in st.session_state:
    st.session_state.comparison_saved_focus = [
        focus
        for focus in st.session_state.comparison_dataset_toggle
        if focus in ("데이터셋1", "데이터셋2")
    ]
if "dataset_save_notice" not in st.session_state:
    st.session_state.dataset_save_notice = False

st.markdown(
    """
    <style>
        .stApp { background: #f7f8fa; }
        .block-container { max-width: 1320px; padding-top: 2.4rem; }
        .eyebrow { color:#697386; font-size:.76rem; font-weight:700; letter-spacing:.08em; }
        .main-title { margin:.15rem 0 .2rem; font-size:1.7rem; color:#172033; }
        .description { color:#697386; margin-bottom:1.2rem; }
        .section-title {
            display:flex; align-items:center; gap:.48rem;
            margin:.15rem 0 .7rem; color:#172033;
            font-size:1.05rem; font-weight:800; line-height:1.25;
        }
        .section-title::before {
            content:""; width:4px; height:18px; flex:0 0 4px;
            border-radius:3px; background:#172033;
        }
        .selected-card {
            background:white; border:1px solid #e3e7ed; border-radius:9px;
            padding:.6rem .8rem; box-shadow:0 3px 12px rgba(23,32,51,.05);
        }
        .selected-label { color:#7a8495; font-size:.68rem; margin-bottom:.12rem; }
        .selected-value { color:#172033; font-size:.92rem; font-weight:700; }
        .st-key-item_select_button { padding-top:8px; }
        .item-warning {
            min-height:46px; box-sizing:border-box; display:flex; align-items:center;
            justify-content:center; text-align:center; margin-top:1.35rem;
            padding:.65rem .9rem; border:1px solid #eee8a9; border-radius:8px;
            background:#fffde7; color:#8a6900; font-size:.9rem;
            animation:warning-attention .62s ease-in-out 1;
        }
        .workspace-title { color:#172033; font-size:1.22rem; font-weight:750; margin-bottom:.7rem; }
        .chart-placeholder {
            height:330px; border:1px dashed #bdc6d4; border-radius:10px; background:white;
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            color:#8a94a6; text-align:center;
        }
        .chart-placeholder strong { color:#536072; margin-bottom:.35rem; }
        .chart-legend {
            display:flex; align-items:center; justify-content:center; flex-wrap:wrap;
            gap:.35rem 1rem; margin:-.45rem 0 .1rem;
            color:#697386; font-size:.78rem;
        }
        .chart-legend-item { display:inline-flex; align-items:center; gap:.32rem; }
        .chart-legend-dot {
            width:10px; height:10px; border-radius:50%; display:inline-block;
        }
        .chart-legend-placeholder { visibility:hidden; }
        .forecast-card-title {
            min-height:40px; display:flex; align-items:center;
            color:#172033; font-size:1.05rem; font-weight:400;
            white-space:nowrap; margin:0;
        }
        .st-key-variable_forecast_header,
        .st-key-price_forecast_header {
            min-height:78px;
            display:flex;
            align-items:flex-start;
            justify-content:flex-start;
            margin-bottom:-.55rem !important;
        }
        .st-key-variable_forecast_header {
            margin-bottom:-.55rem !important;
        }
        .st-key-variable_forecast_header .forecast-card-title {
            transform:none;
            white-space:nowrap;
            min-height:36px;
        }
        .st-key-price_forecast_header .forecast-card-title {
            transform:none;
            min-height:36px;
        }
        .st-key-forecast_variable_selector [data-testid="stButtonGroup"] {
            flex-wrap:nowrap !important;
            width:auto !important;
        }
        .st-key-forecast_variable_selector button[data-variant="segmented_control"] {
            min-width:0 !important;
            min-height:32px !important;
            height:32px !important;
            padding:.12rem .62rem !important;
            font-size:.72rem !important;
            white-space:nowrap !important;
        }
        .st-key-forecast_variable_selector button[data-variant="segmented_control"] p {
            font-size:.72rem !important;
            line-height:1 !important;
        }
        .st-key-item_result_variable_scroll {
            box-sizing:border-box !important;
            width:100% !important;
            min-width:0 !important;
            max-width:100% !important;
            overflow:visible !important;
            min-height:36px;
            margin:0 !important;
            padding:1px !important;
        }
        .st-key-item_result_variable_scroll
        [data-testid="stButtonGroup"] {
            display:flex !important;
            flex-wrap:wrap !important;
            width:100% !important;
            min-width:0 !important;
            gap:.38rem !important;
            margin:0 !important;
        }
        .st-key-item_result_variable_scroll
        button {
            flex:0 0 auto !important;
            white-space:nowrap !important;
            min-height:34px !important;
            height:auto !important;
            padding:.28rem .72rem !important;
            font-size:.72rem !important;
            border-radius:999px !important;
        }
        .st-key-item_result_variable_scroll
        button p {
            white-space:nowrap !important;
            font-size:.72rem !important;
            line-height:1.35 !important;
        }
        .st-key-comparison_current_focus button {
            width:100% !important;
            justify-content:center !important;
        }
        .st-key-comparison_current_focus {
            margin-bottom:-.35rem !important;
        }
        .dataset-panel-title {
            color:#172033; font-size:.96rem; font-weight:750;
            margin:.1rem 0 .25rem;
        }
        .dataset-panel-caption {
            color:#7a8495; font-size:.72rem; line-height:1.45;
            margin-bottom:.45rem;
        }
        .st-key-item_dataset_focus_section {
            margin-top:.65rem !important;
        }
        .dataset-slot {
            display:flex; align-items:center; justify-content:space-between;
            gap:.45rem; min-height:38px; margin:.38rem 0;
            padding:.48rem .58rem; border:1px solid #e1e6ed;
            border-radius:8px; background:#f8fafc;
        }
        .dataset-slot-name {
            color:#3e4a5d; font-size:.78rem; font-weight:700;
        }
        .dataset-slot-state {
            color:#7a8495; font-size:.66rem; white-space:nowrap;
        }
        .dataset-slot.current {
            border-color:#f1c1bf; background:#fff7f6;
        }
        .dataset-slot.current .dataset-slot-state { color:#d95450; }
        .st-key-dataset_save_button button {
            border-color:#506780 !important; color:#40566f !important;
            background:#f5f8fb !important;
        }
        .dataset-save-notice {
            min-height:2.1rem; box-sizing:border-box; padding-top:.38rem;
            color:#7a8495; font-size:.72rem; line-height:1.35;
        }
        .period-inline-label {
            color:#7a8495; font-size:.72rem; line-height:30px;
            white-space:nowrap;
        }
        .st-key-history_period_control,
        .st-key-forecast_period_control {
            width:245px; margin-top:-.15rem;
        }
        .st-key-history_period_control [data-testid="stWidgetLabel"],
        .st-key-forecast_period_control [data-testid="stWidgetLabel"] {
            margin-bottom:.1rem;
        }
        .st-key-history_period_control [data-testid="stWidgetLabel"] p,
        .st-key-forecast_period_control [data-testid="stWidgetLabel"] p {
            font-size:.72rem; color:#7a8495;
        }
        .st-key-history_period_control [data-testid="stSegmentedControl"] button,
        .st-key-forecast_period_control [data-testid="stSegmentedControl"] button {
            min-height:30px !important; padding:.15rem .55rem !important;
            font-size:.72rem !important;
        }
        div[data-testid="stButton"] > button { min-height:39px; border-radius:8px; font-weight:700; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:white; border-color:#dfe4eb; border-radius:11px;
            box-shadow:0 5px 18px rgba(23,32,51,.06);
        }
        .grid-header, .grid-label, .final-cell {
            min-height:37px; border-radius:7px; display:flex; align-items:center;
            justify-content:center; padding:.25rem .2rem; margin-inline:2px;
            font-size:.76rem;
        }
        .st-key-scenario_grid div[data-testid="stButton"] > button {
            width:calc(100% - 4px) !important; margin-inline:2px;
        }
        .grid-header {
            color:#6f7a8c; background:#eef1f5; font-weight:700;
            margin-bottom:4px;
        }
        .grid-label { color:#536072; background:#f3f5f8; font-weight:700; }
        .st-key-scenario_grid button[data-testid="stBaseButton-primary"] {
            background:#6f9692 !important; border-color:#6f9692 !important; color:white !important;
        }
        .st-key-scenario_grid button[data-testid="stBaseButton-primary"]:hover {
            background:#638985 !important; border-color:#638985 !important;
        }
        .st-key-scenario_grid button[data-testid="stBaseButton-tertiary"] {
            background:#b7793e !important; border:1px solid #b7793e !important;
            color:white !important;
        }
        .st-key-scenario_grid button[data-testid="stBaseButton-tertiary"]:hover {
            background:#9f6634 !important; border-color:#9f6634 !important;
        }
        [class*="st-key-chart_focus_"] button {
            min-height:39px !important; padding:.25rem .1rem !important;
            border-radius:7px !important; font-size:.76rem !important;
            white-space:nowrap !important;
        }
        [class*="st-key-chart_focus_"] button[data-testid="stBaseButton-secondary"] {
            background:#f3f5f8 !important; border-color:#f3f5f8 !important;
            color:#536072 !important;
        }
        .st-key-chart_focus_base button[data-testid="stBaseButton-primary"] {
            background:#2563eb !important; border-color:#2563eb !important;
            color:white !important;
        }
        .st-key-chart_focus_base button[data-testid="stBaseButton-primary"]:hover {
            background:#1d4ed8 !important; border-color:#1d4ed8 !important;
            color:white !important;
        }
        .st-key-chart_focus_high button[data-testid="stBaseButton-primary"] {
            background:#c026d3 !important; border-color:#c026d3 !important;
            color:white !important;
        }
        .st-key-chart_focus_high button[data-testid="stBaseButton-primary"]:hover {
            background:#a21caf !important; border-color:#a21caf !important;
            color:white !important;
        }
        .st-key-chart_focus_low button[data-testid="stBaseButton-primary"] {
            background:#eab308 !important; border-color:#eab308 !important;
            color:#172033 !important;
        }
        .st-key-chart_focus_low button[data-testid="stBaseButton-primary"]:hover {
            background:#ca8a04 !important; border-color:#ca8a04 !important;
            color:#172033 !important;
        }
        .st-key-chart_focus_final button[data-testid="stBaseButton-primary"] {
            background:#ef6461 !important; border-color:#ef6461 !important;
            color:white !important;
        }
        .st-key-chart_focus_final button[data-testid="stBaseButton-primary"]:hover {
            background:#dc5653 !important; border-color:#dc5653 !important;
            color:white !important;
        }
        .st-key-chart_focus_final div[data-testid="stButton"] > button {
            height:39px !important; min-height:39px !important;
        }
        .final-wrap {
            margin-top:1.1rem; padding-top:.8rem; border-top:2px solid #d9e1ec;
        }
        .final-label { background:#6f9692; color:white; font-weight:800; }
        .final-value { background:#6f9692; color:white; font-weight:750; }
        .st-key-final_grid div[data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:.82fr repeat(5, minmax(0, 1fr));
            gap:4px !important;
        }
        .st-key-final_grid div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {
            width:auto !important; min-width:0 !important;
            flex:none !important;
        }
        [class*="st-key-final_clean_"] input {
            width:100%; margin:0;
            height:46px; background:#6f9692 !important; color:white !important;
            border-color:#6f9692 !important; text-align:center; font-weight:750;
        }
        [class*="st-key-final_clean_"] input:focus {
            background:#b7793e !important; border-color:#b7793e !important;
            box-shadow:inset 0 0 0 1px #b7793e !important;
            outline:none !important;
        }
        [class*="st-key-final_dirty_"] input {
            width:100%; margin:0;
            height:46px; background:#b7793e !important; color:white !important;
            border-color:#b7793e !important; text-align:center; font-weight:750;
        }
        [class*="st-key-final_empty_"] input {
            width:100%; margin:0;
            height:46px; background:white !important; text-align:center; font-weight:700;
        }
        [class*="st-key-final_locked_"] div[data-testid="stButton"] > button {
            width:100% !important; margin:0;
            height:39px; min-height:39px;
        }
        .commit-warning {
            height:46px; box-sizing:border-box; display:flex; align-items:center;
            justify-content:center; text-align:center; line-height:1.25;
            padding:0 .95rem; border:1px solid #eee8a9; border-radius:8px;
            background:#fffde7; color:#8a6900; font-size:.9rem;
            animation:warning-attention .62s ease-in-out 1;
        }
        @keyframes warning-attention {
            0%, 100% {
                background:#fffde7;
                border-color:#eee8a9;
                box-shadow:0 0 0 rgba(234, 179, 8, 0);
                transform:scale(1);
            }
            50% {
                background:#fff4bf;
                border-color:#e2bd3b;
                box-shadow:0 0 0 4px rgba(234, 179, 8, .16);
                transform:scale(1.008);
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .item-warning,
            .commit-warning {
                animation:none;
            }
        }
        .st-key-commit_row div[data-testid="stButton"] > button {
            height:46px; min-height:46px; padding-inline:.35rem;
            white-space:nowrap;
        }
        .st-key-final_commit_row div[data-testid="stButton"] > button {
            height:46px; min-height:46px; padding-inline:.35rem;
            white-space:nowrap;
        }
        .st-key-next_stage_action div[data-testid="stButton"] > button {
            height:46px; min-height:46px;
        }
        .st-key-next_stage_action button[data-testid="stBaseButton-primary"]:not(:disabled) {
            background:#2f7d5c !important; border-color:#2f7d5c !important;
            color:white !important;
        }
        .st-key-next_stage_action button[data-testid="stBaseButton-primary"]:not(:disabled):hover {
            background:#25664b !important; border-color:#25664b !important;
            color:white !important;
        }
        .st-key-variable_navigation[data-testid="stHorizontalBlock"],
        .st-key-variable_navigation > div[data-testid="stHorizontalBlock"],
        .st-key-gold_variable_navigation[data-testid="stHorizontalBlock"],
        .st-key-gold_variable_navigation > div[data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:360px 90px !important;
            justify-content:start !important;
            gap:16px !important;
        }
        .st-key-variable_navigation[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-variable_navigation > div[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-gold_variable_navigation[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-gold_variable_navigation > div[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"] {
            width:100% !important; min-width:0 !important;
        }
        .st-key-prediction_navigation[data-testid="stHorizontalBlock"],
        .st-key-prediction_navigation > div[data-testid="stHorizontalBlock"],
        .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"],
        .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:170px 110px !important;
            justify-content:start !important;
            align-items:center !important;
            gap:12px !important;
            flex-wrap:nowrap !important;
        }
        .st-key-prediction_navigation[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-prediction_navigation > div[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"] {
            width:100% !important;
            min-width:0 !important;
        }
        @media (max-width: 1100px) {
            .st-key-variable_navigation[data-testid="stHorizontalBlock"],
            .st-key-variable_navigation > div[data-testid="stHorizontalBlock"],
            .st-key-gold_variable_navigation[data-testid="stHorizontalBlock"],
            .st-key-gold_variable_navigation > div[data-testid="stHorizontalBlock"] {
                grid-template-columns:330px 88px !important;
                gap:12px !important;
            }
            .st-key-prediction_navigation[data-testid="stHorizontalBlock"],
            .st-key-prediction_navigation > div[data-testid="stHorizontalBlock"],
            .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"],
            .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"] {
                grid-template-columns:160px 100px !important;
                gap:10px !important;
            }
        }
        @media (max-width: 620px) {
            .st-key-variable_navigation[data-testid="stHorizontalBlock"],
            .st-key-variable_navigation > div[data-testid="stHorizontalBlock"],
            .st-key-gold_variable_navigation[data-testid="stHorizontalBlock"],
            .st-key-gold_variable_navigation > div[data-testid="stHorizontalBlock"] {
                grid-template-columns:minmax(210px, 1fr) 82px !important;
            }
            .st-key-prediction_navigation[data-testid="stHorizontalBlock"],
            .st-key-prediction_navigation > div[data-testid="stHorizontalBlock"],
            .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"],
            .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"] {
                grid-template-columns:minmax(150px, 1fr) 96px !important;
            }
        }
        @media (max-width: 1180px) {
            .st-key-item_selection_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-item_summary_layout),
            .st-key-analysis_content_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-scenario_data_panel),
            .st-key-gold_content_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_data_panel) {
                flex-wrap:wrap !important;
                align-items:flex-start !important;
            }
            .st-key-item_selection_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-item_summary_layout)
            > div[data-testid="stColumn"],
            .st-key-analysis_content_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-scenario_data_panel)
            > div[data-testid="stColumn"],
            .st-key-gold_content_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_data_panel)
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
            .st-key-forecast_comparison_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-forecast_graph_grid) {
                flex-wrap:wrap !important;
                align-items:flex-start !important;
            }
            .st-key-forecast_comparison_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-forecast_graph_grid)
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
            .st-key-dataset_comparison_panel {
                height:auto !important;
                min-height:0 !important;
            }
        }
        @media (max-width: 960px) {
            .st-key-workspace_header_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-variable_navigation),
            .st-key-gold_workspace_header
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_variable_navigation) {
                flex-wrap:wrap !important;
                align-items:flex-start !important;
            }
            .st-key-workspace_header_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-variable_navigation)
            > div[data-testid="stColumn"],
            .st-key-gold_workspace_header
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_variable_navigation)
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
        }
        @media (max-width: 820px) {
            .st-key-forecast_graph_grid
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]) {
                flex-wrap:wrap !important;
            }
            .st-key-forecast_graph_grid
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"])
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
            .st-key-variable_forecast_header {
                flex-wrap:wrap !important;
                align-items:flex-start !important;
            }
            .st-key-variable_forecast_header .forecast-card-title,
            .st-key-price_forecast_header .forecast-card-title {
                transform:none;
            }
            .st-key-variable_forecast_header .forecast-card-title {
                width:100%;
                min-height:30px;
            }
            .st-key-forecast_variable_selector {
                max-width:100% !important;
                overflow-x:auto;
            }
            .st-key-variable_forecast_card,
            .st-key-price_forecast_card {
                width:100% !important;
                min-width:0 !important;
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding-left:.75rem !important;
                padding-right:.75rem !important;
            }
            .st-key-item_summary_layout
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]),
            .st-key-next_stage_controls
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]) {
                flex-wrap:wrap !important;
            }
            .st-key-item_summary_layout
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"])
            > div[data-testid="stColumn"],
            .st-key-next_stage_controls
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"])
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
            .st-key-variable_selector [data-testid="stButtonGroup"],
            .st-key-gold_variable_selector [data-testid="stButtonGroup"] {
                flex-wrap:nowrap !important;
                overflow-x:auto !important;
                scrollbar-width:thin;
            }
            .st-key-scenario_data_panel,
            .st-key-gold_data_panel {
                overflow-x:auto !important;
                padding-bottom:.25rem;
            }
            .st-key-scenario_grid,
            .st-key-commit_row,
            .st-key-final_grid,
            .st-key-final_commit_row {
                min-width:620px !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">COMMODITY DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">원자재 가격 대시보드</h1>', unsafe_allow_html=True)
st.markdown('<div class="description">분석할 품목을 선택해 주세요.</div>', unsafe_allow_html=True)

item_selection_layout = st.container(key="item_selection_layout")
summary_column, panel_column = item_selection_layout.columns([1.85, 2.8], gap="medium")
with summary_column:
    item_summary_layout = st.container(key="item_summary_layout")
    button_column, selected_column = item_summary_layout.columns([0.8, 1.05], gap="medium")
    with button_column:
        with st.container(key="item_select_button"):
            button_text = "선택창 닫기" if st.session_state.selector_open else "품목 선택"
            if st.button(button_text, type="primary", width="stretch"):
                st.session_state.selector_open = not st.session_state.selector_open
                if not st.session_state.selector_open:
                    st.session_state.item_warning = None
                st.rerun()

    with selected_column:
        st.markdown(
            f'<div class="selected-card"><div class="selected-label">현재 선택 품목</div>'
            f'<div class="selected-value">{st.session_state.selected_item}</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.item_warning:
        st.markdown(
            f'<div class="item-warning">{st.session_state.item_warning}</div>',
            unsafe_allow_html=True,
        )

with panel_column:
    if st.session_state.selector_open:
        with st.container(border=True):
            header, close = st.columns([7, 1])
            with header:
                st.markdown(
                    '<div class="section-title">품목 선택</div>',
                    unsafe_allow_html=True,
                )
            with close:
                if st.button("✕", help="선택창 닫기"):
                    st.session_state.selector_open = False
                    st.session_state.item_warning = None
                    st.rerun()

            st.selectbox(
                "선택 품목",
                ITEMS,
                key="item_candidate",
            )
            apply_column, cancel_column = st.columns([2.2, 1])
            with apply_column:
                st.button(
                    "적용",
                    type="primary",
                    width="stretch",
                    on_click=apply_item_selection,
                )
            with cancel_column:
                if st.button("취소", width="stretch"):
                    st.session_state.selector_open = False
                    st.session_state.item_warning = None
                    st.rerun()
    else:
        st.info("왼쪽의 **품목 선택** 버튼을 누르면 선택창이 열립니다.")

if (
    st.session_state.item_confirmed
    and st.session_state.selected_item_code in ITEM_VARIABLE_CONFIG
):
    st.divider()
    render_item_variable_dashboard()
    st.stop()

if st.session_state.item_confirmed:
    st.divider()

    st.html(
        """
        <div id="variable-settings-anchor" style="height:0"></div>
        <script>
            if (!window.__dashboardScrollObserver) {
                const moveToDashboardAnchor = (marker) => {
                    const requestId = marker.dataset.scrollRequest;
                    if (
                        !requestId
                        || requestId === window.__lastDashboardScrollRequest
                    ) {
                        return;
                    }
                    const anchor = document.getElementById(
                        marker.dataset.scrollTarget
                    );
                    if (!anchor) {
                        return;
                    }
                    window.__lastDashboardScrollRequest = requestId;
                    const header = document.querySelector(
                        '[data-testid="stHeader"]'
                    );
                    const headerHeight = header
                        ? header.getBoundingClientRect().height
                        : 72;
                    const scrollCandidates = [
                        document.querySelector('[data-testid="stMain"]'),
                        document.querySelector(
                            '[data-testid="stAppViewContainer"]'
                        ),
                        document.scrollingElement
                    ];
                    const scroller = scrollCandidates.find(
                        (element) => element
                            && element.scrollHeight > element.clientHeight + 1
                    ) || document.scrollingElement;
                    const targetTop = (
                        scroller.scrollTop
                        + anchor.getBoundingClientRect().top
                        - headerHeight
                        - 18
                    );
                    scroller.scrollTo({
                        top: Math.max(0, targetTop),
                        behavior: "smooth"
                    });
                };
                const scanDashboardScrollRequests = () => {
                    document.querySelectorAll(
                        '[data-dashboard-scroll-request]'
                    ).forEach(moveToDashboardAnchor);
                };
                const observer = new MutationObserver(
                    scanDashboardScrollRequests
                );
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: [
                        "data-scroll-request",
                        "data-scroll-target"
                    ]
                });
                window.__dashboardScrollObserver = observer;
                scanDashboardScrollRequests();
            }
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    with st.container(border=True, key="analysis_workspace"):
        workspace_header_layout = st.container(key="workspace_header_layout")
        controls, prediction = workspace_header_layout.columns([1.35, 1], gap="medium")
        with controls:
            st.markdown(
                '<div class="section-title">주요 변수 설정</div>',
                unsafe_allow_html=True,
            )
            selected_variable = st.segmented_control(
                "주요 변수",
                VARIABLES,
                key="variable_selector",
                label_visibility="collapsed",
                disabled=not st.session_state.prediction_confirmed,
                on_change=enforce_variable_selection,
            )
            detail_options = [
                f"{st.session_state.selected_variable}{number}"
                for number in DATASET_NUMBERS
            ]
            expected_detail = (
                f"{st.session_state.selected_variable}"
                f"{dataset_number_for(st.session_state.selected_variable)}"
            )
            if st.session_state.get("dataset_selector") not in detail_options:
                st.session_state.dataset_selector = expected_detail

            with st.container(
                key="variable_navigation",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                selected_detail = st.selectbox(
                    "데이터 세트",
                    detail_options,
                    label_visibility="collapsed",
                    key="dataset_selector",
                    width=360,
                    disabled=not st.session_state.prediction_confirmed,
                )
                st.button(
                    "다음",
                    width=90,
                    on_click=advance_dataset,
                    disabled=not st.session_state.prediction_confirmed,
                )

            selected_dataset_number = int(selected_detail[-1])
            previous_dataset_number = dataset_number_for(
                st.session_state.selected_variable
            )
            st.session_state.variable_dataset_choices[
                st.session_state.selected_variable
            ] = selected_dataset_number
            if selected_dataset_number != previous_dataset_number:
                update_result_visibility_after_edit()
                st.session_state.next_stage_warning = None

        with prediction:
            st.markdown(
                '<div class="section-title">예측시점</div>',
                unsafe_allow_html=True,
            )
            with st.container(
                key="prediction_navigation",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                prediction_date = st.selectbox(
                    "예측시점",
                    PREDICTION_DATES,
                    key="prediction_candidate",
                    label_visibility="collapsed",
                    width=170,
                    on_change=mark_prediction_pending,
                )
                st.button(
                    "조회하기",
                    type="primary",
                    width=110,
                    on_click=apply_prediction_date,
                )
            if not st.session_state.prediction_confirmed:
                st.caption("조회하기를 눌러 예측시점을 먼저 확정해 주세요.")

        st.markdown(
            '<div class="section-title">그래프 및 시나리오 데이터</div>',
            unsafe_allow_html=True,
        )
        history_period = st.session_state.get("history_period") or "1년"
        active_dataset_number = dataset_number_for(
            st.session_state.selected_variable
        )
        scenarios = scenario_sets[active_dataset_number]
        context_key = context_key_for(st.session_state.selected_variable)
        chart_focus = st.session_state.chart_focus.setdefault(context_key, "final")
        analysis_content_layout = st.container(key="analysis_content_layout")
        graph_column, data_column = analysis_content_layout.columns([1.35, 1], gap="medium")
        with graph_column:
            forecast_quarters = five_quarters_from(st.session_state.applied_quarter)
            history_years = int(history_period[0])
            history_quarter_count = history_years * 4
            historical_quarters = [
                shift_quarter(st.session_state.applied_quarter, offset)
                for offset in range(-history_quarter_count, 0)
            ]
            if not st.session_state.prediction_confirmed:
                st.markdown(
                    """
                    <div class="chart-placeholder">
                        <strong>그래프 영역</strong>
                        <span>예측시점을 조회하면 그래프가 표시됩니다.</span>
                    </div>
                    <div class="chart-legend chart-legend-placeholder" aria-hidden="true">
                        <span class="chart-legend-item">base</span>
                        <span class="chart-legend-item">high</span>
                        <span class="chart-legend-item">low</span>
                        <span class="chart-legend-item">final</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                historical_values = values_for_quarters(
                    scenarios["base"],
                    st.session_state.selected_variable,
                    historical_quarters,
                )
                forecast_scenario_values = {
                    scenario_name: values_for_quarters(
                        scenarios[scenario_name],
                        st.session_state.selected_variable,
                        forecast_quarters,
                    )
                    for scenario_name in ("base", "high", "low")
                }
                committed_values = st.session_state.final_values.get(context_key, {})
                chart = create_trend_chart(
                    st.session_state.selected_variable,
                    tuple(historical_quarters),
                    tuple(historical_values),
                    tuple(forecast_quarters),
                    tuple(forecast_scenario_values["base"]),
                    tuple(forecast_scenario_values["high"]),
                    tuple(forecast_scenario_values["low"]),
                    tuple(committed_values.get(quarter) for quarter in forecast_quarters),
                    chart_focus,
                )
                st.altair_chart(chart, width="stretch")
                st.markdown(
                    """
                    <div class="chart-legend">
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#2563eb"></span>base
                        </span>
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#c026d3"></span>high
                        </span>
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#eab308"></span>low
                        </span>
                        <span class="chart-legend-item">
                            <span class="chart-legend-dot" style="background:#ef6461"></span>final
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.container(
                key="history_period_control",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                st.markdown(
                    '<span class="period-inline-label">기간</span>',
                    unsafe_allow_html=True,
                )
                st.segmented_control(
                    "기간",
                    options=["1년", "2년", "3년"],
                    default="1년",
                    key="history_period",
                    selection_mode="single",
                    width="content",
                    label_visibility="collapsed",
                )

        with data_column.container(key="scenario_data_panel"):
            st.caption(
                f"{st.session_state.selected_variable} · "
                f"예측시점 {st.session_state.applied_prediction_date} · "
                f"{st.session_state.applied_quarter}부터 5개 분기"
            )
            if st.session_state.prediction_confirmed:
                scenario_table = build_scenario_table(
                    scenarios,
                    st.session_state.selected_variable,
                    st.session_state.applied_quarter,
                )
            else:
                scenario_table = pd.DataFrame(
                    index=["high", "base", "low"],
                    columns=five_quarters_from(st.session_state.applied_quarter),
                    dtype=float,
                )
            committed_values = st.session_state.final_values.get(context_key, {})
            selected_for_context = st.session_state.selected_cells.setdefault(context_key, {})
            draft_for_context = st.session_state.final_drafts.get(context_key, {})

            with st.container(key="scenario_grid"):
                header_columns = st.columns([.82, 1, 1, 1, 1, 1], gap=None)
                header_columns[0].empty()
                for column, quarter in zip(header_columns[1:], scenario_table.columns):
                    column.markdown(
                        f'<div class="grid-header">{quarter}</div>',
                        unsafe_allow_html=True,
                    )

                for scenario_name in ("high", "base", "low"):
                    row_columns = st.columns([.82, 1, 1, 1, 1, 1], gap=None)
                    row_columns[0].button(
                        scenario_name,
                        key=f"chart_focus_{scenario_name}",
                        type=(
                            "primary"
                            if chart_focus == scenario_name
                            else "secondary"
                        ),
                        width="stretch",
                        on_click=select_scenario_row,
                        args=(
                            context_key,
                            scenario_name,
                            tuple(scenario_table.columns),
                        ),
                    )
                    for column, quarter in zip(row_columns[1:], scenario_table.columns):
                        value = scenario_table.loc[scenario_name, quarter]
                        label = "—" if pd.isna(value) else f"{value:,.0f}"
                        is_selected = selected_for_context.get(quarter) == scenario_name
                        has_uncommitted_change = (
                            is_selected
                            and quarter in draft_for_context
                            and not values_match(value, draft_for_context.get(quarter))
                        )
                        button_type = (
                            "tertiary"
                            if has_uncommitted_change
                            else "primary"
                            if is_selected
                            else "secondary"
                        )
                        column.button(
                            label,
                            key=f"cell_{context_key}_{quarter}_{scenario_name}",
                            type=button_type,
                            width="stretch",
                            on_click=select_scenario_cell,
                            args=(context_key, quarter, scenario_name),
                        )

            with st.container(key="commit_row"):
                commit_space, commit_button = st.columns([4.4, 1], gap="small")
                with commit_button:
                    st.button(
                        "가져오기",
                        type="primary",
                        width="stretch",
                        on_click=import_selected_values,
                        args=(context_key, list(scenario_table.columns), scenario_table),
                    )

                with commit_space:
                    if (
                        st.session_state.table_warning
                        and not st.session_state.final_drafts.get(context_key)
                    ):
                        st.markdown(
                            f'<div class="commit-warning">{st.session_state.table_warning}</div>',
                            unsafe_allow_html=True,
                        )

            committed_values = st.session_state.final_values.get(context_key, {})
            draft_values = (
                st.session_state.final_drafts.setdefault(context_key, {})
                if st.session_state.prediction_confirmed
                else {}
            )
            if (
                st.session_state.prediction_confirmed
                and st.session_state.active_final_input_context != context_key
            ):
                for quarter in scenario_table.columns:
                    restored_value = draft_values.get(
                        quarter,
                        committed_values.get(quarter),
                    )
                    st.session_state[final_input_key(context_key, quarter)] = (
                        ""
                        if restored_value is None or pd.isna(restored_value)
                        else f"{float(restored_value):.0f}"
                    )
                st.session_state.active_final_input_context = context_key
            st.markdown('<div class="final-wrap"></div>', unsafe_allow_html=True)
            final_grid = st.container(key="final_grid")
            final_columns = final_grid.columns([.82, 1, 1, 1, 1, 1], gap=None)
            final_columns[0].button(
                "final",
                key="chart_focus_final",
                type="primary" if chart_focus == "final" else "secondary",
                width="stretch",
                disabled=not st.session_state.prediction_confirmed,
                on_click=set_chart_focus,
                args=(context_key, "final"),
            )
            if not st.session_state.prediction_confirmed:
                for index, column in enumerate(final_columns[1:]):
                    column.button(
                        "—",
                        key=f"final_locked_{index}",
                        type="secondary",
                        width="stretch",
                        on_click=warn_prediction_required,
                    )
            else:
                for index, (column, quarter) in enumerate(
                    zip(final_columns[1:], scenario_table.columns)
                ):
                    input_key = final_input_key(context_key, quarter)
                    if input_key not in st.session_state:
                        initial_value = draft_values.get(
                            quarter,
                            committed_values.get(quarter),
                        )
                        st.session_state[input_key] = (
                            "" if initial_value is None else f"{float(initial_value):.0f}"
                        )
                    raw_value = str(st.session_state[input_key]).strip()
                    current_value = draft_values.get(
                        quarter,
                        committed_values.get(quarter),
                    )
                    committed_value = committed_values.get(quarter)
                    is_dirty = bool(raw_value) and (
                        current_value is None
                        or committed_value is None
                        or pd.isna(committed_value)
                        or not values_match(current_value, committed_value)
                    )
                    cell_state = (
                        "dirty"
                        if is_dirty
                        else "clean"
                        if current_value is not None
                        else "empty"
                    )
                    with column:
                        with st.container(key=f"final_{cell_state}_{index}"):
                            st.text_input(
                                quarter,
                                key=input_key,
                                placeholder="—",
                                label_visibility="collapsed",
                                on_change=update_final_draft,
                                args=(context_key, quarter),
                            )

            with st.container(key="final_commit_row"):
                warning_space, final_commit_button = st.columns([4.4, 1], gap="small")
                with warning_space:
                    if (
                        st.session_state.table_warning
                        and st.session_state.prediction_confirmed
                        and st.session_state.final_drafts.get(context_key)
                    ):
                        st.markdown(
                            f'<div class="commit-warning">{st.session_state.table_warning}</div>',
                            unsafe_allow_html=True,
                        )
                with final_commit_button:
                    st.button(
                        "commit",
                        type="primary",
                        width="stretch",
                        on_click=commit_final_draft,
                        args=(context_key, list(scenario_table.columns)),
                    )

            if st.session_state.final_commit_notice:
                st.toast("final 값이 반영되었습니다.", icon="✅")
                st.session_state.final_commit_notice = False

    st.space("small")
    with st.container(key="next_stage_action"):
        next_stage_controls = st.container(key="next_stage_controls")
        next_warning, quick_button, next_button = next_stage_controls.columns([2.5, .75, 1.2])
        with next_warning:
            if st.session_state.next_stage_warning:
                st.markdown(
                    (
                        '<div id="variable-warning-anchor" class="commit-warning">'
                        f"{st.session_state.next_stage_warning}</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if st.session_state.scroll_to_variable_settings:
                    st.html(
                        """
                        <script>
                            setTimeout(() => {
                                const warning = document.getElementById(
                                    "variable-warning-anchor"
                                );
                                if (warning) {
                                    warning.scrollIntoView({
                                        behavior: "smooth",
                                        block: "end"
                                    });
                                    setTimeout(() => {
                                        window.scrollBy({
                                            top: 20,
                                            behavior: "smooth"
                                        });
                                    }, 180);
                                }
                            }, 120);
                        </script>
                        """,
                        unsafe_allow_javascript=True,
                    )
                    st.session_state.scroll_to_variable_settings = False
        with quick_button:
            st.button(
                "원클릭 설정",
                width="stretch",
                on_click=prepare_one_click_test,
                help="테스트용 임의 데이터로 모든 주요 변수의 commit을 완료합니다.",
            )
        with next_button:
            st.button(
                f"{st.session_state.selected_item} 가격 예측하기",
                type="primary",
                width="stretch",
                on_click=proceed_to_next_stage,
                disabled=not st.session_state.prediction_confirmed,
            )

    if st.session_state.one_click_notice:
        st.toast("테스트용 임의 데이터의 commit이 완료되었습니다.", icon="✅")
        st.session_state.one_click_notice = False
    if st.session_state.scroll_after_one_click:
        st.html(
            (
                '<span hidden data-dashboard-scroll-request '
                'data-scroll-target="variable-settings-anchor" '
                f'data-scroll-request="one-click-'
                f'{st.session_state.one_click_scroll_nonce}"></span>'
            )
        )
        st.session_state.scroll_after_one_click = False

    if st.session_state.show_next_stage:
        if st.session_state.scroll_to_next_stage:
            st.html(
                (
                    '<div id="next-stage-anchor" '
                    'style="height:0"></div>'
                    '<span hidden data-dashboard-scroll-request '
                    'data-scroll-target="next-stage-anchor" '
                    f'data-scroll-request="next-stage-'
                    f'{st.session_state.next_stage_scroll_nonce}"></span>'
                )
            )
            st.session_state.scroll_to_next_stage = False

        with st.container(border=True, key="next_stage_interface"):
            st.markdown(
                '<div class="section-title">가격 예측 결과</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "예측시점이 다른 데이터셋은 공통 분기 축에 정렬하며, "
                "그래프 강조를 바꾸면 선택한 데이터셋의 예측시점과 실제가 구간도 함께 적용됩니다."
            )
            forecast_column = (
                st.session_state.forecast_column
                or random.choice(FORECAST_COLUMNS)
            )
            st.session_state.forecast_column = forecast_column

            forecast_history_period = (
                st.session_state.get("forecast_history_period") or "1년"
            )
            history_years = int(forecast_history_period[0])
            history_quarter_count = history_years * 4
            forecast_start = st.session_state.applied_quarter
            forecast_quarters = five_quarters_from(forecast_start)
            saved_forecast_datasets = (
                st.session_state.saved_forecast_datasets
            )
            dataset1_snapshot = saved_forecast_datasets.get("데이터셋1")
            dataset2_snapshot = saved_forecast_datasets.get("데이터셋2")
            dataset_focus = tuple(
                st.session_state.get("comparison_dataset_toggle") or ()
            )
            dataset1_start = (
                dataset1_snapshot.get("quarter", forecast_start)
                if dataset1_snapshot is not None
                else forecast_start
            )
            dataset2_start = (
                dataset2_snapshot.get("quarter", forecast_start)
                if dataset2_snapshot is not None
                else forecast_start
            )
            dataset1_quarters = (
                five_quarters_from(dataset1_start)
                if dataset1_snapshot is not None
                else []
            )
            dataset2_quarters = (
                five_quarters_from(dataset2_start)
                if dataset2_snapshot is not None
                else []
            )
            comparison_starts = [forecast_start]
            if dataset1_snapshot is not None:
                comparison_starts.append(dataset1_start)
            if dataset2_snapshot is not None:
                comparison_starts.append(dataset2_start)
            fixed_comparison_axis = quarters_between(
                shift_quarter(
                    min(comparison_starts, key=quarter_key),
                    -history_quarter_count,
                ),
                shift_quarter(
                    max(comparison_starts, key=quarter_key),
                    4,
                ),
            )
            selected_starts = []
            if "현재 그래프" in dataset_focus:
                selected_starts.append(forecast_start)
            if (
                "데이터셋1" in dataset_focus
                and dataset1_snapshot is not None
            ):
                selected_starts.append(dataset1_start)
            if (
                "데이터셋2" in dataset_focus
                and dataset2_snapshot is not None
            ):
                selected_starts.append(dataset2_start)

            earliest_comparison_start = min(
                comparison_starts,
                key=quarter_key,
            )
            latest_comparison_start = max(
                comparison_starts,
                key=quarter_key,
            )
            actual_reference_start = (
                max(selected_starts, key=quarter_key)
                if selected_starts
                else earliest_comparison_start
            )
            historical_quarters = quarters_between(
                fixed_comparison_axis[0],
                shift_quarter(actual_reference_start, -1),
            )
            muted_historical_quarters = (
                quarters_between(
                    shift_quarter(actual_reference_start, -1),
                    shift_quarter(latest_comparison_start, -1),
                )
                if (
                    not selected_starts
                    and quarter_key(actual_reference_start)
                    < quarter_key(latest_comparison_start)
                )
                else []
            )

            price_historical_values = values_for_quarters(
                price_forecast_data,
                "실제가",
                historical_quarters,
            )
            muted_price_historical_values = values_for_quarters(
                price_forecast_data,
                "실제가",
                muted_historical_quarters,
            )
            forecast_values = values_for_quarters(
                price_forecast_data,
                forecast_column,
                forecast_quarters,
            )
            dataset1_name = escape(dataset_display_name("데이터셋1"))
            dataset2_name = escape(dataset_display_name("데이터셋2"))
            dataset1_price_values = tuple(
                (
                    dataset1_snapshot.get(
                        "forecast_values",
                        {},
                    ).get(quarter)
                    if dataset1_snapshot is not None
                    else None
                )
                for quarter in dataset1_quarters
            )
            dataset2_price_values = tuple(
                (
                    dataset2_snapshot.get(
                        "forecast_values",
                        {},
                    ).get(quarter)
                    if dataset2_snapshot is not None
                    else None
                )
                for quarter in dataset2_quarters
            )
            comparison_price_actual_values = values_for_quarters(
                price_forecast_data,
                "실제가",
                fixed_comparison_axis,
            )
            fixed_price_y_domain = padded_y_domain(
                comparison_price_actual_values,
                forecast_values,
                dataset1_price_values,
                dataset2_price_values,
            )
            price_chart = create_price_forecast_chart(
                st.session_state.selected_item,
                tuple(historical_quarters),
                tuple(price_historical_values),
                tuple(forecast_quarters),
                tuple(forecast_values),
                False,
                285,
                dataset1_price_values,
                dataset_focus,
                dataset2_price_values,
                tuple(dataset1_quarters),
                tuple(dataset2_quarters),
                tuple(fixed_comparison_axis),
                fixed_price_y_domain,
                tuple(muted_historical_quarters),
                tuple(muted_price_historical_values),
                False,
            )

            current_forecast_dataset = {
                "item": st.session_state.selected_item,
                "prediction_date": st.session_state.applied_prediction_date,
                "quarter": st.session_state.applied_quarter,
                "variable_choices": deepcopy(
                    st.session_state.variable_dataset_choices
                ),
                "final_values": deepcopy(st.session_state.final_values),
                "forecast_column": forecast_column,
                "forecast_values": dict(
                    zip(forecast_quarters, forecast_values)
                ),
            }

            comparison_layout = st.container(
                key="forecast_comparison_layout",
            )
            graph_area, dataset_panel = comparison_layout.columns(
                [4.1, 1],
                gap="small",
            )
            graph_grid = graph_area.container(key="forecast_graph_grid")
            variable_graph, price_graph = graph_grid.columns(
                2,
                gap="small",
            )
            with variable_graph:
                with st.container(
                    border=True,
                    key="variable_forecast_card",
                ):
                    with st.container(
                        key="variable_forecast_header",
                        horizontal=True,
                        gap="small",
                        vertical_alignment="center",
                    ):
                        st.markdown(
                            '<div class="forecast-card-title">• 주요 변수 전망</div>',
                            unsafe_allow_html=True,
                        )
                        forecast_variable = st.segmented_control(
                            "주요 변수",
                            options=VARIABLES,
                            default=VARIABLES[0],
                            key="forecast_variable_selector",
                            selection_mode="single",
                            width="content",
                            label_visibility="collapsed",
                        )
                    selected_forecast_variable = (
                        forecast_variable or VARIABLES[0]
                    )
                    variable_dataset_number = dataset_number_for(
                        selected_forecast_variable
                    )
                    historical_dataset_number = variable_dataset_number
                    variable_context_key = context_key_for(
                        selected_forecast_variable
                    )
                    variable_historical_values = values_for_quarters(
                        scenario_sets[historical_dataset_number]["base"],
                        selected_forecast_variable,
                        historical_quarters,
                    )
                    muted_variable_historical_values = values_for_quarters(
                        scenario_sets[historical_dataset_number]["base"],
                        selected_forecast_variable,
                        muted_historical_quarters,
                    )
                    variable_committed_values = st.session_state.final_values.get(
                        variable_context_key,
                        {},
                    )
                    dataset1_variable_values = snapshot_final_values(
                        dataset1_snapshot,
                        selected_forecast_variable,
                        dataset1_quarters,
                    )
                    dataset2_variable_values = (
                        snapshot_final_values(
                            dataset2_snapshot,
                            selected_forecast_variable,
                            dataset2_quarters,
                        )
                        if dataset2_snapshot is not None
                        else ()
                    )
                    current_variable_values = tuple(
                        variable_committed_values.get(quarter)
                        for quarter in forecast_quarters
                    )
                    comparison_variable_actual_values = [
                        values_for_quarters(
                            scenario_sets[variable_dataset_number]["base"],
                            selected_forecast_variable,
                            fixed_comparison_axis,
                        )
                    ]
                    for snapshot in (dataset1_snapshot, dataset2_snapshot):
                        if snapshot is None:
                            continue
                        snapshot_dataset_number = snapshot.get(
                            "variable_choices",
                            {},
                        ).get(
                            selected_forecast_variable,
                            variable_dataset_number,
                        )
                        comparison_variable_actual_values.append(
                            values_for_quarters(
                                scenario_sets[snapshot_dataset_number]["base"],
                                selected_forecast_variable,
                                fixed_comparison_axis,
                            )
                        )
                    fixed_variable_y_domain = padded_y_domain(
                        *comparison_variable_actual_values,
                        current_variable_values,
                        dataset1_variable_values,
                        dataset2_variable_values,
                    )
                    empty_scenario_values = tuple(
                        None for _ in forecast_quarters
                    )
                    variable_chart = create_trend_chart(
                        selected_forecast_variable,
                        tuple(historical_quarters),
                        tuple(variable_historical_values),
                        tuple(forecast_quarters),
                        empty_scenario_values,
                        empty_scenario_values,
                        empty_scenario_values,
                        current_variable_values,
                        "final",
                        False,
                        285,
                        dataset1_variable_values,
                        dataset_focus,
                        dataset2_variable_values,
                        tuple(dataset1_quarters),
                        tuple(dataset2_quarters),
                        tuple(fixed_comparison_axis),
                        fixed_variable_y_domain,
                        tuple(muted_historical_quarters),
                        tuple(muted_variable_historical_values),
                        False,
                        True,
                    )
                    st.altair_chart(
                        variable_chart,
                        key="comparison_variable_chart",
                        width="stretch",
                    )
                    variable_legend_items = ""
                    if dataset1_snapshot is not None:
                        variable_legend_items += (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" style="background:#2563EB"></span>'
                            f'{dataset1_name}</span>'
                        )
                    if dataset2_snapshot is not None or any(
                        value is not None for value in dataset2_variable_values
                    ):
                        variable_legend_items += (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" style="background:#CA8A04"></span>'
                            f'{dataset2_name}</span>'
                        )
                    variable_legend_items += (
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" style="background:#ef6461"></span>'
                        '현재 데이터</span>'
                    )
                    st.markdown(
                        f'<div class="chart-legend">{variable_legend_items}</div>',
                        unsafe_allow_html=True,
                    )

            with price_graph:
                with st.container(
                    border=True,
                    key="price_forecast_card",
                ):
                    with st.container(
                        key="price_forecast_header",
                        horizontal=True,
                        gap="small",
                        vertical_alignment="center",
                    ):
                        st.markdown(
                            '<div class="forecast-card-title">• 전기동 가격 전망</div>',
                            unsafe_allow_html=True,
                        )
                    st.altair_chart(
                        price_chart,
                        key="comparison_price_chart",
                        width="stretch",
                    )
                    price_legend_items = ""
                    if dataset1_snapshot is not None:
                        price_legend_items += (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" style="background:#2563EB"></span>'
                            f'{dataset1_name}</span>'
                        )
                    if dataset2_snapshot is not None:
                        price_legend_items += (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" style="background:#CA8A04"></span>'
                            f'{dataset2_name}</span>'
                        )
                    price_legend_items += (
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" style="background:#ef6461"></span>'
                        '현재 데이터</span>'
                    )
                    st.markdown(
                        f'<div class="chart-legend">{price_legend_items}</div>',
                        unsafe_allow_html=True,
                    )

            with dataset_panel:
                with st.container(
                    border=True,
                    key="dataset_comparison_panel",
                    height="stretch",
                ):
                    st.markdown(
                        '<div class="dataset-panel-title">• 데이터셋 비교</div>'
                        '<div class="dataset-panel-caption">'
                        '현재 결과를 저장할 슬롯과 강조할 데이터셋을 선택합니다.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    save_target_widget_key = (
                        "dataset_save_target_widget_"
                        f"{st.session_state.dataset_save_widget_version}"
                    )
                    if save_target_widget_key not in st.session_state:
                        st.session_state[save_target_widget_key] = (
                            st.session_state.dataset_save_target
                        )
                    save_target = st.selectbox(
                        "저장 위치",
                        options=["데이터셋1", "데이터셋2"],
                        format_func=dataset_display_name,
                        key=save_target_widget_key,
                        on_change=sync_dataset_save_target,
                        args=(save_target_widget_key,),
                        width="stretch",
                    )
                    st.session_state.dataset_save_target = save_target
                    st.button(
                        "데이터셋 저장",
                        key="dataset_save_button",
                        icon=":material/save:",
                        width="stretch",
                        on_click=save_forecast_dataset,
                        args=(current_forecast_dataset,),
                    )

                    with st.popover(
                        "데이터셋 관리",
                        icon=":material/edit:",
                        width="stretch",
                    ):
                        st.text_input(
                            "데이터셋1 이름",
                            key="dataset_name_input_1",
                            max_chars=16,
                        )
                        st.text_input(
                            "데이터셋2 이름",
                            key="dataset_name_input_2",
                            max_chars=16,
                        )
                        if st.button(
                            "이름 적용",
                            key="dataset_name_apply",
                            width="stretch",
                        ):
                            name1 = st.session_state.dataset_name_input_1.strip()
                            name2 = st.session_state.dataset_name_input_2.strip()
                            if not name1 or not name2:
                                st.session_state.dataset_manage_notice = (
                                    "데이터셋 이름을 모두 입력해 주세요."
                                )
                            elif name1 == name2:
                                st.session_state.dataset_manage_notice = (
                                    "서로 다른 데이터셋 이름을 입력해 주세요."
                                )
                            else:
                                st.session_state.dataset_display_names = {
                                    "데이터셋1": name1,
                                    "데이터셋2": name2,
                                }
                                st.session_state.dataset_save_widget_version += 1
                                st.session_state.dataset_manage_notice = (
                                    "데이터셋 이름을 변경했습니다."
                                )
                            st.rerun()

                        st.caption("삭제한 저장 내용은 복구할 수 없습니다.")
                        if st.button(
                            f"{dataset_display_name('데이터셋1')} 내용 삭제",
                            key="delete_dataset_1",
                            width="stretch",
                        ):
                            st.session_state.saved_forecast_datasets[
                                "데이터셋1"
                            ] = None
                            st.session_state.saved_forecast_dataset = None
                            remaining_focuses = [
                                focus
                                for focus in st.session_state.comparison_dataset_toggle
                                if focus != "데이터셋1"
                            ]
                            set_comparison_focus(remaining_focuses)
                            st.session_state.dataset_save_notice = False
                            st.session_state.dataset_manage_notice = (
                                f"{dataset_display_name('데이터셋1')} 내용을 삭제했습니다."
                            )
                            st.rerun()
                        if st.button(
                            f"{dataset_display_name('데이터셋2')} 내용 삭제",
                            key="delete_dataset_2",
                            width="stretch",
                        ):
                            st.session_state.saved_forecast_datasets[
                                "데이터셋2"
                            ] = None
                            remaining_focuses = [
                                focus
                                for focus in st.session_state.comparison_dataset_toggle
                                if focus != "데이터셋2"
                            ]
                            set_comparison_focus(remaining_focuses)
                            st.session_state.dataset_save_notice = False
                            st.session_state.dataset_manage_notice = (
                                f"{dataset_display_name('데이터셋2')} 내용을 삭제했습니다."
                            )
                            st.rerun()

                    dataset1_state = (
                        "저장됨"
                        if dataset1_snapshot is not None
                        else "비어 있음"
                    )
                    dataset2_state = (
                        "저장됨"
                        if dataset2_snapshot is not None
                        else "비어 있음"
                    )
                    dataset1_class = (
                        "dataset-slot current"
                        if save_target == "데이터셋1"
                        else "dataset-slot"
                    )
                    dataset2_class = (
                        "dataset-slot current"
                        if save_target == "데이터셋2"
                        else "dataset-slot"
                    )
                    st.markdown(
                        f"""
                        <div class="{dataset1_class}">
                            <span class="dataset-slot-name">{dataset1_name}</span>
                            <span class="dataset-slot-state">{dataset1_state}</span>
                        </div>
                        <div class="{dataset2_class}">
                            <span class="dataset-slot-name">{dataset2_name}</span>
                            <span class="dataset-slot-state">{dataset2_state}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("그래프 강조")
                    st.pills(
                        "현재 그래프 강조",
                        options=["현재 그래프"],
                        key="comparison_current_focus",
                        selection_mode="multi",
                        label_visibility="collapsed",
                        on_change=sync_comparison_focus,
                        width="stretch",
                    )
                    st.pills(
                        "저장 데이터셋 강조",
                        options=["데이터셋1", "데이터셋2"],
                        format_func=dataset_display_name,
                        key="comparison_saved_focus",
                        selection_mode="multi",
                        label_visibility="collapsed",
                        on_change=sync_comparison_focus,
                        width="stretch",
                    )
                    dataset_save_notice = (
                        "현재 결과를 "
                        f"{dataset_display_name(st.session_state.dataset_save_notice)}에 저장했습니다."
                        if st.session_state.dataset_save_notice
                        else ""
                    )
                    st.markdown(
                        '<div class="dataset-save-notice">'
                        f"{escape(dataset_save_notice) or '&nbsp;'}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.session_state.dataset_manage_notice:
                        st.caption(st.session_state.dataset_manage_notice)

            with graph_area:
                with st.container(
                    key="forecast_period_control",
                    horizontal=True,
                    vertical_alignment="center",
                    gap="small",
                ):
                    st.markdown(
                        '<span class="period-inline-label">기간</span>',
                        unsafe_allow_html=True,
                    )
                    st.segmented_control(
                        "기간",
                        options=["1년", "2년", "3년"],
                        default="1년",
                        key="forecast_history_period",
                        selection_mode="single",
                        width="content",
                        label_visibility="collapsed",
                    )
else:
    st.divider()
    st.caption("품목을 선택하고 적용하면 주요 변수 설정 화면이 표시됩니다.")
