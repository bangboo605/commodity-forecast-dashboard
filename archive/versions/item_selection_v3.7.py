# Version 3.7
from copy import deepcopy
from html import escape
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from _compat import (
    GOLD_PRICE_PATH,
    ITEM_DATA_DIRS,
    SILVER_PRICE_PATH,
)
from gold_price_model import predict_final_gold_prices


st.set_page_config(
    page_title="원자재 가격 대시보드",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ITEMS = ["알루미늄", "전기동", "금", "은"]
ITEM_CODES = {
    "알루미늄": "aluminum",
    "전기동": "copper",
    "금": "gold",
    "은": "silver",
}
SELECTED_ITEM_LABELS = {code: label for label, code in ITEM_CODES.items()}
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
SILVER_PREDICTION_DATES = ("2026-01-21", "2026-02-21", "2026-03-21")
COPPER_PREDICTION_DATES = ("2026-01-21", "2026-02-21", "2026-03-21")
ALUMINUM_PREDICTION_DATES = ("2026-01-21", "2026-02-21", "2026-03-21")
GOLD_BACKTEST_PREDICTION_DATES = (
    "2023-04-21", "2023-05-21", "2023-06-21", "2023-07-21", "2023-08-21",
    "2023-09-21", "2023-10-21", "2023-11-21", "2023-12-21",
    "2024-01-21", "2024-02-21", "2024-03-21", "2024-04-21", "2024-05-21",
    "2024-06-21",
)
SILVER_VARIABLE_LABELS = {
    "ind_demand_yoy": "은 산업수요 YoY",
    "solar_demand_4q": "태양광(PV) 수요 4분기 합",
    "mine_supply_yoy": "광산 공급 YoY",
    "X2": "X2",
    "dxy_yoy": "달러 인덱스 YoY",
}
SILVER_VARIABLE_BUTTON_LABELS = {
    "ind_demand_yoy": "산업수요 YoY",
    "solar_demand_4q": "태양광 수요",
    "mine_supply_yoy": "광산 공급",
    "X2": "X2",
    "dxy_yoy": "달러 인덱스",
}
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
        "prediction_dates": GOLD_BACKTEST_PREDICTION_DATES,
    },
    "silver": {
        "actual_csv": "silver_major_variables.csv",
        "forecast_csv": "silver_major_variables_forecast.csv",
        "labels": SILVER_VARIABLE_BUTTON_LABELS,
        "full_labels": SILVER_VARIABLE_LABELS,
        "prediction_dates": SILVER_PREDICTION_DATES,
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
        "prediction_dates": COPPER_PREDICTION_DATES,
    },
    "aluminum": {
        "actual_csv": "aluminum_major_variables.csv",
        "forecast_csv": "aluminum_major_variables_forecast.csv",
        "labels": {
            "demand": "수요",
            "supply": "공급",
            "energy_cost": "에너지 비용",
        },
        "prediction_dates": ALUMINUM_PREDICTION_DATES,
    },
}
PRICE_PATHS = {"gold": GOLD_PRICE_PATH, "silver": SILVER_PRICE_PATH}


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


def date_to_quarter(date_text: str) -> str:
    date = pd.Timestamp(date_text)
    return f"{date.year}-Q{((date.month - 1) // 3) + 1}"


def five_quarters_from(start_quarter: str, count: int = 5) -> list[str]:
    year, quarter = quarter_key(start_quarter)
    quarters = []
    for offset in range(count):
        zero_based = (quarter - 1) + offset
        quarters.append(f"{year + zero_based // 4}-Q{zero_based % 4 + 1}")
    return quarters


def shift_quarter(quarter_label_text: str, amount: int) -> str:
    year, quarter = quarter_key(quarter_label_text)
    zero_based = (quarter - 1) + amount
    return f"{year + zero_based // 4}-Q{zero_based % 4 + 1}"




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


def compute_gold_price_final(
    applied_quarter: str, quarters: list[str]
) -> dict[str, float]:
    """TVP-model price prediction from committed final values (gold only)."""
    raw_final_values = {
        variable: st.session_state.gold_final_values.get(
            gold_context_key(variable, applied_quarter), {}
        )
        for variable in GOLD_VARIABLES
    }
    return predict_final_gold_prices(
        raw_final_values,
        quarters,
        ITEM_VARIABLE_CONFIG["gold"]["actual_column_aliases"],
    )


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


def hide_price_stage_unless_dataset_saved() -> None:
    """저장된 데이터셋(1 또는 2)이 있으면 가격 예측 결과 영역을 계속 보여준다 —
    저장된 비교 결과는 예측시점을 바꾸거나 final을 다시 만지는 중에도 여전히 유효하므로."""
    saved_slots = st.session_state.get("saved_forecast_datasets", {})
    if not any(saved_slots.get(slot) for slot in ("데이터셋1", "데이터셋2")):
        st.session_state.item_price_stage_visible = False


def mark_gold_prediction_pending() -> None:
    st.session_state.gold_prediction_confirmed = False
    hide_price_stage_unless_dataset_saved()
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
    forecast_quarters = ITEM_VARIABLE_CONFIG.get(item_code, {}).get("forecast_quarters", 7)
    quarters = five_quarters_from(st.session_state.gold_applied_quarter, forecast_quarters)
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
        st.session_state.gold_chart_focus[context_key] = {"final"}
        st.session_state.gold_final_input_versions[context_key] = (
            st.session_state.gold_final_input_versions.get(context_key, 0) + 1
        )
    st.session_state.gold_active_final_input_context = None
    st.session_state.gold_table_warning = None
    hide_price_stage_unless_dataset_saved()
    st.session_state.item_price_stage_warning = None
    st.session_state.gold_variable_data_scroll = True
    st.session_state.gold_variable_data_scroll_nonce = (
        st.session_state.get("gold_variable_data_scroll_nonce", 0) + 1
    )


def prepare_gold_one_click_test() -> None:
    """Randomly commit every variable so the price-prediction screen can be tested quickly."""
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.item_price_stage_warning = (
            "예측시점을 먼저 조회해 주세요."
        )
        return

    item_code = st.session_state.get("selected_item_code", "gold")
    config = ITEM_VARIABLE_CONFIG[item_code]

    _, gold_forecast_data, variables, _ = load_configured_item_data(item_code)
    quarters = five_quarters_from(
        st.session_state.gold_applied_quarter, config.get("forecast_quarters", 7)
    )

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
        st.session_state.gold_chart_focus[context_key] = {"final"}
        st.session_state.gold_final_input_versions[context_key] = (
            st.session_state.gold_final_input_versions.get(context_key, 0) + 1
        )

    st.session_state.gold_active_final_input_context = None
    st.session_state.gold_table_warning = None
    hide_price_stage_unless_dataset_saved()
    st.session_state.item_price_stage_warning = None
    if variables:
        st.session_state.gold_variable_selector = variables[0]


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
    active = st.session_state.gold_chart_focus.setdefault(context_key, {"final"})
    if series_name in active:
        active.discard(series_name)
    else:
        active.add(series_name)


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
    hide_price_stage_unless_dataset_saved()
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


def adjust_gold_final_draft(
    context_key: str, quarters: tuple[str, ...], percent: float
) -> None:
    """지금 화면에 적힌 final 값(draft, 없으면 committed) 기준으로 분기마다 ±percent% 조정.

    절대값 기준으로 가감한다(값 * (1+percent/100)이 아님) — 음수 값에 곱셈으로 %를
    적용하면 "+10%"가 오히려 더 음수 방향으로 가버리는 문제가 있어서, high/low
    시나리오 생성 때와 같은 방식(값 +- |값|*비율)으로 통일했다.
    값이 아예 없는 분기는 건드리지 않는다 — 조정이지 채워넣기가 아니므로.
    """
    if not st.session_state.gold_prediction_confirmed:
        return
    draft = st.session_state.gold_final_drafts.setdefault(context_key, {})
    committed = st.session_state.gold_final_values.get(context_key, {})
    adjusted = {
        quarter: float(current) + abs(float(current)) * (percent / 100)
        for quarter in quarters
        if (current := draft.get(quarter, committed.get(quarter))) is not None
    }
    if not adjusted:
        st.session_state.gold_table_warning = (
            "조정할 final 값이 없습니다. 먼저 값을 입력해 주세요."
        )
        return
    draft.update(adjusted)
    st.session_state.gold_final_input_versions[context_key] = (
        st.session_state.gold_final_input_versions.get(context_key, 0) + 1
    )
    st.session_state.gold_active_final_input_context = None
    hide_price_stage_unless_dataset_saved()
    st.session_state.gold_table_warning = "변경 내용을 commit해 주세요."


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
    st.session_state.gold_chart_focus[context_key] = {"final"}
    st.session_state.gold_table_warning = None
    st.session_state.gold_final_commit_notice = True
    # commit 버튼이 st.fragment(render_gold_data_panel) 안에 있어서, 기본 rerun은
    # fragment만 다시 그림 — fragment 밖의 final 그래프까지 갱신하려면 전체 rerun이
    # 필요한데, st.rerun()은 콜백 안에서는 no-op이라 여기서 바로 호출할 수 없다.
    # 플래그만 세워두고, fragment 본문(렌더 중)에서 대신 rerun을 건다.
    st.session_state.gold_final_commit_rerun_pending = True


def incomplete_item_variables(
    variables: tuple[str, ...],
    applied_quarter: str,
    item_code: str = "gold",
) -> list[str]:
    """Return variables whose final values are missing or not committed."""
    forecast_quarters = ITEM_VARIABLE_CONFIG.get(item_code, {}).get("forecast_quarters", 7)
    quarters = five_quarters_from(applied_quarter, forecast_quarters)
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


def mismatched_item_variables(
    variables: tuple[str, ...],
    applied_quarter: str,
    item_code: str = "gold",
) -> list[str]:
    """Return variables whose selected table cell differs from the committed final value."""
    _, forecast_data, _, item_config = load_configured_item_data(item_code)
    quarters = five_quarters_from(applied_quarter, item_config.get("forecast_quarters", 7))
    mismatched: list[str] = []
    for variable in variables:
        context_key = gold_context_key(variable, applied_quarter)
        selected = st.session_state.gold_selected_cells.get(context_key, {})
        committed = st.session_state.gold_final_values.get(context_key, {})
        for quarter in quarters:
            scenario_name = selected.get(quarter)
            if scenario_name is None:
                continue
            value = gold_forecast_value_for(forecast_data, quarter, scenario_name, variable)
            if value is None or pd.isna(value):
                continue
            committed_value = committed.get(quarter)
            if committed_value is None or pd.isna(committed_value):
                continue
            if not values_match(value, committed_value):
                mismatched.append(variable)
                break
    return mismatched


def proceed_to_item_price_stage() -> None:
    if not st.session_state.gold_prediction_confirmed:
        st.session_state.item_price_stage_warning = (
            "예측시점을 먼저 조회해 주세요."
        )
        return

    variables = tuple(st.session_state.get("active_item_variables", ()))
    mismatched = mismatched_item_variables(
        variables, st.session_state.gold_applied_quarter, st.session_state.selected_item_code
    )
    # 불일치는 진행을 막지 않고 경고만 띄운 채로 계속 진행한다.
    mismatch_warning = (
        "선택된 데이터와 final 값이 다릅니다. 가져오기를 눌러 반영할 수 있습니다."
        if mismatched
        else None
    )

    incomplete = incomplete_item_variables(
        variables,
        st.session_state.gold_applied_quarter,
        st.session_state.selected_item_code,
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

    st.session_state.item_price_stage_warning = mismatch_warning
    st.session_state.item_price_stage_visible = True
    st.session_state.item_price_stage_scroll = True
    # 결과 화면의 "주요 변수 전망" pill이 방금 위에서 보고 있던 변수를 그대로 이어받도록
    # 동기화 — 안 그러면 항상 첫 번째 변수로 리셋돼서 안 눌린 것처럼 보임.
    active_variable = st.session_state.get("gold_variable_selector")
    if active_variable in variables:
        st.session_state["item_result_variable_selector"] = active_variable
    set_comparison_focus(["현재 그래프"])

    # 두 저장 슬롯 중 하나만 비어 있으면 새 예측 결과의 기본 저장 위치로 지정한다.
    saved_slots = st.session_state.saved_forecast_datasets
    empty_slots = [
        slot for slot in ("데이터셋1", "데이터셋2") if saved_slots.get(slot) is None
    ]
    if len(empty_slots) == 1 and st.session_state.dataset_save_target != empty_slots[0]:
        st.session_state.dataset_save_target = empty_slots[0]
        st.session_state.dataset_save_widget_version += 1


def enforce_gold_variable_selection() -> None:
    selected = st.session_state.get("gold_variable_selector")
    variables = tuple(st.session_state.get("active_item_variables", ()))
    if variables and selected not in variables:
        st.session_state.gold_variable_selector = variables[0]






@st.cache_data(show_spinner=False)


@st.cache_data(show_spinner=False)














def values_match(left: float | None, right: float | None) -> bool:
    left_missing = left is None or pd.isna(left)
    right_missing = right is None or pd.isna(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return float(left) == float(right)




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
    # 빈 입력칸은 "이 슬롯은 안 바꿈"으로 취급 — 하나만 입력해도 그것만 바뀜.
    name1 = st.session_state.dataset_name_input_1.strip()
    name2 = st.session_state.dataset_name_input_2.strip()
    updated = dict(st.session_state.dataset_display_names)
    if name1:
        updated["데이터셋1"] = name1
    if name2:
        updated["데이터셋2"] = name2
    if not name1 and not name2:
        st.session_state.dataset_manage_notice = (
            "데이터셋 이름을 입력해 주세요."
        )
    elif updated["데이터셋1"] == updated["데이터셋2"]:
        st.session_state.dataset_manage_notice = (
            "서로 다른 데이터셋 이름을 입력해 주세요."
        )
    else:
        st.session_state.dataset_display_names = updated
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






























@st.fragment
def render_gold_data_panel(
    data_column,
    active_label: str,
    applied_quarter: str,
    forecast_quarters: list[str],
    context_key: str,
    active_variable: str,
    gold_forecast_data: pd.DataFrame,
) -> None:
    """데이터표(high/base/low 선택 + final 입력)만 따로 재실행되도록
    fragment로 분리 — 셀 클릭마다 아래 차트(가격 예측 결과)까지 전체 리런되던
    지연을 해결함.
    """
    if st.session_state.pop("gold_final_commit_rerun_pending", False):
        # commit 콜백은 fragment 안에서만 리런을 트리거하므로, fragment 밖의
        # final 그래프까지 갱신하려면 fragment 본문에서 전체 rerun을 걸어야 한다.
        st.rerun(scope="app")
    with data_column.container(key="gold_data_panel"):
        st.caption(
            f"{active_label} · 예측시점 "
            f"{st.session_state.gold_applied_prediction_date} · "
            f"{applied_quarter}부터 {len(forecast_quarters)}개 분기"
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
        committed_values = st.session_state.gold_final_values.get(context_key, {})
        with st.container(key="scenario_grid"):
            header_columns = st.columns(
                [.65, 1, 1, 1, 1, 1, 1, 1], gap=None
            )
            header_columns[0].empty()
            for column, quarter in zip(header_columns[1:], forecast_quarters):
                column.markdown(
                    f'<div class="grid-header">{quarter}</div>',
                    unsafe_allow_html=True,
                )

            for scenario_name in ("high", "base", "low"):
                scenario_values = scenario_table_values[scenario_name]
                row_columns = st.columns(
                    [.65, 1, 1, 1, 1, 1, 1, 1], gap=None
                )
                row_columns[0].button(
                    scenario_name,
                    key=f"gold_{scenario_name}_row",
                    type="secondary",
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
                        st.session_state.gold_prediction_confirmed
                        and selected_for_context.get(quarter) == scenario_name
                    )
                    committed_value = committed_values.get(quarter)
                    is_mismatched = (
                        is_selected
                        and value is not None
                        and committed_value is not None
                        and not pd.isna(committed_value)
                        and not values_match(value, committed_value)
                    )
                    with column:
                        with st.container(
                            key=(
                                f"scenario_{'mismatch' if is_mismatched else 'match'}"
                                f"_{scenario_name}_{index}"
                            )
                        ):
                            st.button(
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
            _, import_button = st.columns([4.4, 1], gap="small")
            with import_button:
                st.button(
                    "가져오기",
                    key="gold_import_button",
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
        final_columns = final_grid.columns(
            [.65, 1, 1, 1, 1, 1, 1, 1], gap=None
        )
        final_columns[0].markdown(
            '<div class="final-row-label{}">final</div>'.format(
                ""
                if st.session_state.gold_prediction_confirmed
                else " final-row-label-disabled"
            ),
            unsafe_allow_html=True,
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
            warning_space, plus_button, minus_button, commit_button = st.columns(
                [4.0, 0.85, 0.85, 1], gap="small"
            )
            with warning_space:
                if st.session_state.gold_table_warning:
                    st.markdown(
                        f'<div class="commit-warning">{st.session_state.gold_table_warning}</div>',
                        unsafe_allow_html=True,
                    )
            with plus_button:
                st.button(
                    "+10%",
                    key="gold_final_increase_button",
                    width="stretch",
                    on_click=adjust_gold_final_draft,
                    args=(context_key, tuple(forecast_quarters), 10),
                )
            with minus_button:
                st.button(
                    "-10%",
                    key="gold_final_decrease_button",
                    width="stretch",
                    on_click=adjust_gold_final_draft,
                    args=(context_key, tuple(forecast_quarters), -10),
                )
            with commit_button:
                st.button(
                    "commit",
                    key="gold_final_commit_button",
                    type="primary",
                    width="stretch",
                    on_click=commit_gold_final_draft,
                    args=(context_key, tuple(forecast_quarters)),
                )

        if st.session_state.gold_final_commit_notice:
            st.toast("final 값이 반영되었습니다.")
            st.session_state.gold_final_commit_notice = False


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
            [1.2, 1],
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
                st.button(
                    "데이터 무작위선택",
                    key="item_price_stage_quick_button",
                    width=128,
                    on_click=prepare_gold_one_click_test,
                    help="테스트용 임의 데이터로 모든 주요 변수의 commit을 완료합니다.",
                )
            if not st.session_state.gold_prediction_confirmed:
                st.caption("조회하기를 눌러 예측시점을 확정해 주세요.")

        if st.session_state.gold_variable_data_scroll:
            scroll_nonce = st.session_state.gold_variable_data_scroll_nonce
            st.html(
                f"""
                <style>
                    .st-key-gold_final_commit_button {{
                        scroll-margin-bottom: 24px;
                    }}
                </style>
                <script>
                    // scroll-nonce: {scroll_nonce}
                    (function () {{
                        const selector = ".st-key-gold_final_commit_button";
                        // Anchor to the commit button itself (screen-size/zoom
                        // independent) instead of a fixed pixel offset. Streamlit
                        // replaces this button's DOM node on this same rerun, so
                        // re-query it fresh every check instead of caching one
                        // reference (a cached one would go stale/detached and
                        // scrollIntoView on it would silently do nothing). Poll
                        // until its position stops moving (waits out the chart
                        // above it loading), then scroll it to the bottom of the
                        // view. Single attempt only — no delayed re-check, so it
                        // never fights a manual scroll afterward.
                        let lastTop = null;
                        let stableCount = 0;
                        let attempts = 0;
                        const check = () => {{
                            const target = document.querySelector(selector);
                            attempts += 1;
                            if (!target) {{
                                if (attempts > 40) {{
                                    return;
                                }}
                                window.setTimeout(check, 80);
                                return;
                            }}
                            const top = target.getBoundingClientRect().top;
                            stableCount = (
                                lastTop !== null && Math.abs(top - lastTop) < 1
                            ) ? stableCount + 1 : 0;
                            lastTop = top;
                            if (stableCount >= 3 || attempts > 40) {{
                                target.scrollIntoView({{
                                    behavior: "smooth",
                                    block: "end"
                                }});
                                return;
                            }}
                            window.setTimeout(check, 80);
                        }};
                        window.setTimeout(check, 80);
                    }})();
                </script>
                """,
                unsafe_allow_javascript=True,
            )
            st.session_state.gold_variable_data_scroll = False

        st.markdown(
            '<div class="section-title">그래프 및 주요 변수 데이터</div>',
            unsafe_allow_html=True,
        )
        content_layout = st.container(key="gold_content_layout")
        graph_column, data_column = content_layout.columns(
            [1.05, 1],
            gap="medium",
        )

        active_variable = (
            selected_variable
            if selected_variable in item_variables
            else item_variables[0]
        )
        active_label = full_labels.get(active_variable, active_variable)
        applied_quarter = st.session_state.gold_applied_quarter
        forecast_quarters = five_quarters_from(
            applied_quarter, item_config.get("forecast_quarters", 7)
        )
        context_key = gold_context_key(active_variable, applied_quarter)
        chart_focus = st.session_state.gold_chart_focus.setdefault(
            context_key, {"final"}
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
                compact_points = history_years >= 2
                historical_point_size = 18 if compact_points else 30
                scenario_point_size = 24 if compact_points else 45
                final_point_size = 30 if compact_points else 55
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
                actual_forecast_rows = [
                    {
                        "분기": historical_quarters[-1],
                        "값": gold_value_for(
                            gold_data, historical_quarters[-1], active_variable
                        ),
                    },
                    *[
                        {
                            "분기": quarter,
                            "값": gold_value_for(gold_data, quarter, active_variable),
                        }
                        for quarter in forecast_quarters
                    ],
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
                actual_forecast_frame = pd.DataFrame(actual_forecast_rows).dropna(
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
                    .mark_line(
                        color="#263238",
                        strokeWidth=2.5,
                        point=alt.OverlayMarkDef(size=historical_point_size),
                    )
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                actual_forecast_line = (
                    alt.Chart(actual_forecast_frame)
                    .mark_line(
                        color="#263238",
                        strokeWidth=2.5,
                        opacity=1,
                        point=alt.OverlayMarkDef(
                            size=historical_point_size, opacity=1
                        ),
                    )
                    .encode(
                        x=x_axis,
                        y=y_axis,
                        tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                    )
                )
                base_opacity = 1 if "base" in chart_focus else 0
                high_opacity = 1 if "high" in chart_focus else 0
                low_opacity = 1 if "low" in chart_focus else 0
                final_opacity = 1 if "final" in chart_focus else 0
                base_line = (
                    alt.Chart(scenario_frames["base"])
                    .mark_line(
                        color="#2563eb",
                        strokeWidth=2.7,
                        point=alt.OverlayMarkDef(
                            size=scenario_point_size, opacity=base_opacity
                        ),
                        opacity=base_opacity,
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
                        point=alt.OverlayMarkDef(
                            size=scenario_point_size, opacity=high_opacity
                        ),
                        opacity=high_opacity,
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
                        point=alt.OverlayMarkDef(
                            size=scenario_point_size, opacity=low_opacity
                        ),
                        opacity=low_opacity,
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
                        point=alt.OverlayMarkDef(
                            size=final_point_size, opacity=final_opacity
                        ),
                        opacity=final_opacity,
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
                        actual_forecast_line,
                        base_line,
                        high_line,
                        low_line,
                        final_line,
                        boundary_rule,
                        boundary_label,
                    )
                    .properties(height=350)
                    .configure_view(stroke=None)
                )
                st.markdown(
                    f'<div class="chart-inline-title">{escape(active_label)} 추이</div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(chart, width="stretch")
                with st.container(
                    key="gold_chart_legend",
                    horizontal=True,
                    vertical_alignment="center",
                    gap="small",
                ):
                    for series_name in ("high", "base", "low", "final"):
                        st.button(
                            series_name,
                            key=f"gold_chart_legend_{series_name}",
                            type="primary" if series_name in chart_focus else "secondary",
                            on_click=set_gold_chart_focus,
                            args=(context_key, series_name),
                        )

            st.segmented_control(
                "기간",
                list(GOLD_HISTORY_PERIOD_LABELS),
                key="gold_history_period",
                format_func=lambda period: GOLD_HISTORY_PERIOD_LABELS[period],
            )

        render_gold_data_panel(
            data_column,
            active_label,
            applied_quarter,
            forecast_quarters,
            context_key,
            active_variable,
            gold_forecast_data,
        )
    item_label = SELECTED_ITEM_LABELS.get(item_code, item_code)
    with st.container(key="item_price_stage_action"):
        warning_column, button_column = st.columns([4.85, 1.25], gap="small")
        with warning_column:
            if st.session_state.item_price_stage_warning:
                st.markdown(
                    f'<div class="commit-warning">'
                    f'{escape(st.session_state.item_price_stage_warning)}</div>',
                    unsafe_allow_html=True,
                )
        with button_column:
            st.button(
                "가격 예측하기",
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
            result_point_scale = 0.6 if history_years >= 2 else 1.0
            result_variable = st.session_state[result_variable_key]
            result_quarters = five_quarters_from(
                applied_quarter, item_config.get("forecast_quarters", 7)
            )
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
            result_actual_forecast_frame = pd.DataFrame(
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
                            "값": gold_value_for(gold_data, quarter, result_variable),
                        }
                        for quarter in result_quarters
                    ],
                ]
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
                axis=alt.Axis(
                    grid=True, gridDash=[2, 3], minExtent=46, maxExtent=46
                ),
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
                .mark_line(
                    color="#263238",
                    strokeWidth=2.5,
                    point=alt.OverlayMarkDef(
                        color="#0b6fc2",
                        filled=True,
                        size=round(42 * result_point_scale),
                    ),
                )
                .encode(
                    x=result_x,
                    y=result_y,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
                alt.Chart(result_actual_forecast_frame)
                .mark_line(
                    color="#263238",
                    strokeWidth=2.5,
                    opacity=1,
                    point=alt.OverlayMarkDef(
                        color="#0b6fc2",
                        filled=True,
                        size=round(42 * result_point_scale),
                        opacity=1,
                    ),
                )
                .encode(
                    x=result_x,
                    y=result_y,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
            ]
            if not dataset2_forecast_frame.empty:
                result_layers.append(
                    alt.Chart(dataset2_forecast_frame)
                    .mark_line(
                        color="#A21CAF",
                        strokeWidth=3 if dataset2_is_focused else 1.6,
                        strokeDash=[6, 4],
                        opacity=0.95 if dataset2_is_focused else 0.25,
                        point=alt.OverlayMarkDef(
                            size=round(
                                (55 if dataset2_is_focused else 35)
                                * result_point_scale
                            ),
                            filled=True,
                            stroke="white",
                            strokeWidth=1,
                            opacity=0.95 if dataset2_is_focused else 0.25,
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
                        strokeDash=[6, 4],
                        opacity=0.95 if dataset1_is_focused else 0.25,
                        point=alt.OverlayMarkDef(
                            size=round(
                                (55 if dataset1_is_focused else 35)
                                * result_point_scale
                            ),
                            filled=True,
                            stroke="white",
                            strokeWidth=1,
                            opacity=0.95 if dataset1_is_focused else 0.25,
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
                    strokeDash=[6, 4],
                    opacity=1 if current_is_focused else 0.3,
                    point=alt.OverlayMarkDef(
                        size=round(
                            (55 if current_is_focused else 35) * result_point_scale
                        ),
                        filled=True,
                        stroke="white",
                        strokeWidth=1,
                        opacity=1 if current_is_focused else 0.3,
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
            if item_code in PRICE_PATHS:
                gold_price_path = PRICE_PATHS[item_code]
                try:
                    if not gold_price_path.exists():
                        raise FileNotFoundError(f"{gold_price_path.name} 파일이 없습니다.")
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
                    price_history_frame = price_full_frame[
                        price_full_frame["분기"].isin(result_history_quarters)
                    ].dropna(subset=["실제가"])
                    price_actual_forecast_frame = price_full_frame[
                        price_full_frame["분기"].isin(
                            [result_history_quarters[-1], *result_quarters]
                        )
                    ].dropna(subset=["실제가"])

                    def price_value_at(quarter: str, column: str) -> float | None:
                        matched = gold_price_data.loc[
                            gold_price_data["분기"] == quarter, column
                        ]
                        if matched.empty or pd.isna(matched.iloc[0]):
                            return None
                        return float(matched.iloc[0])

                    last_actual_quarter = result_history_quarters[-1]
                    gold_linear_prices = (
                        compute_gold_price_final(applied_quarter, result_quarters)
                        if item_code == "gold"
                        else {}
                    )
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
                                    "예측가": gold_linear_prices.get(
                                        quarter, price_value_at(quarter, "예측가")
                                    ),
                                }
                                for quarter in result_quarters
                            ],
                        ]
                    ).dropna(subset=["예측가"])
                    price_forecast_available = not price_forecast_frame.empty

                    def price_dataset_frame(slot: str) -> pd.DataFrame:
                        snapshot = saved_forecast_datasets.get(slot)
                        if not snapshot or snapshot.get("item_code") != item_code:
                            return pd.DataFrame(columns=["분기", "예측가"])
                        price_values = snapshot.get("price_values") or {}
                        return pd.DataFrame(
                            [
                                {
                                    "분기": last_actual_quarter,
                                    "예측가": price_value_at(
                                        last_actual_quarter, "실제가"
                                    ),
                                },
                                *[
                                    {"분기": quarter, "예측가": price_values.get(quarter)}
                                    for quarter in result_quarters
                                ],
                            ]
                        ).dropna(subset=["예측가"])

                    dataset1_price_frame = price_dataset_frame("데이터셋1")
                    dataset2_price_frame = price_dataset_frame("데이터셋2")
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
                            minExtent=46,
                            maxExtent=46,
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
                                size=round(42 * result_point_scale),
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
                    if not price_actual_forecast_frame.empty:
                        price_actual_chart = price_actual_chart + (
                            alt.Chart(price_actual_forecast_frame)
                            .mark_line(
                                color="#263238",
                                strokeWidth=2.6,
                                opacity=1,
                                point=alt.OverlayMarkDef(
                                    color="#0b6fc2",
                                    filled=True,
                                    size=round(42 * result_point_scale),
                                    opacity=1,
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
                    if not dataset2_price_frame.empty:
                        price_actual_chart = price_actual_chart + (
                            alt.Chart(dataset2_price_frame)
                            .mark_line(
                                color="#A21CAF",
                                strokeWidth=3 if dataset2_is_focused else 1.6,
                                strokeDash=[6, 4],
                                opacity=0.95 if dataset2_is_focused else 0.25,
                                point=alt.OverlayMarkDef(
                                    size=round(
                                        (55 if dataset2_is_focused else 35)
                                        * result_point_scale
                                    ),
                                    filled=True,
                                    stroke="white",
                                    strokeWidth=1,
                                    opacity=0.95 if dataset2_is_focused else 0.25,
                                ),
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
                    if not dataset1_price_frame.empty:
                        price_actual_chart = price_actual_chart + (
                            alt.Chart(dataset1_price_frame)
                            .mark_line(
                                color="#2563EB",
                                strokeWidth=3 if dataset1_is_focused else 1.6,
                                strokeDash=[6, 4],
                                opacity=0.95 if dataset1_is_focused else 0.25,
                                point=alt.OverlayMarkDef(
                                    size=round(
                                        (55 if dataset1_is_focused else 35)
                                        * result_point_scale
                                    ),
                                    filled=True,
                                    stroke="white",
                                    strokeWidth=1,
                                    opacity=0.95 if dataset1_is_focused else 0.25,
                                ),
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
                    if price_forecast_available:
                        price_actual_chart = price_actual_chart + (
                            alt.Chart(price_forecast_frame)
                            .mark_line(
                                color="#ef6461",
                                strokeWidth=3 if current_is_focused else 1.6,
                                strokeDash=[6, 4],
                                opacity=1 if current_is_focused else 0.3,
                                point=alt.OverlayMarkDef(
                                    size=round(
                                        (55 if current_is_focused else 35)
                                        * result_point_scale
                                    ),
                                    filled=True,
                                    stroke="white",
                                    strokeWidth=1,
                                    opacity=1 if current_is_focused else 0.3,
                                ),
                            )
                            .encode(
                                x=price_x,
                                y=alt.Y(
                                    "예측가:Q",
                                    title=None,
                                    scale=alt.Scale(zero=False, padding=45),
                                    axis=alt.Axis(minExtent=46, maxExtent=46),
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
                "price_values": (
                    compute_gold_price_final(applied_quarter, result_quarters)
                    if item_code == "gold"
                    else None
                ),
            }

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
                        'style="background:#A21CAF"></span>'
                        f'{escape(dataset_display_name("데이터셋2"))}</span>'
                        if not dataset2_forecast_frame.empty
                        else ""
                    )
                    st.markdown(
                        '<div class="chart-legend">'
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" '
                        'style="background:#263238"></span>실제값</span>'
                        '<span class="chart-legend-item">'
                        '<span class="chart-legend-dot" '
                        'style="background:#ef6461"></span>commit된 final</span>'
                        f'{dataset1_legend_item}'
                        f'{dataset2_legend_item}'
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
                        dataset1_price_legend_item = (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" '
                            'style="background:#2563EB"></span>'
                            f'{escape(dataset_display_name("데이터셋1"))}</span>'
                            if not dataset1_price_frame.empty
                            else ""
                        )
                        dataset2_price_legend_item = (
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" '
                            'style="background:#A21CAF"></span>'
                            f'{escape(dataset_display_name("데이터셋2"))}</span>'
                            if not dataset2_price_frame.empty
                            else ""
                        )
                        st.markdown(
                            '<div class="chart-legend">'
                            '<span class="chart-legend-item">'
                            '<span class="chart-legend-dot" '
                            'style="background:#263238"></span>실제가</span>'
                            f'{forecast_legend_item}'
                            f'{dataset1_price_legend_item}'
                            f'{dataset2_price_legend_item}'
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

            # 두 카드(주요 변수 전망/가격 전망)의 헤더 높이를 맞춰 차트 시작 위치를 정렬한다.
            # 두 헤더가 이미 DOM에 존재한 뒤(두 컬럼 렌더링이 끝난 뒤) + 두 컬럼과
            # 형제 위치(어느 카드 안에도 안 속함)에 둬야 한다 — 카드 안에 넣으면
            # st.html 엘리먼트 자체의 여백 때문에 그 카드에만 여분의 간격이 생기고,
            # 컬럼보다 먼저 두면 헤더가 아직 없어서 스크립트가 아무것도 못 찾는다.
            graph_grid.html(
                """
                <script>
                    (function () {
                        const va = ".st-key-variable_forecast_header";
                        const pa = ".st-key-price_forecast_header";
                        const sync = () => {
                            const vh = document.querySelector(va);
                            const ph = document.querySelector(pa);
                            if (!vh || !ph) {
                                return;
                            }
                            vh.style.minHeight = "0px";
                            ph.style.minHeight = "0px";
                            const maxH = Math.max(
                                vh.scrollHeight, ph.scrollHeight
                            );
                            vh.style.minHeight = maxH + "px";
                            ph.style.minHeight = maxH + "px";
                        };
                        sync();
                        // Reuse one observer across reruns instead of
                        // stacking a new one each time this renders.
                        if (!window.__forecastHeaderSyncObserver) {
                            window.__forecastHeaderSyncObserver = (
                                new ResizeObserver(sync)
                            );
                        }
                        const vh = document.querySelector(va);
                        const ph = document.querySelector(pa);
                        if (vh) {
                            window.__forecastHeaderSyncObserver.observe(vh);
                        }
                        if (ph) {
                            window.__forecastHeaderSyncObserver.observe(ph);
                        }
                    })();
                </script>
                """,
                unsafe_allow_javascript=True,
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
                    saved_slots = st.session_state.saved_forecast_datasets
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
    st.session_state.gold_variable_data_scroll = False


for key, default in {
    "selector_open": False,
    "selected_item": "전기동",
    "selected_item_code": "copper",
    "item_candidate": "전기동",
    "item_warning": None,
    "item_confirmed": False,
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
    "gold_variable_data_scroll": False,
    "gold_variable_data_scroll_nonce": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
if "comparison_dataset_toggle" not in st.session_state:
    st.session_state.comparison_dataset_toggle = ["현재 그래프"]
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
        /* Responsive breakpoints, three tiers only (narrower rules were folded
           into the next-larger one so nothing triggers later than before):
           700px = sm (dense grids force horizontal scroll below this)
           960px = md (multi-column sections start stacking)
           1180px = lg (top-level two-column layout collapses to one) */
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
        .chart-inline-title {
            color:#172033; font-size:.95rem; font-weight:700;
            text-align:center; margin-bottom:.6rem;
        }
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
        .st-key-gold_chart_legend[data-testid="stHorizontalBlock"],
        .st-key-gold_chart_legend > div[data-testid="stHorizontalBlock"] {
            justify-content:center !important;
            flex-wrap:nowrap !important;
            gap:.15rem 1rem !important;
            margin:-.45rem 0 .1rem;
        }
        .st-key-gold_chart_legend div[data-testid="stButton"] > button {
            border:none !important; background:transparent !important;
            box-shadow:none !important; padding:.1rem .5rem !important;
            min-height:0 !important; height:auto !important;
            font-size:.78rem !important; font-weight:400 !important;
            color:#697386 !important; opacity:.5 !important;
            display:inline-flex !important; align-items:center !important;
            gap:.32rem !important; border-radius:999px !important;
            transition:opacity .15s ease;
        }
        .st-key-gold_chart_legend div[data-testid="stButton"]
        > button[data-testid="stBaseButton-primary"] {
            color:#1f2937 !important; font-weight:600 !important;
            opacity:1 !important; background:rgba(15,23,42,.07) !important;
        }
        .st-key-gold_chart_legend div[data-testid="stButton"] > button::before {
            content:""; width:10px; height:10px; border-radius:50%;
            display:inline-block; flex-shrink:0;
        }
        .st-key-gold_chart_legend_base div[data-testid="stButton"] > button::before {
            background:#2563eb;
        }
        .st-key-gold_chart_legend_high div[data-testid="stButton"] > button::before {
            background:#c026d3;
        }
        .st-key-gold_chart_legend_low div[data-testid="stButton"] > button::before {
            background:#eab308;
        }
        .st-key-gold_chart_legend_final div[data-testid="stButton"] > button::before {
            background:#ef6461;
        }
        .forecast-card-title {
            min-height:40px; display:flex; align-items:center;
            color:#172033; font-size:1.05rem; font-weight:400;
            white-space:nowrap; margin:0;
        }
        .st-key-variable_forecast_header,
        .st-key-price_forecast_header {
            /* height is synced dynamically via JS below, regardless of how
               many rows the variable pills wrap into. */
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
        .st-key-gold_import_button button[data-testid="stBaseButton-secondary"] {
            background:white !important; border:1.5px solid #ef6461 !important;
            color:#ef6461 !important;
        }
        .st-key-gold_import_button button[data-testid="stBaseButton-secondary"]:hover {
            background:#fff3f2 !important; border-color:#dc5653 !important;
            color:#dc5653 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:white; border-color:#dfe4eb; border-radius:11px;
            box-shadow:0 5px 18px rgba(23,32,51,.06);
        }
        .grid-header, .grid-label, .final-cell {
            min-height:37px; border-radius:7px; display:flex; align-items:center;
            justify-content:center; padding:.25rem .1rem; margin-inline:1px;
            font-size:.72rem;
        }
        .st-key-scenario_grid [data-testid="stHorizontalBlock"] {
            flex-wrap:nowrap !important;
        }
        .st-key-scenario_grid [data-testid="stColumn"] {
            min-width:0 !important;
        }
        .st-key-scenario_grid div[data-testid="stButton"] > button {
            width:calc(100% - 2px) !important; margin-inline:1px;
            padding-left:.15rem !important; padding-right:.15rem !important;
            font-size:.72rem !important;
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
        .st-key-gold_high_row button[data-testid="stBaseButton-secondary"],
        .st-key-gold_base_row button[data-testid="stBaseButton-secondary"],
        .st-key-gold_low_row button[data-testid="stBaseButton-secondary"] {
            background:white !important; border:1.5px solid #ef6461 !important;
            color:#ef6461 !important;
        }
        .st-key-gold_high_row button[data-testid="stBaseButton-secondary"]:hover,
        .st-key-gold_base_row button[data-testid="stBaseButton-secondary"]:hover,
        .st-key-gold_low_row button[data-testid="stBaseButton-secondary"]:hover {
            background:#fff3f2 !important; border-color:#dc5653 !important;
            color:#dc5653 !important;
        }
        .final-wrap {
            margin-top:1.1rem; padding-top:.8rem; border-top:2px solid #d9e1ec;
        }
        .final-label { background:#6f9692; color:white; font-weight:800; }
        .final-value { background:#6f9692; color:white; font-weight:750; }
        .st-key-final_grid div[data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:.65fr repeat(7, minmax(0, 1fr));
            gap:3px !important;
        }
        .st-key-final_grid div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {
            width:auto !important; min-width:0 !important;
            flex:none !important;
        }
        .st-key-final_grid div[data-testid="stButton"] > button {
            font-size:.72rem !important;
        }
        .final-row-label {
            min-height:39px; border-radius:8px; margin-inline:1px;
            display:flex; align-items:center; justify-content:center;
            font-size:.72rem; font-weight:700; white-space:nowrap;
            background:#ff4b4b; color:white;
        }
        .final-row-label-disabled {
            background:transparent; color:rgba(49, 51, 63, .4);
        }
        [class*="st-key-final_clean_"] input {
            width:100%; margin:0;
            height:46px; background:#6f9692 !important; color:white !important;
            border-color:#6f9692 !important; text-align:center; font-weight:750;
            font-size:.78rem;
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
            font-size:.78rem;
        }
        [class*="st-key-scenario_mismatch_"] div[data-testid="stButton"] > button {
            background:#b7793e !important; border-color:#b7793e !important; color:white !important;
        }
        [class*="st-key-final_empty_"] input {
            width:100%; margin:0;
            height:46px; background:white !important; text-align:center; font-weight:700;
            font-size:.78rem;
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
        .st-key-gold_final_increase_button button,
        .st-key-gold_final_decrease_button button {
            font-size:.65rem;
        }
        .st-key-gold_variable_navigation[data-testid="stHorizontalBlock"],
        .st-key-gold_variable_navigation > div[data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:minmax(210px, 360px) 90px !important;
            justify-content:start !important;
            gap:16px !important;
        }
        .st-key-gold_variable_navigation[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-gold_variable_navigation > div[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"] {
            width:100% !important; min-width:0 !important;
        }
        .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"],
        .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"] {
            display:grid !important;
            grid-template-columns:170px 110px 1fr !important;
            justify-content:start !important;
            align-items:center !important;
            gap:12px !important;
            flex-wrap:nowrap !important;
        }
        .st-key-item_price_stage_quick_button {
            display:flex !important; justify-content:flex-end !important;
        }
        .st-key-item_price_stage_quick_button div[data-testid="stButton"] button {
            font-size:.8rem !important; padding-inline:.3rem !important;
            white-space:nowrap !important;
        }
        .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"],
        .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"]
        > div[data-testid="stElementContainer"] {
            width:100% !important;
            min-width:0 !important;
        }
        .st-key-forecast_comparison_layout [data-testid="stHorizontalBlock"] {
            /* Streamlit's own per-column min-width default only makes sense
               once a row has actually wrapped; without nowrap here, this
               row's columns fight over space and the sidebar shrinks well
               before the 1180px breakpoint forces a real wrap. */
            flex-wrap:nowrap !important;
        }
        .st-key-forecast_comparison_layout [data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {
            min-width:0 !important;
        }
        .st-key-forecast_comparison_layout [data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:has(.st-key-dataset_comparison_panel) {
            /* Without a floor, this sidebar's fixed 1:4.1 share keeps
               shrinking right up to the 1180px wrap point, so its own text
               (dataset name + status) starts overlapping well before the
               row actually wraps. */
            min-width:260px !important;
        }
        .st-key-forecast_graph_grid [data-testid="stHorizontalBlock"] {
            align-items:stretch !important;
        }
        .st-key-forecast_graph_grid [data-testid="stColumn"] > div,
        .st-key-forecast_graph_grid [data-testid="stColumn"] > div > div,
        .st-key-forecast_graph_grid [data-testid="stColumn"] > div > div > div {
            height:100% !important;
        }
        .st-key-variable_forecast_card,
        .st-key-price_forecast_card {
            height:100% !important;
        }
        @media (max-width: 1180px) {
            .st-key-item_selection_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-item_summary_layout),
            .st-key-gold_content_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_data_panel) {
                flex-wrap:wrap !important;
                align-items:flex-start !important;
            }
            .st-key-item_selection_layout
            [data-testid="stHorizontalBlock"]:has(.st-key-item_summary_layout)
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
            .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"],
            .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"] {
                grid-template-columns:160px 100px 1fr !important;
                gap:10px !important;
            }
        }
        @media (max-width: 960px) {
            .st-key-gold_workspace_header
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_variable_navigation) {
                flex-wrap:wrap !important;
                align-items:flex-start !important;
            }
            .st-key-gold_workspace_header
            [data-testid="stHorizontalBlock"]:has(.st-key-gold_variable_navigation)
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
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
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]) {
                flex-wrap:wrap !important;
            }
            .st-key-item_summary_layout
            [data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"])
            > div[data-testid="stColumn"] {
                flex:1 1 100% !important;
                width:100% !important;
                min-width:0 !important;
            }
            .st-key-gold_variable_selector [data-testid="stButtonGroup"] {
                flex-wrap:nowrap !important;
                overflow-x:auto !important;
                scrollbar-width:thin;
            }
            .st-key-gold_data_panel {
                overflow-x:auto !important;
                padding-bottom:.25rem;
            }
            .st-key-scenario_grid,
            .st-key-commit_row,
            .st-key-final_grid,
            .st-key-final_commit_row {
                /* Forced wide + horizontal scroll rather than stacking into
                   cards: this is a quarter-by-quarter comparison table, and
                   losing side-by-side columns would defeat its purpose. */
                min-width:620px !important;
            }
            .st-key-gold_prediction_navigation[data-testid="stHorizontalBlock"],
            .st-key-gold_prediction_navigation > div[data-testid="stHorizontalBlock"] {
                grid-template-columns:minmax(120px, 1fr) 96px 108px !important;
            }
        }
        /* 그래프 강조 title spans the full row, with all three pills (현재
           그래프 / 데이터셋1 / 데이터셋2) on one line below it in equal
           thirds, regardless of viewport width. The two st.pills() calls
           each carry their own label + button-toolbar, so their wrapper
           divs are unwrapped (display:contents) to let the label and both
           toolbars become independent grid items placed via grid-area. */
        .st-key-item_dataset_focus_section {
            display:grid !important;
            grid-template-columns:1fr 1fr 1fr;
            grid-template-areas:
                "title title title"
                "cur ds ds";
            align-items:center;
            gap:.35rem .5rem;
        }
        .st-key-comparison_current_focus,
        .st-key-comparison_saved_focus,
        .st-key-item_dataset_focus_section [data-testid="stButtonGroup"] {
            display:contents !important;
        }
        .st-key-comparison_current_focus [data-testid="stWidgetLabel"] {
            grid-area:title;
        }
        .st-key-comparison_saved_focus [data-testid="stWidgetLabel"] {
            display:none !important;
        }
        .st-key-comparison_current_focus [role="toolbar"] {
            grid-area:cur;
        }
        .st-key-comparison_saved_focus [role="toolbar"] {
            grid-area:ds;
        }
        .st-key-item_dataset_focus_section [role="toolbar"] {
            display:flex !important;
            flex-wrap:nowrap !important;
            gap:.4rem;
        }
        .st-key-item_dataset_focus_section [role="toolbar"] button {
            min-width:0 !important;
            flex:1 1 0 !important;
            padding-left:.3rem !important;
            padding-right:.3rem !important;
        }
        .st-key-item_dataset_focus_section [role="toolbar"] button p {
            font-size:.72rem !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
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
