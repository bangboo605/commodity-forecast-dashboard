# Version 1.7
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from _compat import legacy_scenario_path


st.set_page_config(
    page_title="원자재 가격 대시보드",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ITEMS = ["비철금속", "전기동"]
VARIABLES = ["수요", "공급", "원재료", "투자", "거시경제"]
DATASET_NUMBERS = (1, 2, 3)
PREDICTION_DATES = ["2026-07-21", "2025-03-21", "2026-01-21"]


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


def quarter_key(label: str) -> tuple[int, int]:
    normalized = str(label).replace("/", "-").upper()
    year_text, quarter_text = normalized.split("-")
    return int(year_text), int(quarter_text.replace("Q", ""))


def quarter_label(label: str) -> str:
    year, quarter = quarter_key(label)
    return f"{year}-Q{quarter}"


def current_quarter() -> str:
    today = pd.Timestamp.today()
    return f"{today.year}-Q{((today.month - 1) // 3) + 1}"


def date_to_quarter(date_text: str) -> str:
    date = pd.Timestamp(date_text)
    return f"{date.year}-Q{((date.month - 1) // 3) + 1}"


def four_quarters_from(start_quarter: str) -> list[str]:
    year, quarter = quarter_key(start_quarter)
    quarters = []
    for offset in range(4):
        zero_based = (quarter - 1) + offset
        quarters.append(f"{year + zero_based // 4}-Q{zero_based % 4 + 1}")
    return quarters


def shift_quarter(quarter_label_text: str, amount: int) -> str:
    year, quarter = quarter_key(quarter_label_text)
    zero_based = (quarter - 1) + amount
    return f"{year + zero_based // 4}-Q{zero_based % 4 + 1}"


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
    final_values: tuple[float | None, ...],
) -> alt.Chart:
    axis_quarters = list(historical_quarters + forecast_quarters)
    historical_data = pd.DataFrame(
        {"분기": historical_quarters, "값": historical_values}
    ).dropna(subset=["값"])
    final_data = pd.DataFrame(
        {"분기": forecast_quarters, "값": final_values}
    ).dropna(subset=["값"])

    x_encoding = alt.X(
        "분기:N",
        title=None,
        sort=axis_quarters,
        scale=alt.Scale(domain=axis_quarters),
        axis=alt.Axis(
            values=axis_quarters,
            labelAngle=-30,
            labelOverlap=False,
            grid=False,
        ),
    )
    y_encoding = alt.Y(
        "값:Q",
        title=None,
        scale=alt.Scale(zero=False, padding=25),
        axis=alt.Axis(
            format=",.0f",
            grid=True,
            gridDash=[2, 3],
        ),
    )
    historical_line = (
        alt.Chart(historical_data)
        .mark_line(color="#263238", strokeWidth=2.5, point=alt.OverlayMarkDef(size=55))
        .encode(
            x=x_encoding,
            y=y_encoding,
            tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
        )
    )
    chart_layers = [historical_line]
    if not final_data.empty:
        connector_data = final_data.copy()
        if not historical_data.empty:
            connector_data = pd.concat(
                [historical_data.tail(1), connector_data],
                ignore_index=True,
            )
        chart_layers.extend(
            [
                alt.Chart(connector_data)
                .mark_line(color="#ef6461", strokeWidth=2.5, strokeDash=[7, 5])
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
                    size=75,
                    strokeWidth=1.5,
                )
                .encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=["분기:N", alt.Tooltip("값:Q", format=",")],
                ),
            ]
        )

    chart_layers.append(
        alt.Chart(pd.DataFrame({"분기": [forecast_quarters[0]]}))
        .mark_rule(color="#ef6461", strokeDash=[3, 3], opacity=.7)
        .encode(x=x_encoding)
    )
    return (
        alt.layer(*chart_layers)
        .properties(
            height=330,
            title=alt.TitleParams(
                text=f"{variable} 추이",
                anchor="middle",
                frame="bounds",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_title(anchor="middle", fontSize=17, color="#172033")
    )


def select_scenario_cell(context_key: str, quarter: str, scenario_name: str) -> None:
    if st.session_state.selected_variable is None:
        st.session_state.table_warning = "먼저 주요 변수를 선택해 주세요."
        return

    selected = st.session_state.selected_cells.setdefault(context_key, {})
    selected[quarter] = scenario_name
    st.session_state.table_warning = None


def final_input_key(context_key: str, quarter: str) -> str:
    return f"final_input|{context_key}|{quarter}"


def warn_variable_required() -> None:
    st.session_state.table_warning = "먼저 주요 변수를 선택해 주세요."


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


def import_selected_values(
    context_key: str,
    quarters: list[str],
    scenario_table: pd.DataFrame,
) -> None:
    if st.session_state.selected_variable is None:
        warn_variable_required()
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


def update_final_draft(context_key: str, quarter: str) -> None:
    input_key = final_input_key(context_key, quarter)
    raw_value = str(st.session_state[input_key]).strip()
    normalized_value = raw_value.replace(",", "")
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


def commit_final_draft(context_key: str, quarters: list[str]) -> None:
    if st.session_state.selected_variable is None:
        warn_variable_required()
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
    st.session_state.table_warning = None
    st.session_state.final_commit_notice = True


def build_scenario_table(
    scenarios: dict[str, pd.DataFrame],
    variable: str,
    start_quarter: str,
) -> pd.DataFrame:
    selected_labels = four_quarters_from(start_quarter)

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


def advance_dataset() -> None:
    variable = st.session_state.selected_variable
    if variable is None:
        return

    current_number = dataset_number_for(variable)
    current_index = DATASET_NUMBERS.index(current_number)
    next_number = DATASET_NUMBERS[(current_index + 1) % len(DATASET_NUMBERS)]
    st.session_state.variable_dataset_choices[variable] = next_number
    st.session_state.dataset_selector = f"{variable}{next_number}"
    st.session_state.table_warning = None


def mark_prediction_pending() -> None:
    st.session_state.prediction_confirmed = False
    st.session_state.selected_variable = None
    st.session_state.variable_selector = None
    st.session_state.table_warning = None


def apply_item_selection() -> None:
    candidate = st.session_state.item_candidate
    if candidate != "전기동":
        st.session_state.item_warning = "현재 비철금속 데이터는 준비 중입니다."
        return

    st.session_state.selected_item = candidate
    st.session_state.item_confirmed = True
    st.session_state.selected_variable = None
    st.session_state.variable_selector = None
    st.session_state.prediction_confirmed = False
    st.session_state.table_warning = None
    st.session_state.item_warning = None
    st.session_state.selector_open = False


for key, default in {
    "selector_open": False,
    "selected_item": "전기동",
    "item_candidate": "전기동",
    "item_warning": None,
    "item_confirmed": False,
    "selected_variable": None,
    "variable_selector": None,
    "prediction_candidate": PREDICTION_DATES[0],
    "prediction_confirmed": False,
    "table_warning": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

scenario_sets = {
    dataset_number: load_scenarios(dataset_number)
    for dataset_number in DATASET_NUMBERS
}
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
if "imported_selections" not in st.session_state:
    st.session_state.imported_selections = {}
if "imported_values" not in st.session_state:
    st.session_state.imported_values = {}
if "final_commit_notice" not in st.session_state:
    st.session_state.final_commit_notice = False
if "variable_dataset_choices" not in st.session_state:
    st.session_state.variable_dataset_choices = {
        variable: 1 for variable in VARIABLES
    }

st.markdown(
    """
    <style>
        .stApp { background: #f7f8fa; }
        .block-container { max-width: 1240px; padding-top: 2.4rem; }
        .eyebrow { color:#697386; font-size:.76rem; font-weight:700; letter-spacing:.08em; }
        .main-title { margin:.15rem 0 .2rem; font-size:1.7rem; color:#172033; }
        .description { color:#697386; margin-bottom:1.2rem; }
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
        }
        .workspace-title { color:#172033; font-size:1.22rem; font-weight:750; margin-bottom:.7rem; }
        .chart-placeholder {
            height:330px; border:1px dashed #bdc6d4; border-radius:10px; background:white;
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            color:#8a94a6; text-align:center;
        }
        .chart-placeholder strong { color:#536072; margin-bottom:.35rem; }
        div[data-testid="stButton"] > button { min-height:39px; border-radius:8px; font-weight:700; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:white; border-color:#dfe4eb; border-radius:11px;
            box-shadow:0 5px 18px rgba(23,32,51,.06);
        }
        .grid-header, .grid-label, .final-cell {
            min-height:39px; border-radius:7px; display:flex; align-items:center;
            justify-content:center; padding:.35rem; font-size:.82rem;
        }
        .grid-header { color:#6f7a8c; background:#eef1f5; font-weight:700; }
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
        .final-wrap {
            margin-top:1.1rem; padding-top:.8rem; border-top:2px solid #d9e1ec;
        }
        .final-label { background:#6f9692; color:white; font-weight:800; }
        .final-value { background:#6f9692; color:white; font-weight:750; }
        [class*="st-key-final_clean_"] input {
            height:46px; background:#6f9692 !important; color:white !important;
            border-color:#6f9692 !important; text-align:center; font-weight:750;
        }
        [class*="st-key-final_clean_"] input:focus {
            background:#b7793e !important; border-color:#b7793e !important;
            box-shadow:0 0 0 1px #b7793e !important;
        }
        [class*="st-key-final_dirty_"] input {
            height:46px; background:#b7793e !important; color:white !important;
            border-color:#b7793e !important; text-align:center; font-weight:750;
        }
        [class*="st-key-final_empty_"] input {
            height:46px; background:white !important; text-align:center; font-weight:700;
        }
        [class*="st-key-final_locked_"] div[data-testid="stButton"] > button {
            height:46px; min-height:46px;
        }
        .commit-warning {
            height:46px; box-sizing:border-box; display:flex; align-items:center;
            justify-content:center; text-align:center; line-height:1.25;
            padding:0 .95rem; border:1px solid #eee8a9; border-radius:8px;
            background:#fffde7; color:#8a6900; font-size:.9rem;
        }
        .st-key-commit_row div[data-testid="stButton"] > button {
            height:46px; min-height:46px;
        }
        .st-key-final_commit_row div[data-testid="stButton"] > button {
            height:46px; min-height:46px;
        }
        @media (max-width: 900px) {
            .st-key-variable_navigation div[data-testid="stHorizontalBlock"] {
                flex-wrap:wrap;
            }
            .st-key-variable_navigation div[data-testid="stColumn"]:nth-child(1) {
                flex:1 1 calc(70% - .5rem);
                min-width:180px;
            }
            .st-key-variable_navigation div[data-testid="stColumn"]:nth-child(2) {
                flex:0 0 105px;
                min-width:105px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">COMMODITY DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">원자재 가격 대시보드</h1>', unsafe_allow_html=True)
st.markdown('<div class="description">분석할 품목을 선택해 주세요.</div>', unsafe_allow_html=True)

summary_column, panel_column = st.columns([1.85, 2.8], gap="medium")
with summary_column:
    button_column, selected_column = st.columns([0.8, 1.05], gap="medium")
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
                st.markdown("#### 품목 선택")
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

if st.session_state.item_confirmed:
    st.divider()

    with st.container(border=True):
        controls, prediction = st.columns([1.15, 1], gap="large")
        with controls:
            st.markdown('<div class="workspace-title">주요 변수 설정</div>', unsafe_allow_html=True)
            selected_variable = st.segmented_control(
                "주요 변수",
                VARIABLES,
                key="variable_selector",
                label_visibility="collapsed",
                disabled=not st.session_state.prediction_confirmed,
            )
            if selected_variable != st.session_state.selected_variable:
                st.session_state.selected_variable = selected_variable
                st.session_state.table_warning = None

            with st.container(key="variable_navigation"):
                detail_column, next_column = st.columns([1.55, .75])
                with detail_column:
                    if st.session_state.selected_variable is None:
                        detail_options = ["주요 변수 선택"]
                        expected_detail = detail_options[0]
                    else:
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

                    selected_detail = st.selectbox(
                        "데이터 세트",
                        detail_options,
                        label_visibility="collapsed",
                        key="dataset_selector",
                        disabled=(
                            not st.session_state.prediction_confirmed
                            or st.session_state.selected_variable is None
                        ),
                    )
                    if st.session_state.selected_variable is not None:
                        selected_dataset_number = int(selected_detail[-1])
                        st.session_state.variable_dataset_choices[
                            st.session_state.selected_variable
                        ] = selected_dataset_number

                with next_column:
                    st.button(
                        "다음",
                        width="stretch",
                        on_click=advance_dataset,
                        disabled=(
                            not st.session_state.prediction_confirmed
                            or st.session_state.selected_variable is None
                        ),
                    )

        with prediction:
            st.markdown('<div class="workspace-title">예측시점</div>', unsafe_allow_html=True)
            with st.container(
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                prediction_date = st.selectbox(
                    "예측시점",
                    PREDICTION_DATES,
                    key="prediction_candidate",
                    label_visibility="collapsed",
                    width=190,
                    on_change=mark_prediction_pending,
                )
                if st.button("조회하기", type="primary", width=120):
                    st.session_state.applied_prediction_date = prediction_date
                    st.session_state.applied_quarter = date_to_quarter(prediction_date)
                    st.session_state.prediction_confirmed = True
                    st.session_state.table_warning = None
                    st.rerun()
            if not st.session_state.prediction_confirmed:
                st.caption("조회하기를 눌러 예측시점을 먼저 확정해 주세요.")

        st.markdown("##### 그래프 및 시나리오 데이터")
        active_dataset_number = (
            dataset_number_for(st.session_state.selected_variable)
            if st.session_state.selected_variable is not None
            else 1
        )
        scenarios = scenario_sets[active_dataset_number]
        context_key = context_key_for(
            st.session_state.selected_variable or "미선택"
        )
        graph_column, data_column = st.columns([1.15, 1], gap="large")
        with graph_column:
            forecast_quarters = four_quarters_from(st.session_state.applied_quarter)
            historical_start = shift_quarter(st.session_state.applied_quarter, -4)
            historical_quarters = four_quarters_from(historical_start)
            if st.session_state.selected_variable is None:
                st.markdown(
                    """
                    <div class="chart-placeholder">
                        <strong>그래프 영역</strong>
                        <span>주요 변수를 선택하면 그래프가 표시됩니다.</span>
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
                committed_values = st.session_state.final_values.get(context_key, {})
                chart = create_trend_chart(
                    st.session_state.selected_variable,
                    tuple(historical_quarters),
                    tuple(historical_values),
                    tuple(forecast_quarters),
                    tuple(committed_values.get(quarter) for quarter in forecast_quarters),
                )
                st.altair_chart(chart, width="stretch")

        with data_column:
            st.caption(
                f"{st.session_state.selected_variable or '주요 변수 미선택'} · "
                f"예측시점 {st.session_state.applied_prediction_date} · "
                f"{st.session_state.applied_quarter}부터 4개 분기"
            )
            scenario_table = build_scenario_table(
                scenarios,
                st.session_state.selected_variable,
                st.session_state.applied_quarter,
            )
            committed_values = st.session_state.final_values.get(context_key, {})
            selected_for_context = st.session_state.selected_cells.setdefault(context_key, {})
            draft_for_context = st.session_state.final_drafts.get(context_key, {})

            with st.container(key="scenario_grid"):
                header_columns = st.columns([.72, 1, 1, 1, 1], gap="small")
                header_columns[0].empty()
                for column, quarter in zip(header_columns[1:], scenario_table.columns):
                    column.markdown(
                        f'<div class="grid-header">{quarter}</div>',
                        unsafe_allow_html=True,
                    )

                for scenario_name in scenario_table.index:
                    row_columns = st.columns([.72, 1, 1, 1, 1], gap="small")
                    row_columns[0].markdown(
                        f'<div class="grid-label">{scenario_name}</div>',
                        unsafe_allow_html=True,
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
                commit_space, commit_button = st.columns([3, 1])
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
                        and (
                            st.session_state.selected_variable is None
                            or not st.session_state.final_drafts.get(context_key)
                        )
                    ):
                        st.markdown(
                            f'<div class="commit-warning">{st.session_state.table_warning}</div>',
                            unsafe_allow_html=True,
                        )

            committed_values = st.session_state.final_values.get(context_key, {})
            draft_values = (
                st.session_state.final_drafts.setdefault(context_key, {})
                if st.session_state.selected_variable is not None
                else {}
            )
            st.markdown('<div class="final-wrap"></div>', unsafe_allow_html=True)
            final_columns = st.columns([.72, 1, 1, 1, 1], gap="small")
            final_columns[0].markdown(
                '<div class="final-cell final-label">final</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.selected_variable is None:
                for index, column in enumerate(final_columns[1:]):
                    column.button(
                        "—",
                        key=f"final_locked_{index}",
                        type="secondary",
                        width="stretch",
                        on_click=warn_variable_required,
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
                warning_space, final_commit_button = st.columns([3, 1])
                with warning_space:
                    if (
                        st.session_state.table_warning
                        and st.session_state.selected_variable is not None
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
else:
    st.divider()
    st.caption("품목을 선택하고 적용하면 주요 변수 설정 화면이 표시됩니다.")
