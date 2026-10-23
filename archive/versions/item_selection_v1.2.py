# Version 1.2
from pathlib import Path

import pandas as pd
import streamlit as st

from _compat import legacy_default_path


st.set_page_config(
    page_title="원자재 가격 대시보드",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ITEMS = ["비철금속", "전기동"]
VARIABLES = ["수요", "공급", "원재료", "투자", "거시경제"]
SCENARIO_FILES = {
    "base": legacy_default_path("base"),
    "high": legacy_default_path("high"),
    "low": legacy_default_path("low"),
}
PREDICTION_DATES = ["2026-07-21", "2025-03-21", "2026-01-21"]


@st.cache_data
def load_scenarios() -> dict[str, pd.DataFrame]:
    scenarios = {}
    for name, path in SCENARIO_FILES.items():
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


for key, default in {
    "selector_open": False,
    "selected_item": "전기동",
    "item_confirmed": False,
    "selected_variable": "수요",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

scenarios = load_scenarios()
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
        .final-wrap {
            margin-top:1.1rem; padding-top:.8rem; border-top:2px solid #d9e1ec;
        }
        .final-label { background:#6f9692; color:white; font-weight:800; }
        .final-value { background:#6f9692; color:white; font-weight:750; }
        .commit-warning {
            height:46px; box-sizing:border-box; display:flex; align-items:center;
            padding:0 .95rem; border:1px solid #eee8a9; border-radius:8px;
            background:#fffde7; color:#8a6900; font-size:.9rem;
        }
        .st-key-commit_row div[data-testid="stButton"] > button {
            height:46px; min-height:46px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">COMMODITY DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">원자재 가격 대시보드</h1>', unsafe_allow_html=True)
st.markdown('<div class="description">분석할 품목을 선택해 주세요.</div>', unsafe_allow_html=True)

button_column, selected_column, panel_column = st.columns([0.8, 1.05, 2.8], gap="medium")
with button_column:
    button_text = "선택창 닫기" if st.session_state.selector_open else "품목 선택"
    if st.button(button_text, type="primary", width="stretch"):
        st.session_state.selector_open = not st.session_state.selector_open
        st.rerun()

with selected_column:
    st.markdown(
        f'<div class="selected-card"><div class="selected-label">현재 선택 품목</div>'
        f'<div class="selected-value">{st.session_state.selected_item}</div></div>',
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
                    st.rerun()

            item = st.selectbox(
                "선택 품목",
                ITEMS,
                index=ITEMS.index(st.session_state.selected_item),
            )
            apply_column, cancel_column = st.columns([2.2, 1])
            with apply_column:
                if st.button("적용", type="primary", width="stretch"):
                    st.session_state.selected_item = item
                    st.session_state.item_confirmed = True
                    st.session_state.selector_open = False
                    st.rerun()
            with cancel_column:
                if st.button("취소", width="stretch"):
                    st.session_state.selector_open = False
                    st.rerun()
    else:
        st.info("왼쪽의 **품목 선택** 버튼을 누르면 선택창이 열립니다.")

if st.session_state.item_confirmed:
    st.divider()

    with st.container(border=True):
        controls, prediction = st.columns([2.1, 1], gap="large")
        with controls:
            st.markdown('<div class="workspace-title">주요 변수 설정</div>', unsafe_allow_html=True)
            selected_variable = st.segmented_control(
                "주요 변수",
                VARIABLES,
                default=st.session_state.selected_variable,
                label_visibility="collapsed",
            )
            if selected_variable:
                st.session_state.selected_variable = selected_variable

        with prediction:
            st.markdown('<div class="workspace-title">예측시점</div>', unsafe_allow_html=True)
            date_column, search_column = st.columns([2.2, 1])
            with date_column:
                prediction_date = st.selectbox(
                    "예측시점",
                    PREDICTION_DATES,
                    index=PREDICTION_DATES.index(st.session_state.applied_prediction_date),
                    label_visibility="collapsed",
                )
            with search_column:
                if st.button("조회하기", type="primary", width="stretch"):
                    st.session_state.applied_prediction_date = prediction_date
                    st.session_state.applied_quarter = date_to_quarter(prediction_date)
                    st.rerun()

        st.markdown("##### 그래프 및 시나리오 데이터")
        graph_column, data_column = st.columns([1.15, 1], gap="large")
        with graph_column:
            st.markdown(
                f"""
                <div class="chart-placeholder">
                    <strong>{st.session_state.selected_variable} 그래프 영역</strong>
                    <span>그래프 로직은 다음 단계에서 연결합니다.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with data_column:
            st.caption(
                f"{st.session_state.selected_variable} · "
                f"예측시점 {st.session_state.applied_prediction_date} · "
                f"{st.session_state.applied_quarter}부터 4개 분기"
            )
            scenario_table = build_scenario_table(
                scenarios,
                st.session_state.selected_variable,
                st.session_state.applied_quarter,
            )
            context_key = (
                f"{st.session_state.selected_item}|"
                f"{st.session_state.selected_variable}|"
                f"{st.session_state.applied_quarter}"
            )
            selected_for_context = st.session_state.selected_cells.setdefault(context_key, {})

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
                        if column.button(
                            label,
                            key=f"cell_{context_key}_{quarter}_{scenario_name}",
                            type="primary" if is_selected else "secondary",
                            width="stretch",
                            help=f"{quarter}의 {scenario_name} 값 선택",
                        ):
                            selected_for_context[quarter] = scenario_name
                            st.session_state.selected_cells[context_key] = selected_for_context
                            st.rerun()

            with st.container(key="commit_row"):
                commit_space, commit_button = st.columns([3, 1])
                show_selection_warning = False
                with commit_button:
                    if st.button("commit", type="primary", width="stretch"):
                        missing_quarters = [
                            quarter
                            for quarter in scenario_table.columns
                            if quarter not in selected_for_context
                        ]
                        if missing_quarters:
                            show_selection_warning = True
                        else:
                            st.session_state.final_values[context_key] = {
                                quarter: scenario_table.loc[selected_for_context[quarter], quarter]
                                for quarter in scenario_table.columns
                            }
                            st.toast("선택한 값이 final에 반영되었습니다.", icon="✅")
                            st.rerun()

                with commit_space:
                    if show_selection_warning:
                        st.markdown(
                            '<div class="commit-warning">각 분기에서 값을 하나씩 선택해 주세요.</div>',
                            unsafe_allow_html=True,
                        )

            committed_values = st.session_state.final_values.get(context_key, {})
            st.markdown('<div class="final-wrap"></div>', unsafe_allow_html=True)
            final_columns = st.columns([.72, 1, 1, 1, 1], gap="small")
            final_columns[0].markdown(
                '<div class="final-cell final-label">final</div>',
                unsafe_allow_html=True,
            )
            for column, quarter in zip(final_columns[1:], scenario_table.columns):
                final_value = committed_values.get(quarter)
                final_label = "—" if final_value is None or pd.isna(final_value) else f"{final_value:,.0f}"
                column.markdown(
                    f'<div class="final-cell final-value">{final_label}</div>',
                    unsafe_allow_html=True,
                )
else:
    st.divider()
    st.caption("품목을 선택하고 적용하면 주요 변수 설정 화면이 표시됩니다.")
