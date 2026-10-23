# Version 1.1
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


def build_scenario_table(
    scenarios: dict[str, pd.DataFrame],
    variable: str,
    start_quarter: str,
) -> pd.DataFrame:
    all_dates = scenarios["base"].get("Date", pd.Series(dtype=str)).astype(str).tolist()
    ordered_dates = sorted(all_dates, key=quarter_key)
    start_index = next(
        (index for index, value in enumerate(ordered_dates) if quarter_label(value) == start_quarter),
        0,
    )
    selected_dates = ordered_dates[start_index : start_index + 4]
    selected_labels = [quarter_label(value) for value in selected_dates]

    rows = []
    for scenario_name in ("base", "high", "low"):
        frame = scenarios[scenario_name].copy()
        if frame.empty or variable not in frame.columns:
            values = [None] * len(selected_dates)
        else:
            frame["_quarter"] = frame["Date"].astype(str).map(quarter_label)
            values = [
                frame.loc[frame["_quarter"] == label, variable].iloc[0]
                if (frame["_quarter"] == label).any()
                else None
                for label in selected_labels
            ]
        rows.append([scenario_name, *values])

    rows.append(["final", *([None] * len(selected_dates))])
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
default_quarter = current_quarter()
if default_quarter not in quarter_options and quarter_options:
    default_quarter = quarter_options[0]
if "applied_quarter" not in st.session_state:
    st.session_state.applied_quarter = default_quarter

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
        div[data-testid="stDataFrame"] { border:1px solid #dfe4eb; border-radius:8px; }
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
                prediction_quarter = st.selectbox(
                    "예측시점",
                    quarter_options,
                    index=quarter_options.index(st.session_state.applied_quarter),
                    label_visibility="collapsed",
                )
            with search_column:
                if st.button("조회하기", type="primary", width="stretch"):
                    st.session_state.applied_quarter = prediction_quarter
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
                f"{st.session_state.applied_quarter}부터 4개 분기"
            )
            scenario_table = build_scenario_table(
                scenarios,
                st.session_state.selected_variable,
                st.session_state.applied_quarter,
            )
            st.dataframe(scenario_table, width="stretch")
            commit_space, commit_button = st.columns([3, 1])
            with commit_button:
                if st.button("commit", type="primary", width="stretch"):
                    st.toast("commit 기능은 다음 단계에서 연결합니다.")
else:
    st.divider()
    st.caption("품목을 선택하고 적용하면 주요 변수 설정 화면이 표시됩니다.")
