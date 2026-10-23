# Version 1.0
import streamlit as st


st.set_page_config(
    page_title="품목 선택",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ITEMS = ["비철금속", "전기동"]

if "selector_open" not in st.session_state:
    st.session_state.selector_open = False
if "selected_item" not in st.session_state:
    st.session_state.selected_item = "전기동"

st.markdown(
    """
    <style>
        .stApp { background: #f7f8fa; }
        .block-container { max-width: 1050px; padding-top: 3rem; }
        .eyebrow {
            color: #697386; font-size: .78rem; font-weight: 700;
            letter-spacing: .08em; margin-bottom: .25rem;
        }
        .main-title { margin: 0 0 .25rem; font-size: 1.75rem; color: #172033; }
        .description { color: #697386; margin-bottom: 1.5rem; }
        .selected-card {
            background: white; border: 1px solid #e3e7ed; border-radius: 10px;
            padding: .7rem .85rem; box-shadow: 0 3px 12px rgba(23,32,51,.05);
        }
        .selected-label { color: #7a8495; font-size: .7rem; margin-bottom: .15rem; }
        .selected-value { color: #172033; font-size: .95rem; font-weight: 700; }
        div[data-testid="stButton"] > button[kind="primary"] {
            min-height: 40px; border-radius: 8px; font-weight: 700;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: white; border-color: #dfe4eb; border-radius: 12px;
            box-shadow: 0 6px 20px rgba(23,32,51,.07);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">COMMODITY DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">원자재 가격 대시보드</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">분석할 품목을 선택해 주세요.</div>',
    unsafe_allow_html=True,
)

button_column, selected_column, panel_column = st.columns([.9, 1.2, 2.6], gap="medium")

with button_column:
    button_text = "선택창 닫기" if st.session_state.selector_open else "품목 선택"
    if st.button(button_text, type="primary", width="stretch"):
        st.session_state.selector_open = not st.session_state.selector_open
        st.rerun()

with selected_column:
    st.markdown(
        f"""
        <div class="selected-card">
            <div class="selected-label">현재 선택 품목</div>
            <div class="selected-value">{st.session_state.selected_item}</div>
        </div>
        """,
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
                    st.session_state.selector_open = False
                    st.rerun()
            with cancel_column:
                if st.button("취소", width="stretch"):
                    st.session_state.selector_open = False
                    st.rerun()
    else:
        st.info("왼쪽의 **품목 선택** 버튼을 누르면 선택창이 열립니다.")

st.divider()
st.caption("다음 단계에서 선택된 품목의 데이터와 차트를 아래에 연결할 수 있습니다.")
