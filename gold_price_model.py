"""TVP(시변계수) 금 가격 모델 — final 값 기반 예측.

근거(ground truth): notebooks/2.예측시점 별 출력값.ipynb. 핵심 메커니즘은 2016Q1 고정
시작 + 예측시점 직전까지 유동적 훈련기간, order=(0,0,0) TVP SARIMAX,
get_prediction(dynamic=True)로 다단계 예측. 학습용 원본 데이터는
`data/gold/gold_major_variables.csv` + `gold_price.csv`에서 직접 읽는다
(부동소수점 정밀도 손실 없이 float_precision="round_trip"으로 파싱).

이 모델은 X(주요변수)가 있어야 Y(yoy_log_gold)를 계산하는 회귀모델이지, X 자체를 예측하지
않는다. 예측 대상 분기의 X값은 항상 호출부(raw_final_values, 즉 committed final 값)에서
와야 하며, 실제 관측치가 없는 미래 분기는 시나리오/전망치를 넣어줘야 계산이 가능하다.
"""
from math import exp

import numpy as np
import pandas as pd
import streamlit as st

from config import GOLD_PRICE_PATH, ITEM_DATA_DIRS

ACTUAL_CSV = ITEM_DATA_DIRS["gold"] / "gold_major_variables.csv"
PRICE_CSV = GOLD_PRICE_PATH

TRAIN_START = "2016Q1"
TVP_MAXITER = 3000

MODEL_X_COLS = [
    "X1_l1",
    "X2",
    "X3_l1",
    "X4",
    "X5_l1",
    "X6_l1",
]
STD_X_COLS = [
    "X1_l1",
    "X2",
    "X3_l1",
    "X4_z",
    "X5_l1",
    "X6_l1",
]


def _q_to_ts(quarter: str) -> pd.Timestamp:
    return pd.Period(quarter, freq="Q").to_timestamp()


def _load_source_data() -> pd.DataFrame:
    x_actual = pd.read_csv(ACTUAL_CSV, encoding="utf-8-sig", float_precision="round_trip")
    x_actual = x_actual.dropna(subset=["quarter"])

    price = pd.read_csv(PRICE_CSV, encoding="utf-8-sig", float_precision="round_trip")
    price.columns = ["quarter", "gold", "gold_pred"]  # 분기 / 실제가 / 예측가
    price = price.dropna(subset=["quarter"])[["quarter", "gold"]]

    df = x_actual.merge(price, on="quarter", how="outer")
    df["qdate"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
    df = df.sort_values("qdate").reset_index(drop=True)

    df["log_gold"] = np.log(df["gold"])
    df["yoy_log_gold"] = df["log_gold"] - df["log_gold"].shift(4)
    return df


@st.cache_data
def _cached_source_data(actual_mtime_ns: int, price_mtime_ns: int) -> pd.DataFrame:
    """원본 CSV가 바뀔 때만 다시 로드; mtime이 캐시 키 역할."""
    del actual_mtime_ns, price_mtime_ns
    return _load_source_data()


@st.cache_resource
def _fit_train_tvp(
    actual_mtime_ns: int, price_mtime_ns: int, train_end_q: str
) -> tuple[np.ndarray, pd.Series, pd.Series, float, float]:
    """train_end_q(예측시점 직전 분기)까지의 실제 데이터로만 TVP MLE 파라미터를 학습.

    train_end_q가 바뀌면 다른 훈련기간이므로 별도로 다시 학습(캐시 키에 포함).
    """
    import statsmodels.api as sm

    del actual_mtime_ns, price_mtime_ns
    df = _load_source_data()
    d = df[["quarter", "qdate", "yoy_log_gold", *MODEL_X_COLS]].dropna().reset_index(drop=True)
    d = d[d["qdate"] >= _q_to_ts(TRAIN_START)].reset_index(drop=True)

    train = d[d["qdate"] <= _q_to_ts(train_end_q)].reset_index(drop=True)

    # X4 이중 표준화 1단계: train 기준 z-score (leakage 방지).
    mu_b4 = float(train["X4"].mean())
    sd_b4 = float(train["X4"].std(ddof=0)) or 1.0
    train = train.copy()
    train["X4_z"] = (train["X4"] - mu_b4) / sd_b4

    # 2단계: STD_X_COLS 전체(이미 z-score된 X4 포함)를 train 기준으로 한 번 더 표준화.
    mu = train[STD_X_COLS].mean()
    sd = train[STD_X_COLS].std(ddof=0).replace(0, 1.0)
    train_std = train.copy()
    train_std[STD_X_COLS] = (train_std[STD_X_COLS] - mu) / sd

    exog_tr = np.column_stack(
        [np.ones(len(train_std)), train_std[STD_X_COLS].astype(float).to_numpy()]
    )
    mod_tr = sm.tsa.SARIMAX(
        endog=train_std["yoy_log_gold"].astype(float).to_numpy(),
        exog=exog_tr,
        order=(0, 0, 0),
        trend="n",
        time_varying_regression=True,
        mle_regression=False,
    )
    try:
        res_tr = mod_tr.fit(disp=False, maxiter=TVP_MAXITER)
    except Exception:
        res_tr = mod_tr.fit(disp=False, maxiter=TVP_MAXITER * 2, method="powell")

    return np.asarray(res_tr.params), mu, sd, mu_b4, sd_b4


def predict_final_gold_prices(
    raw_final_values: dict[str, dict[str, float]],
    quarters: list[str],
    column_aliases: dict[str, str],
) -> dict[str, float]:
    """committed final 값으로 `quarters`의 금 가격 레벨을 예측 (v1과 동일한 시그니처).

    raw_final_values: 앱 변수 키 -> {quarter("YYYY-Qn"): 원본 값}.
    quarters: 예측 대상 분기, "YYYY-Qn" 포맷, 오름차순. quarters[0]이 예측시점.
    column_aliases: ITEM_VARIABLE_CONFIG["gold"]["actual_column_aliases"] (모델 x_col -> 앱 변수 키).
    """
    import statsmodels.api as sm

    actual_mtime_ns = ACTUAL_CSV.stat().st_mtime_ns
    price_mtime_ns = PRICE_CSV.stat().st_mtime_ns
    df = _cached_source_data(actual_mtime_ns, price_mtime_ns)
    d = df[["quarter", "qdate", "yoy_log_gold", *MODEL_X_COLS]].dropna().reset_index(drop=True)
    d = d[d["qdate"] >= _q_to_ts(TRAIN_START)].reset_index(drop=True)
    price_lookup = df.dropna(subset=["gold"]).set_index("quarter")["gold"]

    prediction_quarter = quarters[0].replace("-", "")
    train_end_q = str(pd.Period(prediction_quarter, freq="Q") - 1)
    train = d[d["qdate"] <= _q_to_ts(train_end_q)].reset_index(drop=True)
    if train.empty:
        return {}

    params, mu, sd, mu_b4, sd_b4 = _fit_train_tvp(actual_mtime_ns, price_mtime_ns, train_end_q)

    train = train.copy()
    train["X4_z"] = (train["X4"] - mu_b4) / sd_b4

    forecast_rows = []
    for quarter in quarters:
        model_quarter = quarter.replace("-", "")
        x_raw = {}
        for model_col in MODEL_X_COLS:
            app_key = column_aliases.get(model_col, model_col)
            value = raw_final_values.get(app_key, {}).get(quarter)
            if value is None:
                break
            x_raw[model_col] = value
        if len(x_raw) < len(MODEL_X_COLS):
            continue
        row = {"quarter": model_quarter, "qdate": _q_to_ts(model_quarter), "yoy_log_gold": np.nan}
        row.update(x_raw)
        row["X4_z"] = (x_raw["X4"] - mu_b4) / sd_b4
        forecast_rows.append(row)
    if not forecast_rows:
        return {}

    full = pd.concat([train, pd.DataFrame(forecast_rows)], ignore_index=True)
    full[STD_X_COLS] = (full[STD_X_COLS] - mu) / sd

    exog_full = np.column_stack(
        [np.ones(len(full)), full[STD_X_COLS].astype(float).to_numpy()]
    )
    mod_full = sm.tsa.SARIMAX(
        endog=full["yoy_log_gold"].astype(float).to_numpy(),
        exog=exog_full,
        order=(0, 0, 0),
        trend="n",
        time_varying_regression=True,
        mle_regression=False,
    )
    res_f = mod_full.filter(params)

    start_i = len(train)
    end_i = len(full) - 1
    pred = res_f.get_prediction(start=start_i, end=end_i, dynamic=True)
    yhat = np.asarray(pred.predicted_mean)

    combined_price_lookup = price_lookup.to_dict()
    predicted_quarters = full.iloc[start_i:]["quarter"].tolist()

    prices: dict[str, float] = {}
    for i, model_quarter in enumerate(predicted_quarters):
        prev_quarter = str(pd.Period(model_quarter, freq="Q") - 4)
        price_4q_ago = combined_price_lookup.get(prev_quarter)
        if price_4q_ago is None or pd.isna(price_4q_ago):
            continue
        price = float(price_4q_ago) * exp(yhat[i])
        app_quarter = quarters[i]
        prices[app_quarter] = price
        combined_price_lookup[model_quarter] = price

    return prices
