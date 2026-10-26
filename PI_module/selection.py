# -*- coding: utf-8 -*-
"""[PI_module 폴더용 복사본] model_selection/selection.py에서 가져온 것이다(상대 import와 캐시 위치만 PI 폴더 기준으로 바꿈).

변수 설정 -> 모든 조합 워크포워드 계산 -> 모델 선정 (12시리즈 방식).

- 조합: 모델(AR단독, GARCH, SV) x (d, D) x 스케일(원, 로그) x 구간(자체, 리샘플링). 로그는 값이 모두 양수일 때만.
- 평가: origin 20분기부터 확장 창, 앞 15개 origin은 오차 풀 쌓기용, 이후 origin을 h=1~4로 채점(CRPS, MSE, 커버리지).
- 미래 정보 없음: 적합, 오차 풀(o' + h <= o), 단위 상수(100 / 훈련 구간 표준편차)가 모두 각 origin 이전 데이터만 쓴다.
- 선정 규칙: CRPS 1등의 +-10% 이내 중 커버리지가 90%에 가장 가까운 조합. 계산이 안 되거나 폭주한 조합(inf)은 후보에서 빠진다.
- 캐시(PI_module/cache/): 설정과 CACHE_VERSION이 같으면 이전 결과를 읽고, 데이터가 달라졌으면 처음 달라진 관측이 영향을 주는 origin부터만 다시 계산한다.
  (origin마다 난수 상태와 오차 풀을 저장해 두어 중간부터 이어도 처음부터 계산한 것과 결과가 같다. 점수는 원래 단위로 저장하고 읽을 때 현재 데이터의 표준편차로 정규화한다.)
"""
import copy
import hashlib
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps
from scipy import stats

from ts_models import ALPHA, H, Z, Spec, forecast          # PI_module 폴더 안의 복사본

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(__file__).resolve().parent / "cache"
SCALE, N_ENS, MIN_TRAIN, INITIAL_CALIB = 100.0, 500, 20, 15
SEED = 42                  # 리샘플링 난수 시드. 모든 변수, 모델, (d, D)가 같은 시드를 써서 같은 난수로 비교한다(모델 간 비교에서 난수 잡음이 상쇄된다)
KEY = ["모델", "d", "D", "스케일", "구간"]


@dataclass
class VariableConfig:
    code: str                                   # 변수 코드(데이터 코드가 생기기 전까지는 열 이름)
    csv: str                                    # 프로젝트 루트 기준 CSV 경로 ('quarter' 열이 있어야 함)
    column: str
    models: tuple = ("AR단독", "GARCH", "SV")
    dD: tuple = ((0, 0), (0, 1), (1, 0), (1, 1))
    scales: tuple = ("원", "로그")                # 로그는 값이 모두 양수일 때만 쓴다
    app_column: str = ""                             # 스트림릿 시나리오 CSV의 열 이름(없으면 변수 코드)
    initial_calib: int = 15                          # 첫 채점 origin 앞의 보정 origin 수(오차 풀 쌓기용). PI에서는 첫 origin 날짜로 정한다


def load(cfg):
    raw = pd.read_csv(ROOT / cfg.csv, encoding="utf-8-sig").dropna(subset=["quarter"])
    s = raw[["quarter", cfg.column]].dropna()
    return s["quarter"].values, s[cfg.column].values.astype(float)


def _next_quarter(lab):
    y_, n_ = int(lab[:4]), int(lab[5])
    return f"{y_ + (n_ == 4)}Q{n_ % 4 + 1}"


def _run_one(cfg, q, y, model, d, D, logmode, prev=None, o_start=MIN_TRAIN, levels=None):
    """한 조합의 origin별 결과(원래 단위, 점수는 정규화 전). prev=(이전 결과, 체크포인트)가 있으면 o_start origin부터만 다시 계산해 이어 붙인다.
    levels(분위수 확률 목록)를 주면 두 구간 방식 모두 행마다 '분위수'(원래 단위 배열)를 함께 돌려준다."""
    sp, std = Spec(d=d, D=D), model != "AR단독"
    unit_err = (not std) and (not logmode)                    # 모델 단위로 쌓이는 오차는 원래 단위로 저장해 둔다
    tf = (lambda x: np.exp(np.asarray(x) / SCALE)) if logmode else (lambda x: np.asarray(x))
    U = stats.norm.ppf((np.arange(N_ENS) + 0.5) / N_ENS)
    rng, errs, rows, n = np.random.default_rng(SEED), {h: [] for h in range(1, H + 1)}, [], len(y)
    kept, ckpt = None, {}
    if prev is not None and o_start > MIN_TRAIN:              # 체크포인트(난수 상태, 오차 풀)에서 이어서 계산한다
        old, old_ckpt = prev
        state, e0 = old_ckpt[o_start]
        rng.bit_generator.state = copy.deepcopy(state)
        errs = {h: list(v) for h, v in e0.items()}
        kept, ckpt = old[old["origin"] < o_start], {o: v for o, v in old_ckpt.items() if o < o_start}
    qfull = list(q)
    while len(qfull) < n + H:
        qfull.append(_next_quarter(qfull[-1]))                # 데이터 이후의 예측 분기 이름
    for o in range(o_start if kept is not None else MIN_TRAIN, n + 1):      # 데이터 끝까지의 모든 origin (끝쪽 origin은 정답이 일부 또는 전부 없다)
        ckpt[o] = (copy.deepcopy(rng.bit_generator.state), {h: list(v) for h, v in errs.items()})
        C = 100 / y[:o].std()                                 # 단위 상수: 그 시점까지의 훈련 구간 표준편차
        series = SCALE * np.log(y * C) if logmode else y * C
        fc, sig, src, (p, P) = forecast(model, series[:o], sp)
        if o - MIN_TRAIN >= cfg.initial_calib:
            for h in range(1, H + 1):
                known = o + h - 1 < n                             # 정답이 아직 없는 분기는 예측만 하고 채점하지 않는다
                a, s_h = (y[o + h - 1] if known else np.nan), max(sig[h - 1], 1e-6)
                pool = np.array([e for o2, e in errs[h] if o2 + h <= o]) * (C if unit_err else 1)
                samples = fc[h - 1] + (s_h if std else 1.0) * rng.choice(pool - np.median(pool), size=N_ENS, replace=True)
                lo_r, hi_r = tf(np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2])) / C
                q50_r = float(tf(np.quantile(samples, 0.5))) / C            # 리샘플링 분포의 50% 분위수(구간 끝 5%, 95%와 같은 분포)
                lo_n, hi_n = tf(fc[h - 1] - Z * s_h) / C, tf(fc[h - 1] + Z * s_h) / C
                med = float(tf(fc[h - 1])) / C
                if levels is not None:
                    qs_r, qs_n = tf(np.quantile(samples, levels)) / C, tf(fc[h - 1] + stats.norm.ppf(levels) * s_h) / C
                base = {"모델": model, "d": d, "D": D, "스케일": "로그" if logmode else "원", "origin": o, "h": h, "분기": qfull[o + h - 1], "p": p, "P": P,
                        "src": src, "mse_raw": (a - med) ** 2, "중앙값": med, "실제": a, "채점": o + H <= n}
                rows.append({**base, "구간": "리샘플링", "q50": q50_r, "하한": lo_r, "상한": hi_r, "적중": float(lo_r <= a <= hi_r) * 100 if known else np.nan,
                             "폭_raw": hi_r - lo_r, "crps_raw": ps.crps_ensemble(a, tf(samples) / C) if known else np.nan,
                             **({"분위수": qs_r} if levels is not None else {})})
                rows.append({**base, "구간": "자체", "q50": med, "하한": lo_n, "상한": hi_n, "적중": float(lo_n <= a <= hi_n) * 100 if known else np.nan,
                             "폭_raw": hi_n - lo_n, "crps_raw": ps.crps_ensemble(a, tf(fc[h - 1] + s_h * U) / C) if known else np.nan,
                             **({"분위수": qs_n} if levels is not None else {})})
        for h in range(1, H + 1):
            if o + h - 1 < n and np.isfinite(fc[h - 1]):      # 정답이 있는 오차만, 적합 실패(NaN)는 오차 풀에 넣지 않는다
                e = series[o + h - 1] - fc[h - 1]
                errs[h].append((o, e / max(sig[h - 1], 1e-6) if std else e / (1 if logmode else C)))
    df = pd.DataFrame(rows)
    return (pd.concat([kept, df], ignore_index=True) if kept is not None else df), ckpt


CACHE_VERSION = 2          # 계산 방식(모델, 평가, 저장 형식)을 바꿀 때만 올린다. 그러면 이전 캐시를 읽지 않고 처음부터 다시 계산한다.


def _cache_path(cfg):
    """설정, 시드, CACHE_VERSION이 같을 때만 같은 캐시 파일 이름이 나온다(데이터는 캐시 안에 같이 저장해 비교한다)."""
    d = asdict(cfg)
    if d["initial_calib"] == 15:                                # 기본값이면 이 설정이 생기기 전과 같은 캐시 이름을 쓴다
        d.pop("initial_calib")
    return CACHE_DIR / f"{cfg.code}_{hashlib.sha1(repr(d).encode()).hexdigest()[:10]}_s{SEED}_v{CACHE_VERSION}.pkl"


def _first_diff(q0, y0, q1, y1):
    """이전 데이터와 처음 달라지는 관측 위치. 같으면 None, 한쪽이 더 짧으면 짧은 쪽 길이."""
    m = min(len(y0), len(y1))
    bad = np.flatnonzero((np.asarray(q0[:m]) != np.asarray(q1[:m])) | (np.asarray(y0[:m]) != np.asarray(y1[:m])))
    return int(bad[0]) if len(bad) else (m if len(y0) != len(y1) else None)


def _normalize(res, y):
    sd = y.std()                                              # 점수 정규화 단위: 현재 데이터의 표준편차
    res = res.copy()
    res["crps"], res["MSE"], res["폭"] = res["crps_raw"] / sd, res["mse_raw"] / sd ** 2, res["폭_raw"] / sd
    return res


def run_grid(cfg, use_cache=True, data=None):
    """모든 조합을 계산해 (조합, origin, h)별 점수와 예측값 표를 돌려준다. data=(분기, 값)을 주면 CSV 대신 그 데이터를 쓴다(테스트용)."""
    q, y = data if data is not None else load(cfg)
    path, prev, o_start = _cache_path(cfg), None, MIN_TRAIN
    if use_cache and path.exists():
        prev = pd.read_pickle(path)
        k = _first_diff(prev["q"], prev["y"], q, y)
        if k is None:
            return _normalize(pd.concat(prev["parts"].values(), ignore_index=True), y)
        o_start = max(MIN_TRAIN, k - H + 1)                   # 이 관측을 훈련에 쓰거나 정답으로 쓰는 첫 origin
    logs = (False, True) if (y > 0).all() and "로그" in cfg.scales else (False,)
    parts, ckpts = {}, {}
    for m, (d, D), lg in itertools.product(cfg.models, cfg.dD, logs):
        if not (lg or "원" in cfg.scales):
            continue
        k_ = (m, d, D, lg)
        pv = (prev["parts"][k_], prev["ckpt"][k_]) if prev is not None and k_ in prev["parts"] else None
        parts[k_], ckpts[k_] = _run_one(cfg, q, y, m, d, D, lg, pv, o_start)
    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle({"q": q, "y": y, "parts": parts, "ckpt": ckpts}, path)
    return _normalize(pd.concat(parts.values(), ignore_index=True), y)


def table(res):
    """조합별 CRPS, MSE, 커버리지, 폭 (h 평균 -> origin 평균). 계산이 안 된 값은 inf로 두어 평균이 inf가 된다."""
    res = res[res["채점"]]                                         # 4개 분기 정답이 모두 있는 origin만 채점한다(12시리즈와 같음)
    per = res.fillna({"crps": np.inf, "MSE": np.inf, "폭": np.inf}).groupby(KEY + ["origin"])[["crps", "MSE", "적중", "폭"]].mean(skipna=False)
    return per.groupby(KEY).mean(skipna=False).rename(columns={"crps": "CRPS", "적중": "커버리지(%)"})


def select(res):
    """선정 조합과 근거를 돌려준다: CRPS 1등의 +-10% 이내 중 커버리지가 90%에 가장 가까운 조합."""
    tab = table(res)
    fin = tab[np.isfinite(tab["CRPS"])]
    best = fin["CRPS"].min()
    cand = fin[fin["CRPS"] <= best * 1.10].sort_values("CRPS")
    cand = cand.iloc[np.argsort((cand["커버리지(%)"] - 90).abs().values, kind="stable")]       # 90%에 가까운 순, 같으면 CRPS가 낮은 순
    return {"선정": tuple(cand.index[0]), "후보": cand, "CRPS 1등": fin["CRPS"].idxmin(), "CRPS 1등 값": best,
            "전체 조합": len(tab), "폭주 제외": len(tab) - len(fin), "표": tab}


def forecast_table(res, selected):
    """선정 조합의 모든 origin 예측(90% 구간): 예측시점(첫 예측 분기), 분기, h, 하한(5%), 중앙값(점예측, 12시리즈와 같음), q50(구간 분포의 50% 분위수, 참고), 상한(95%). 정답이 없는 끝쪽 origin도 포함한다."""
    m, d, D, sc, iv = selected
    r = res[(res["모델"] == m) & (res["d"] == d) & (res["D"] == D) & (res["스케일"] == sc) & (res["구간"] == iv)]
    first = r[r["h"] == 1].set_index("origin")["분기"]
    return r.assign(예측시점=r["origin"].map(first))[["예측시점", "분기", "h", "하한", "중앙값", "q50", "상한"]].reset_index(drop=True)
