# -*- coding: utf-8 -*-
"""시계열 CSV 한 개로 모델을 고르고, 첫 origin부터 모든 origin의 예측 분위수와 통합성적표를 만든다.

실행:  (PI_module 폴더에서) python PI.py <CSV 이름> <첫 origin의 마지막 분기 날짜>      예) python PI.py gold_yoy 2024-01-01
함수:  d1, d2 = run_pi("gold_yoy", "2018-12-01")

- CSV: Date, Data 두 열(분기 자료). Date는 그 날짜가 속한 분기로 읽는다(2018-12-01은 2018-Q4). 이름만 주면 현재 폴더, PI_module/data/ 순으로 찾는다. 결과 CSV는 PI_module/out/에 저장한다.
- 첫 origin: 입력한 날짜의 분기까지를 훈련에 쓰는 origin. 그보다 앞(최소 훈련 20분기부터)은 리샘플링 오차 풀을 쌓는 보정 구간이다.
- 모델 선정: 지금까지와 같다. 모델 3종 x (d, D) 4가지 x 스케일 x 구간 방식을 첫 origin 이후 백테스트해서
  CRPS 1등의 +-10% 이내 중 커버리지가 90%에 가장 가까운 조합을 고른다.
- d1: {origin의 마지막 분기('2018-Q4'): 9 x 4 배열}  행은 5, 10, 20, 30, 50, 70, 80, 90, 95% 분위수, 열은 h=1~4(원래 단위).
      데이터 끝 이후 4분기를 예측하는 마지막 origin까지 모두 들어 있다.
- d2: {'mse', 'crps', 'coverage'}  정답이 4분기 모두 있는 origin의 평균. mse와 crps는 노트북과 같이 전체 표준편차로 정규화한 값, coverage는 90% 구간 적중률(%).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                            # 모델 선정 엔진(selection.py, ts_models.py)이 이 폴더에 같이 있다
import selection as S
LEVELS = (5, 10, 20, 30, 50, 70, 80, 90, 95)             # 분위수(%)
MIN_CALIB = 4                                             # 오차 풀이 비지 않으려면 보정 origin이 h=4를 쌓을 만큼은 필요하다


def _label(p):
    return f"{p.year}Q{p.quarter}"


def _key(lab):
    return f"{lab[:4]}-Q{lab[5]}"                         # '2018Q4' -> '2018-Q4'


def read_series(source):
    """CSV 경로, 이름, 또는 DataFrame(Date, Data)을 (분기 이름 배열, 값 배열)로 바꾼다."""
    if isinstance(source, pd.DataFrame):
        df = source
    else:
        p = Path(source)
        name = p if p.suffix else p.with_suffix(".csv")
        found = [c for c in (name, HERE / name, HERE / "data" / name.name) if c.exists()]
        if not found:
            raise FileNotFoundError(f"CSV를 찾지 못했다: {source} (현재 폴더, PI_module/data/ 확인)")
        df = pd.read_csv(found[0], encoding="utf-8-sig")
    missing = {"Date", "Data"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 Date, Data 두 열이 있어야 한다 (없는 열: {', '.join(sorted(missing))}, 있는 열: {', '.join(map(str, df.columns))})")
    df = df[["Date", "Data"]].reset_index(drop=True)
    try:
        dates = pd.to_datetime(df["Date"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Date를 날짜로 읽지 못했다: {e}")
    val = pd.to_numeric(df["Data"], errors="coerce")
    bad = val.isna() & df["Data"].notna() & (df["Data"].astype(str).str.strip() != "")
    if bad.any():
        i = bad.idxmax()
        raise ValueError(f"Data에 숫자가 아닌 값이 있다 (Date {dates[i].date()}: {df['Data'][i]!r})")
    ok = val.notna()
    if not ok.any():
        raise ValueError("Data에 값이 없다")
    inner = val[ok.idxmax():ok[::-1].idxmax() + 1]                  # 앞뒤의 빈 값은 잘라 내고, 가운데 빈 값은 오류
    if inner.isna().any():
        raise ValueError(f"Data의 가운데에 빈 값이 있다 (Date {dates[inner.isna().idxmax()].date()})")
    dates, val = dates[inner.index], inner
    per = pd.PeriodIndex(dates, freq="Q")
    if len(per) > 1 and not ((per[1:] - per[:-1]).map(lambda x: x.n) == 1).all():
        raise ValueError("Date가 오름차순으로 연속된 분기가 아니다 (순서, 중복, 빠진 분기를 확인하세요)")
    return np.array([_label(p) for p in per]), val.astype(float).to_numpy()


def prepare(source, first_insample_end_date, name="series"):
    """[1][2] 입력을 읽고 첫 origin을 정한다. 분기 이름 q, 값 y, 첫 origin, 보정 origin 수, 엔진 설정 cfg를 담은 딕셔너리를 돌려준다."""
    q, y = read_series(source)
    n = len(y)
    if n < S.MIN_TRAIN + MIN_CALIB + S.H:
        raise ValueError(f"데이터가 너무 짧다: 훈련 {S.MIN_TRAIN} + 보정 {MIN_CALIB} + 정답 {S.H}분기, 최소 {S.MIN_TRAIN + MIN_CALIB + S.H}분기가 필요하다 (현재 {n}분기)")
    if np.std(y) == 0:
        raise ValueError("Data 값이 모두 같아서 모델을 적합할 수 없다")
    try:
        lab = _label(pd.Period(pd.Timestamp(first_insample_end_date), freq="Q"))
    except (ValueError, TypeError) as e:
        raise ValueError(f"첫 origin 날짜를 읽지 못했다: {first_insample_end_date!r} (예: '2019-12-01') | {e}")
    if lab not in q:
        raise ValueError(f"첫 origin의 분기 {lab}가 데이터({q[0]}~{q[-1]})에 없다")
    o0 = int(np.flatnonzero(q == lab)[0]) + 1             # 첫 origin: 훈련 관측 수
    calib = o0 - S.MIN_TRAIN
    if calib < MIN_CALIB:
        raise ValueError(f"첫 origin이 너무 이르다: 훈련 {S.MIN_TRAIN}분기 + 보정 {MIN_CALIB}분기가 필요해 {q[S.MIN_TRAIN + MIN_CALIB - 1]} 이후여야 한다")
    if o0 + S.H > n:
        raise ValueError("첫 origin 뒤에 정답이 4분기 이상 있어야 채점할 수 있다")
    cfg = S.VariableConfig(code=name, csv="", column="Data", initial_calib=calib)
    return {"q": q, "y": y, "첫 origin": o0, "첫 origin 분기": lab, "보정 origin 수": calib, "cfg": cfg}


def select_model(p, verbose=True):
    """[3] 모든 조합을 백테스트하고 모델을 고른다. (조합별 전체 결과, 선정 정보)를 돌려준다."""
    res = S.run_grid(p["cfg"], data=(p["q"], p["y"]))      # 모든 조합 백테스트(캐시 사용)
    sel = S.select(res)
    if verbose:
        m, d, D, sc, iv = sel["선정"]
        print(f"선정 모델: {m} (d={d}, D={D}) {sc} 스케일, {iv} 구간 | 후보 {len(sel['후보'])}/{sel['전체 조합']}개, 보정 origin {p['보정 origin 수']}개")
    return res, sel


def forecast_dict(p, sel):
    """[4] 선정 모델로 모든 origin의 9개 분위수를 만든다 -> 딕셔너리 1 {origin의 마지막 분기: 9 x 4 배열}."""
    m, d, D, sc, iv = sel["선정"]
    # 선정 조합 하나만 분위수와 함께 다시 계산한다(같은 시드라 선정 때와 같은 난수)
    df, _ = S._run_one(p["cfg"], p["q"], p["y"], m, d, D, sc == "로그", levels=[x / 100 for x in LEVELS])
    df = df[df["구간"] == iv]
    d1 = {}
    for o, g in df.groupby("origin"):
        g = g.sort_values("h")
        d1[_key(p["q"][o - 1])] = np.vstack(g["분위수"].to_list()).T      # (9, 4)
    return d1


def score_dict(sel):
    """[5] 통합성적표 -> 딕셔너리 2 {'mse', 'crps', 'coverage'} (선정 조합의 평균, 정규화 단위)."""
    t = sel["표"].loc[sel["선정"]]
    return {"mse": float(t["MSE"]), "crps": float(t["CRPS"]), "coverage": float(t["커버리지(%)"])}


def run_pi(source, first_insample_end_date, name="series", verbose=True):
    """위 네 단계를 한 번에: (딕셔너리 1, 딕셔너리 2)를 돌려준다. 선정 모델은 verbose=True면 화면에 출력한다."""
    p = prepare(source, first_insample_end_date, name)
    _, sel = select_model(p, verbose)
    return forecast_dict(p, sel), score_dict(sel)


def to_frame(d1, source=None):
    """d1을 보기 좋은 긴 표로 바꾼다: origin, h, 예측 분기, p5 ~ p95, (source를 주면) actual."""
    rows = []
    actual = dict(zip(*read_series(source))) if source is not None else {}
    for k, arr in d1.items():
        p0 = pd.Period(f"{k[:4]}{k[5:]}", freq="Q")
        for h in range(arr.shape[1]):
            tq = _label(p0 + h + 1)
            rows.append({"origin": k, "h": h + 1, "예측 분기": tq, **{f"p{l}": arr[i, h] for i, l in enumerate(LEVELS)},
                         **({"actual": actual.get(tq, np.nan)} if source is not None else {})})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("사용법: python PI.py <CSV 이름> <첫 origin의 마지막 분기 날짜>   예) python PI.py gold_yoy 2024-01-01")
    csv_name, date = (x.strip("'\"") for x in sys.argv[1:3])          # cmd처럼 작은따옴표를 글자 그대로 넘기는 터미널도 허용
    code = Path(csv_name).stem
    d1, d2 = run_pi(csv_name, date, name=code)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    to_frame(d1).to_csv(out / f"{code}_forecast.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([d2]).to_csv(out / f"{code}_score.csv", index=False, encoding="utf-8-sig")
    (out / f"{code}_dict1.json").write_text(json.dumps({k: v.tolist() for k, v in d1.items()}, ensure_ascii=False), encoding="utf-8")
    (out / f"{code}_dict2.json").write_text(json.dumps(d2, ensure_ascii=False), encoding="utf-8")
    ks = list(d1)
    print(f"딕셔너리 1: origin {len(ks)}개 ({ks[0]} ~ {ks[-1]}), 값은 9 x 4 배열(행 {LEVELS}%, 열 h1~h4). 첫 origin {ks[0]}:")
    print({ks[0]: np.round(d1[ks[0]], 2).tolist()})
    print("딕셔너리 2:", {k: round(v, 4) for k, v in d2.items()})
    print(f"저장(PI_module/out/): {code}_dict1.json, {code}_dict2.json(딕셔너리 그대로), {code}_forecast.csv(표), {code}_score.csv")
