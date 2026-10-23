# -*- coding: utf-8 -*-
"""변수별 모델 선정 -> 선정 모델의 90% 구간 예측 -> 스트림릿 시나리오 CSV.

실행: python -m model_selection.build_scenarios [변수코드 ...] [--out CSV 경로]   (변수코드를 안 주면 금 주요변수 6개 전부)
CSV: origin(예측시점 분기), quarter(예측 분기), scenario(high=95%, base=점예측(12시리즈와 같은 중앙값), low=5%), 변수 열(앱 열 이름).
변수마다 데이터 끝까지만 쓰므로 어떤 origin에 없는 변수는 빈 값이다.
"""
import argparse
from functools import reduce

import pandas as pd

from .selection import ROOT, forecast_table, run_grid, select
from .variables import GOLD

SCEN = (("high", "상한"), ("base", "중앙값"), ("low", "하한"))


def build(cfgs):
    parts = []
    for cfg in cfgs:
        res = run_grid(cfg)
        s = select(res)
        print(f"{cfg.code}: 선정 {s['선정']}, 후보 {len(s['후보'])}/{s['전체 조합']}, 폭주 제외 {s['폭주 제외']}")
        t = forecast_table(res, s["선정"])
        col = cfg.app_column or cfg.code
        parts.append(pd.concat([t.assign(scenario=sc, **{col: t[v]})[["예측시점", "분기", "scenario", col]] for sc, v in SCEN]).rename(columns={"예측시점": "origin", "분기": "quarter"}))
    df = reduce(lambda a, b: a.merge(b, on=["origin", "quarter", "scenario"], how="outer"), parts)
    df["scenario"] = pd.Categorical(df["scenario"], [sc for sc, _ in SCEN])
    return df.sort_values(["origin", "scenario", "quarter"]).reset_index(drop=True).astype({"scenario": str})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*")
    ap.add_argument("--out", default="data/gold/gold_major_variables_forecast_model.csv")
    a = ap.parse_args()
    df = build([GOLD[c] for c in (a.codes or GOLD)])
    df.to_csv(ROOT / a.out, index=False, float_format="%.17g", encoding="utf-8-sig")
    print("저장:", a.out, df.shape)
