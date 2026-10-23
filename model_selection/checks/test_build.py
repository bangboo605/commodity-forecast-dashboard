# -*- coding: utf-8 -*-
"""build_scenarios가 앱 CSV 형식으로 값을 만드는지 확인한다(X2). 실행: python -m model_selection.checks.test_build"""
import numpy as np
import pandas as pd

from model_selection.build_scenarios import build
from model_selection.selection import ROOT
from model_selection.variables import GOLD


def test_matches_app_csv():
    new = build([GOLD["X2"]])
    app = pd.read_csv(ROOT / "data/gold/gold_major_variables_forecast_model.csv")           # 12시리즈 캐시로 만든 앱 CSV(3.8)
    assert list(new.columns) == ["origin", "quarter", "scenario", "X2"]
    assert set(new["scenario"]) == {"high", "base", "low"}
    app = app.dropna(subset=["X2"])                                                  # 변수마다 데이터가 있는 구간만 값이 있다
    m = new.merge(app[["origin", "quarter", "scenario", "X2"]], on=["origin", "quarter", "scenario"], suffixes=("_new", "_app"))
    assert len(m) == len(app) == len(new), (len(m), len(app), len(new))
    assert np.allclose(m["X2_new"], m["X2_app"], rtol=0, atol=1e-12)
    last = new[new["origin"] == new["origin"].max()]
    print("행", len(new), "| origin", new["origin"].min(), "~", new["origin"].max(), "| 앱 CSV와 최대 차이", float((m["X2_new"] - m["X2_app"]).abs().max()))
    print(last.head(3).to_string())


if __name__ == "__main__":
    test_matches_app_csv()
    print("OK")
