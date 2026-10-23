# -*- coding: utf-8 -*-
"""X2(24조합)로 12시리즈 결과를 재현하는지 확인한다. 실행: python -m model_selection.checks.test_X2"""
import numpy as np
import pandas as pd

from model_selection.selection import ROOT, KEY, VariableConfig, run_grid, select

CFG = VariableConfig("X2", "data/gold/gold_major_variables.csv", "X2")


def test_reproduces_12_series():
    res = run_grid(CFG, use_cache=False)
    res = res[res["채점"]]                                                               # 채점 origin만 비교한다
    full = run_grid(CFG)
    old = pd.read_pickle(ROOT / "notebooks/cache/cache_128_X2_df.pkl")          # 12-9의 캐시(훈련 구간 단위 상수)
    idx = KEY + ["origin", "h"]
    a, b = res.set_index(idx).sort_index(), old.set_index(idx).sort_index()
    assert a.index.equals(b.index)
    for c in ("crps", "MSE", "적중", "폭"):
        assert np.allclose(a[c], b[c], rtol=0, atol=1e-9, equal_nan=True), c
    s = select(full)
    assert s["선정"] == ("AR단독", 0, 0, "원", "자체"), s["선정"]
    assert len(s["후보"]) == 2 and abs(s["CRPS 1등 값"] - 0.5248) < 5e-5
    assert s["전체 조합"] == 24 and s["폭주 제외"] == 0
    print("선정:", s["선정"], "| 후보", len(s["후보"]), "| CRPS 1등", round(s["CRPS 1등 값"], 4))


if __name__ == "__main__":
    test_reproduces_12_series()
    print("OK")
