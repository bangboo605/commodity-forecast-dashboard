# -*- coding: utf-8 -*-
"""증분 계산 확인(X2): 데이터가 바뀐 상황마다 증분 결과가 처음부터 계산한 결과와 같고, 다시 계산하는 양이 줄어드는지 본다.
실행: python -m model_selection.checks.test_incremental [변수 코드, 기본 X2]. 로그 스케일이 있는 변수(예: X5_l1)도 확인한다."""
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np

from model_selection import selection as S

from model_selection.variables import GOLD

CFG = GOLD[sys.argv[1] if len(sys.argv) > 1 else "X2"]
q0, y0 = S.load(CFG)
calls = {"n": 0}
_orig = S.forecast


def _counted(*a, **k):
    calls["n"] += 1
    return _orig(*a, **k)


S.forecast = _counted
main, base_dir = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())


def fresh(data):
    """다른 캐시 폴더에서 처음부터 계산한 기준 결과와 계산 횟수."""
    S.CACHE_DIR = base_dir
    for f in base_dir.glob("*.pkl"):
        f.unlink()
    calls["n"] = 0
    res = S.run_grid(CFG, data=data)
    S.CACHE_DIR = main
    return res, calls["n"]


def same(a, b):
    cols = ["crps", "MSE", "적중", "폭", "하한", "상한", "중앙값", "q50", "실제"]
    a, b = a.sort_values(S.KEY + ["origin", "h"]).reset_index(drop=True), b.sort_values(S.KEY + ["origin", "h"]).reset_index(drop=True)
    return a.shape == b.shape and all(np.allclose(a[c], b[c], rtol=0, atol=1e-12, equal_nan=True) for c in cols) and (a["채점"] == b["채점"]).all()


def scenario(name, data, start_cache, less=True):
    for f in main.glob("*.pkl"):
        f.unlink()
    shutil.copy(start_cache, main / S._cache_path(CFG).name)    # 항상 원래 데이터의 캐시에서 시작한다
    S.CACHE_DIR = main
    calls["n"] = 0
    inc = S.run_grid(CFG, data=data)
    n_inc = calls["n"]
    base, n_full = fresh(data)
    ok = same(inc, base)
    print(f"{name}: 결과 {'같음' if ok else '다름'} | 적합 횟수 증분 {n_inc} / 처음부터 {n_full} ({n_inc / n_full * 100:.0f}%)")
    assert ok and (n_inc < n_full if less else n_inc <= n_full)
    return n_inc / n_full


def test_incremental():
    S.CACHE_DIR = main
    S.run_grid(CFG, data=(q0, y0))                              # 원래 데이터로 첫 계산
    start = next(main.glob("*.pkl"))
    keep = Path(tempfile.mkdtemp()) / "start_cache.pkl"
    shutil.copy(start, keep)
    i = lambda lab: int(np.flatnonzero(q0 == lab)[0])
    mid = y0.copy()
    mid[i("2023Q1"):i("2023Q4") + 1] *= 1.3                      # 2-1. 중간의 연속 구간(2023년 4개 분기)만 바뀜
    scenario("중간 구간만 바뀜(2023Q1~Q4)", (q0, mid), keep)
    mix = y0.copy()
    for lab in ("2021Q2", "2023Q3", "2025Q1"):                   # 2-2. 사이사이 떨어진 관측만 바뀜
        mix[i(lab)] += 0.004
    scenario("사이사이만 바뀜(2021Q2, 2023Q3, 2025Q1)", (q0, mix), keep)
    app_q = np.append(q0, S._next_quarter(q0[-1]))
    app_y = np.append(y0, y0[-1] + 0.0007)                       # 2-3. 끝에 한 분기 추가
    scenario("끝에 1분기 추가", (app_q, app_y), keep)
    app2_q = np.append(app_q, S._next_quarter(app_q[-1]))
    app2_y = np.append(app_y, app_y[-1] - 0.0004)
    scenario("끝에 2분기 추가", (app2_q, app2_y), keep)
    last = y0.copy()
    last[-1] += 0.003                                            # 가장 마지막 관측만 바뀜(개정)
    scenario("마지막 관측만 바뀜", (q0, last), keep)
    first = y0.copy()
    first[i("2012Q1")] += 0.004                                  # 아주 앞쪽이 바뀌면 거의 전부 다시 계산해야 한다
    scenario("앞쪽 관측이 바뀜(2012Q1)", (q0, first), keep, less=False)


if __name__ == "__main__":
    test_incremental()
    print("OK")
