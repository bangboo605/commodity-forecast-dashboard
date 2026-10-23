# -*- coding: utf-8 -*-
"""시드를 바꿔도 선정이 같은지 확인한다.

python -m model_selection.checks.seed_check run 42 X2     # 시드 42로 X2를 계산해 캐시에 저장(변수마다 병렬로 돌릴 수 있다)
python -m model_selection.checks.seed_check table 42 1 2 3 4    # 시드별 선정 표(캐시가 없으면 계산)
"""
import sys

import pandas as pd

from model_selection import selection as S
from model_selection.variables import GOLD


def label(s):
    return f"{s[0]}·({s[1]},{s[2]})·{s[3]}·{s[4]}"


def table(seeds):
    rows = {}
    for sd in seeds:
        S.SEED = sd
        for code, cfg in GOLD.items():
            sel = S.select(S.run_grid(cfg))
            rows.setdefault(code, {})[sd] = (sel["선정"], sel["후보"].iloc[0], sel["CRPS 1등"])
    out = pd.DataFrame({sd: {c: label(r[sd][0]) for c, r in rows.items()} for sd in seeds})
    same = {c: len({r[sd][0] for sd in seeds}) == 1 for c, r in rows.items()}
    out["모든 시드에서 같음"] = pd.Series(same)
    return out, rows


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "run":
        S.SEED = int(a[0])
        S.run_grid(GOLD[a[1]])
        print("완료", a)
    else:
        out, _ = table([int(x) for x in a])
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(out.to_string())
