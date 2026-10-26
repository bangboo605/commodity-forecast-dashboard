# -*- coding: utf-8 -*-
"""PI 테스트. PI_module 폴더에서 실행한다.

  python test_PI.py          빠른 테스트: 오류가 나야 하는 경우, 입력 표기 변형, 결과 일관성(캐시된 X1)
  python test_PI.py full     위 + 변수별 가장 이른 날짜, 가장 늦은 날짜, 보정 구간이 다른 날짜(새로 계산해서 30분 안팎)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import PI
import selection as S

ENV = {**os.environ, "PYTHONUTF8": "1"}                    # 명령줄 테스트의 자식 프로세스 출력을 UTF-8로 고정 (윈도우 기본 cp949 대응)
RESULTS = []                                                   # (구분, 이름, 통과 여부, 메모)


def record(kind, name, ok, note=""):
    RESULTS.append((kind, name, ok, note))
    print(("통과 " if ok else "실패 ") + f"[{kind}] {name}" + (f" -> {note}" if note else ""), flush=True)


def fmt(lab):
    return f"{lab[:4]}-Q{lab[5]}"


def check_outputs(name, date, source, d1, d2):
    """결과의 형태와 일관성을 확인한다."""
    q, y = PI.read_series(source)
    n = len(y)
    lab = PI._label(pd.Period(pd.Timestamp(date), freq="Q"))
    o0 = int(np.flatnonzero(q == lab)[0]) + 1
    keys = list(d1)
    ok = keys[0] == fmt(lab) and keys[-1] == fmt(q[-1]) and len(keys) == n - o0 + 1
    ok &= all(a.shape == (9, 4) and np.isfinite(a).all() and (np.diff(a, axis=0) >= -1e-9).all() for a in d1.values())
    f = PI.to_frame(d1, source)
    ok &= int(f["actual"].isna().sum()) == 10                  # 데이터 끝 이후 4분기를 예측하는 마지막 4개 origin: 4 + 3 + 2 + 1
    sc = f[f["actual"].notna() & (f["origin"] <= fmt(q[n - 5]))]  # 4분기 정답이 모두 있는 origin
    cov = float(((sc["actual"] >= sc["p5"]) & (sc["actual"] <= sc["p95"])).mean() * 100)
    ok &= abs(cov - d2["coverage"]) < 1e-6                     # 딕셔너리 1의 5~95% 구간으로 다시 센 커버리지가 딕셔너리 2와 같다
    ok &= all(np.isfinite(v) for v in d2.values()) and d2["mse"] > 0 and d2["crps"] > 0 and 0 <= d2["coverage"] <= 100
    record("정상", name, bool(ok), f"origin {len(keys)}개, 채점 origin {len(sc) // 4}개, coverage {d2['coverage']:.1f}%")
    return ok


def expect_error(name, fn, exc, text=None):
    try:
        fn()
    except exc as e:
        msg = str(e).split(chr(10))[0]
        record("오류", name, text is None or text in str(e), msg[:110])
    except Exception as e:                                     # 예상과 다른 종류의 오류
        record("오류", name, False, f"{type(e).__name__}: {str(e)[:100]}")
    else:
        record("오류", name, False, "오류가 나야 하는데 그냥 실행됨")


def base_frame(col="X2"):
    return pd.read_csv(HERE / "data" / f"{col}.csv", encoding="utf-8-sig", dtype={"Data": str})


def write(tmp, name, df):
    p = Path(tmp) / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


def quick():
    base = base_frame()                                        # X2 60분기 (2011Q1 ~ 2025Q4), 가장 이른 날짜 2016Q4, 가장 늦은 날짜 2024Q4
    with tempfile.TemporaryDirectory() as tmp:
        ok_csv = write(tmp, "ok", base)
        # ---- 오류가 나야 하는 경우 (계산 전에 걸려야 하므로 빠르다)
        expect_error("날짜가 너무 이름 (가장 이른 분기 직전)", lambda: PI.run_pi(ok_csv, "2016-09-01"), ValueError, "너무 이르다")
        expect_error("날짜가 너무 늦음 (가장 늦은 분기 다음)", lambda: PI.run_pi(ok_csv, "2025-01-01"), ValueError, "정답이 4분기")
        expect_error("날짜가 데이터 끝 이후", lambda: PI.run_pi(ok_csv, "2030-01-01"), ValueError, "없다")
        expect_error("날짜가 데이터 시작 이전", lambda: PI.run_pi(ok_csv, "2000-01-01"), ValueError, "없다")
        expect_error("날짜가 날짜 형식이 아님", lambda: PI.run_pi(ok_csv, "abc"), ValueError, "날짜를 읽지 못했다")
        expect_error("CSV가 없음", lambda: PI.run_pi("없는파일", "2019-09-01"), FileNotFoundError, "찾지 못했다")
        expect_error("Data 열이 없음", lambda: PI.run_pi(write(tmp, "nodata", base.rename(columns={"Data": "값"})), "2019-09-01"), ValueError, "두 열")
        expect_error("Date 열이 없음", lambda: PI.run_pi(write(tmp, "nodate", base.rename(columns={"Date": "날짜"})), "2019-09-01"), ValueError, "두 열")
        expect_error("분기가 빠짐", lambda: PI.run_pi(write(tmp, "gap", base.drop(index=30)), "2019-09-01"), ValueError, "연속된 분기")
        expect_error("분기가 중복됨", lambda: PI.run_pi(write(tmp, "dup", pd.concat([base, base.iloc[[30]]]).sort_values("Date")), "2019-09-01"), ValueError, "연속된 분기")
        nan = base.copy(); nan.loc[30, "Data"] = np.nan
        expect_error("값이 비어 있음(중간)", lambda: PI.run_pi(write(tmp, "nan", nan), "2019-09-01"), ValueError, "가운데에 빈 값")
        txt = base.copy(); txt.loc[30, "Data"] = "abc"
        expect_error("값이 숫자가 아님", lambda: PI.run_pi(write(tmp, "txt", txt), "2019-09-01"), ValueError, "숫자가 아닌 값")
        expect_error("데이터가 너무 짧음(20분기)", lambda: PI.run_pi(write(tmp, "short", base.iloc[:20]), "2015-12-01"), ValueError, "너무 짧다")
        const = base.copy(); const["Data"] = "1.0"
        expect_error("값이 모두 같음(상수)", lambda: PI.run_pi(write(tmp, "const", const), "2019-09-01"), ValueError, "모두 같아서")
        expect_error("날짜가 내림차순으로 정렬됨", lambda: PI.run_pi(write(tmp, "desc", base.iloc[::-1]), "2019-09-01"), ValueError, "오름차순")
        # ---- 명령줄
        py = [sys.executable, str(HERE / "PI.py")]
        r = subprocess.run(py + ["하나만"], capture_output=True, text=True, encoding="utf-8", cwd=HERE, env=ENV)
        record("명령줄", "인자 개수가 틀림", r.returncode != 0 and "사용법" in (r.stdout + r.stderr), (r.stdout + r.stderr).strip().split(chr(10))[-1][:100])
        r = subprocess.run(py + ["없는파일", "2019-09-01"], capture_output=True, text=True, encoding="utf-8", cwd=HERE, env=ENV)
        record("명령줄", "CSV가 없음", r.returncode != 0 and "찾지 못했다" in r.stderr, r.stderr.strip().split(chr(10))[-1][:100])
    # ---- 정상: 입력 표기 변형 (캐시된 X1라 빠르다). 모두 같은 결과여야 한다
    ref = PI.run_pi("X1_l1", "2019-12-01", name="X1_l1", verbose=False)
    check_outputs("X1 2019-12-01 (일관성)", "2019-12-01", "X1_l1", *ref)
    for label, src, date in [("날짜: 같은 분기의 다른 날 2019-10-15", "X1_l1", "2019-10-15"), ("날짜: 분기 끝 2019-12-31", "X1_l1", "2019-12-31"),
                             ("날짜: 2019Q4 표기", "X1_l1", "2019Q4"), ("날짜: 2019/12/01", "X1_l1", "2019/12/01"), ("날짜: 20191201", "X1_l1", "20191201"),
                             ("CSV: .csv 붙임", "X1_l1.csv", "2019-12-01"), ("CSV: data/ 경로", "data/X1_l1.csv", "2019-12-01"),
                             ("CSV: 절대 경로", str(HERE / "data" / "X1_l1.csv"), "2019-12-01")]:
        try:
            d1, d2 = PI.run_pi(src, date, name="X1_l1", verbose=False)
            same = list(d1) == list(ref[0]) and all(np.array_equal(d1[k], ref[0][k]) for k in d1) and d2 == ref[1]
            record("정상", label, same, "결과가 기준과 같음" if same else "결과가 다름")
        except Exception as e:
            record("정상", label, False, f"{type(e).__name__}: {str(e)[:100]}")
    inv = pd.read_csv(HERE / "data" / "X1_l1.csv", encoding="utf-8-sig", dtype={"Data": str})
    pad = pd.concat([pd.DataFrame({"Date": ["2010-12-01", "2010-09-01"][::-1], "Data": [None, None]}), inv, pd.DataFrame({"Date": ["2026-03-01", "2026-06-01"], "Data": [None, ""]})], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            d1, d2 = PI.run_pi(write(tmp, "pad", pad), "2019-12-01", name="X1_l1", verbose=False)
            same = list(d1) == list(ref[0]) and all(np.array_equal(d1[k], ref[0][k]) for k in d1) and d2 == ref[1]
            record("정상", "앞뒤에 빈 값이 있는 행(잘라 냄)", same, "결과가 기준과 같음" if same else "결과가 다름")
        except Exception as e:
            record("정상", "앞뒤에 빈 값이 있는 행(잘라 냄)", False, f"{type(e).__name__}: {str(e)[:100]}")
    r = subprocess.run(py + ["'X1_l1'", "'2019-12-01'"], capture_output=True, text=True, encoding="utf-8", cwd=HERE, env=ENV)
    record("명령줄", "작은따옴표를 붙인 인자", r.returncode == 0, "정상 실행" if r.returncode == 0 else r.stderr.strip().split(chr(10))[-1][:100])


def full():
    """변수별 가장 이른/늦은/중간 날짜. 날짜마다 보정 구간이 달라 새로 계산한다(변수당 몇 분)."""
    cases = [("X2", "earliest"), ("X2", "latest"), ("X6_l1", "earliest"), ("X6_l1", "latest"),
             ("X1_l1", "mid"), ("X3_l1", "mid"), ("X4", "mid"), ("X5_l1", "mid")]
    for col, kind in cases:
        q, y = PI.read_series(col)
        i = {"earliest": S.MIN_TRAIN + PI.MIN_CALIB - 1, "latest": len(y) - 5, "mid": 29}[kind]     # mid: 훈련 30분기(보정 10개)
        date = pd.Period(q[i], "Q").end_time.strftime("%Y-%m-01")
        try:
            d1, d2 = PI.run_pi(col, date, name=col, verbose=False)
            check_outputs(f"{col} {kind} ({date})", date, col, d1, d2)
        except Exception as e:
            record("정상", f"{col} {kind} ({date})", False, f"{type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    quick()
    if "full" in sys.argv[1:]:
        full()
    bad = [r for r in RESULTS if not r[2]]
    print(f"{chr(10)}총 {len(RESULTS)}개 중 통과 {len(RESULTS) - len(bad)}개, 실패 {len(bad)}개")
    for r in bad:
        print("  실패:", r[0], r[1], "->", r[3])
    sys.exit(1 if bad else 0)
