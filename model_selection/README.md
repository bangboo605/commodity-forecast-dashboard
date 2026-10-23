# model_selection

주요변수마다 모델과 (d, D) 구조를 선정하고, 선정 모델의 예측으로 스트림릿 시나리오 CSV를 만든다(12시리즈 방식). 앱은 이 모듈을 import하지 않고 CSV만 읽는다.

## 파일

| 파일 | 역할 |
|---|---|
| `ts_models.py` | AR단독, GARCH, SV 모델 구현 |
| `selection.py` | 변수 설정(`VariableConfig`), 모든 조합 계산(`run_grid`), 선정(`select`), 선정 모델의 예측(`forecast_table`) |
| `variables.py` | 금 주요변수 6개의 설정. 새 변수는 한 줄 추가 |
| `build_scenarios.py` | 앱용 시나리오 CSV 만들기 |
| `checks/test_X2.py`, `test_build.py`, `test_incremental.py` | 12시리즈 결과 재현, CSV 형식 확인, 증분 계산 확인 |
| `cache/` | 계산 결과 캐시(git 제외) |

## 사용

```
python -m model_selection.build_scenarios        # 6개 변수 선정 + data/gold/gold_major_variables_forecast_model.csv 생성
python -m model_selection.checks.test_X2          # 테스트
```

```python
from model_selection.variables import GOLD
from model_selection.selection import run_grid, select

s = select(run_grid(GOLD["X6_l1"]))        # s["선정"], s["후보"]
```

## 방식

- 조합: 모델 3종 x (d, D) 4가지 x 스케일(원, 로그) x 구간(자체, 리샘플링)
- 미래 정보 없음: 적합, 오차 풀, 단위 상수(훈련 구간 표준편차)가 모두 각 origin 이전 데이터만 쓴다
- 리샘플링 난수는 `SEED = 42` 하나로 고정(모든 변수, 모델, (d, D)가 같은 난수로 비교). 자체 구간, 점예측, MSE는 난수를 쓰지 않는다
- 선정: CRPS 1등의 +-10% 이내 중 커버리지가 90%에 가장 가까운 조합
- CSV: low=5%, base=점예측, high=95%. 변수마다 데이터가 있는 구간까지만 있다

## 한계

새 데이터가 들어오면 바뀐 origin 뒤쪽만 증분 계산한다(origin별 난수 상태와 오차 풀을 캐시에 저장). 캐시 파일 이름에 시드와 `CACHE_VERSION`이 들어 있어, 계산 코드를 바꾸면 `CACHE_VERSION`을 올려야 한다. horizon은 h=1~4 고정이고 발표 시차와 데이터 수정은 반영하지 않는다.
