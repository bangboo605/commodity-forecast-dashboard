# 파일 구분

## 스트림릿 앱이 쓰는 파일

앱(`item_selection_v3.8.py`)은 `model_selection/`의 파일을 하나도 쓰지 않는다. 이 폴더가 만든 CSV만 읽는다.

| 파일 | 역할 |
|---|---|
| `item_selection_v3.8.py` | 앱 본체 |
| `config.py`, `gold_price_model.py` | 앱 설정, 금 가격 TVP 모델 |
| `data/gold/gold_major_variables_forecast_model.csv` | **이 폴더가 만드는 파일.** 금 주요변수 시나리오(low, base, high) |
| `data/gold/gold_major_variables.csv`, `data/gold/gold_price.csv` | 금 주요변수 실제값, 금 가격 (이 폴더도 입력으로 읽음) |

## `model_selection/` 구성

| 위치 | 내용 |
|---|---|
| `selection.py`, `ts_models.py`, `variables.py`, `build_scenarios.py`, `__init__.py` | 모듈 본체(CSV를 만드는 코드). 앱과 같이 옮겨 쓸 코드 |
| `checks/` | 검증과 보조 스크립트: `test_X2.py`, `test_build.py`, `test_incremental.py`, `seed_check.py`. 실행은 `python -m model_selection.checks.<이름>` |
| `cache/` | 계산 결과 캐시 (git 제외) |
| `README.md`, `FILES.md` | 설명 |
| `__pycache__/` | 파이썬 자동 생성 |
