# Streamlit Chart

원자재 주요 변수와 가격 전망 결과를 시각화하는 Streamlit 대시보드입니다.

## 실행

```powershell
streamlit run item_selection_v3.8.py
```

## 프로젝트 구조

- `item_selection_v3.8.py`: 현재 개발 중인 실행 파일
- `gold_price_model.py`: 금 가격 TVP 회귀 모델 (final 값 기반 실시간 예측)
- `config.py`: 데이터와 프로젝트 자료의 중앙 경로 설정
- `data/`: 품목별 앱 데이터와 이전 시나리오 데이터
- `archive/versions/`: 이전 Streamlit 버전
- `notebooks/`: 분석 및 모델링 노트북
- `research/source_data/`: 모델링 원본 데이터
- `research/model_outputs/`: 모델링 결과 파일과 이미지
- `research/presentations/`: 발표 자료
- `VERSION_HISTORY.txt`: 버전별 변경 결과

## 의존성

`pyproject.toml`과 `uv.lock`을 기준으로 환경을 재현합니다.
