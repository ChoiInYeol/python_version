# CPI Nowcasting System

이 프로젝트는 미국 소비자물가지수(CPI)의 일별 nowcasting을 위한 다중 모델 시스템을 구현합니다.

## 주요 기능

- 다중 빈도 데이터 처리 (일별, 주별, 월별 데이터)
- 세 가지 주요 모델 구현:
  - MIDAS (Mixed Data Sampling) regression
  - Dynamic Factor Model (DFM)
  - Mixed-frequency Random Forest (MO-RFRN)
- Walk-forward nowcasting 평가
- 인터랙티브 시각화

## 프로젝트 구조

```
.
├── data/
│   └── final_df.csv          # 원본 데이터
├── src/
│   ├── preprocessing.py      # 데이터 전처리 모듈
│   ├── models.py            # 모델 구현
│   └── main.py              # 메인 실행 파일
├── notebook/                 # Jupyter 노트북
├── requirements.txt         # 의존성 패키지
└── README.md               # 프로젝트 문서
```

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 데이터 준비:
   - `data/final_df.csv` 파일에 필요한 데이터를 준비합니다.
   - 데이터는 날짜 인덱스와 필요한 모든 변수들을 포함해야 합니다.

2. 실행:
```bash
python src/main.py
```

## 모델 설명

### 1. MIDAS (Mixed Data Sampling) Regression
- 고빈도와 저빈도 데이터를 함께 사용하는 혼합 데이터 샘플링 모델
- 가중치를 통한 고빈도 데이터 집계
- 최적화를 통한 가중치 학습

### 2. Dynamic Factor Model (DFM)
- Kalman Filter 기반의 동적 요인 모델
- PCA를 통한 초기 요인 추출
- 시계열 구조를 고려한 예측

### 3. Mixed-frequency Random Forest (MO-RFRN)
- 결측치를 처리할 수 있는 랜덤 포레스트 모델
- 노드별 분할을 통한 혼합 빈도 데이터 처리
- 스케일링을 통한 특성 정규화

## 성능 평가

- RMSE (Root Mean Square Error)를 통한 모델 성능 평가
- Walk-forward nowcasting을 통한 실시간 예측 성능 측정
- 인터랙티브 시각화를 통한 결과 분석

## 라이선스

MIT License 