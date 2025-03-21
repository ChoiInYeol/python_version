# CPI Nowcasting with Dynamic Factor Model

## 프로젝트 개요
이 프로젝트는 동적 요인 모델(Dynamic Factor Model, DFM)을 사용하여 실시간 CPI(소비자물가지수) 예측을 수행합니다.

### 주요 특징
- 실시간 CPI 예측 (Nowcasting)
- 동적 요인 모델 기반
- ElasticNet을 사용한 요인 추출
- 실시간 성능 모니터링 (RMSE)

## 모델 구조
1. **데이터 처리**
   - 일별 데이터 로드 및 전처리
   - 결측치 처리 및 이상치 제거
   - 시계열 데이터 정규화

2. **요인 추출**
   - ElasticNet을 사용한 요인 추출
   - Grid Search를 통한 하이퍼파라미터 최적화
   - 다중 요인 모델링 지원

3. **예측 및 평가**
   - 실시간 예측 수행
   - Rolling RMSE를 통한 성능 모니터링
   - 시각화 및 결과 저장

## 사용 방법
```python
from dfm_model_gpt import DFMModel

model = DFMModel(
    X_path='data/processed/X_processed.csv',
    y_path='data/processed/y_processed.csv',
    target='CPI_YOY',
    train_window_size=365 * 10,  # 10년
    forecast_horizon=30,          # 30일
    n_factors=2                  # 2개 요인
)

# 모델 학습
model.fit()

# 예측 결과 저장 및 시각화
model.export_nowcast_csv('output/nowcasts.csv')
model.plot_results('output')
model.export_feature_importance('output')
```

## 주요 파라미터
- `train_window_size`: 학습 기간 (일)
- `forecast_horizon`: 예측 수평선 (일)
- `n_factors`: 추출할 요인 수
- `l1_ratio_range`: ElasticNet L1 비율 범위
- `alpha_range`: ElasticNet 알파 범위

## 출력 파일
- `nowcasts.csv`: 예측 결과
- `nowcast_plot.png/svg`: 예측 결과 시각화
- `feature_importance.csv`: 변수 중요도
- `feature_importance.png/svg`: 변수 중요도 시각화

## 개선 방향
1. 예측 안정성 향상
   - 이상치 처리 강화
   - 요인 선택 로직 개선
   - 예측값 스무딩 적용

2. 성능 최적화
   - 병렬 처리 도입
   - 메모리 사용량 최적화
   - 계산 효율성 개선

3. 모니터링 강화
   - 실시간 알림 기능
   - 자동 재학습 기능
   - 예측 신뢰도 평가 