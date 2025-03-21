"""
main.py
모델 실행 파일
"""
from dfm_model_gpt import DFMModel
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_today_nowcast():
    """오늘의 나우캐스팅 값을 계산하고 출력합니다."""
    model = DFMModel(
        X_path='data/processed/X_processed.csv',
        y_path='data/processed/y_processed.csv',
        target='CPI_YOY',
        train_window_size=365 * 5,
        forecast_horizon=40,
        n_factors=2,
        model_path='src/models/dfm_model_gpt.joblib'
    )
    
    # 모델 학습
    model.fit()
    
    # 오늘 날짜의 예측값 계산
    today = pd.Timestamp.today()
    today_period = today.to_period('D')
    
    # 예측 수행
    nowcast = pd.DataFrame(index=model.X.index)
    nowcast['predicted'] = model.predict(model.X)
    
    # 오늘의 예측값 출력
    today_pred = nowcast.loc[today_period, 'predicted']
    logger.info(f"오늘({today.strftime('%Y-%m-%d')})의 CPI 예측값: {today_pred:.2f}%")
    
    # 결과 저장
    model.export_nowcast_csv('output/nowcasts.csv')
    model.plot_results('output')
    model.export_feature_importance('output')
    
    return today_pred

if __name__ == "__main__":
    get_today_nowcast()
