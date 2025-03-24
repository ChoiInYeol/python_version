"""
DFM 모델 실행 메인 스크립트
"""
import pandas as pd
import logging
import os
from datetime import datetime
from model import DFMModel
from visualizer import plot_forecast, plot_factor_contribution

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """
    필요한 디렉토리 생성
    """
    directories = ['output', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"디렉토리 생성/확인: {directory}")

def main():
    """
    메인 실행 함수
    """
    try:
        # 디렉토리 설정
        setup_directories()
        
        # 데이터 설정
        data_name = '9X'
        model_path = f'models/dfm_model_{data_name}.joblib'
        
        # 모델 초기화
        model = DFMModel(
            X_path=f'data/processed/X_{data_name}.csv',
            y_path=f'data/processed/y_{data_name}.csv',
            target='CPI_YOY',
            model_path=model_path,
            forecast_horizon=30
        )
        
        # 모델 적합
        logger.info("모델 적합 시작")
        model.fit()
        
        # Nowcast 저장
        nowcast_df = pd.DataFrame(index=model.X.index)
        nowcast_df['smoothed_factor_sum'] = model.Z_smoothed.sum(axis=1)
        nowcast_df['actual'] = model.y[model.target]
        
        output_path = f'output/nowcasts_{data_name}.csv'
        nowcast_df.to_csv(output_path)
        logger.info(f"Nowcast 결과 저장: {output_path}")
        
        # Forecast 및 신뢰구간 계산
        steps = 30
        forecast_df = model.forecast_target(steps)
        
        # Forecast 결과 저장
        forecast_path = f'output/forecast_{data_name}.csv'
        forecast_df.to_csv(forecast_path)
        logger.info(f"Forecast 결과 저장: {forecast_path}")
        
        # 시각화
        # 1. 예측 결과와 요인
        fig, axes = plot_forecast(forecast_df, model.y[model.target], model.Z_smoothed)
        fig.savefig(f'output/forecast_plot_{data_name}.png', dpi=150, bbox_inches='tight')
        logger.info(f"Forecast 플롯 저장: output/forecast_plot_{data_name}.png")
        
        # 2. 요인 기여도
        factor_contribution_fig = plot_factor_contribution(model.Z_smoothed, model.pca_model)
        factor_contribution_fig.savefig(f'output/factor_contribution_{data_name}.png', 
                                      dpi=150, bbox_inches='tight')
        logger.info(f"요인 기여도 플롯 저장: output/factor_contribution_{data_name}.png")
        
        # 모델 저장
        model.save_model(model_path)
        logger.info(f"모델 저장 완료: {model_path}")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
