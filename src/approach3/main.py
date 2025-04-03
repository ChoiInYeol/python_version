"""
main.py
모델 실행 파일 - 모델 타입 선택 및 오늘 날짜의 Nowcast 출력
"""
from dfm_model import DFMModel
from midas_model import MIDASModel
import pandas as pd
import logging # 로깅 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 데이터 이름 선택: 'full', '9X'
    data_name = 'full'
    
    # 모델 타입 선택: 'elasticnet', 'xgboost', 'lightgbm', 'midas'
    model_type = 'elasticnet'  # 원하는 모델로 변경 가능
    
    # 공통 파라미터 설정
    x_path = f'data/processed/X_{data_name}.csv'
    y_path = f'data/processed/y_{data_name}.csv'
    target_col = 'CPI_YOY'
    
    # 모델별 파라미터 설정
    model_params = {
        'X_path': x_path,
        'y_path': y_path,
        'target': target_col,
        'data_name': data_name
    }
    
    # DFM 모델 특정 파라미터
    if model_type in ['elasticnet', 'xgboost', 'lightgbm']:
        dfm_params = {
            'model_type': model_type,
            'train_window_size': 365 * 10,
            'n_factors': 2,
            'xgb_params': {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]},
            'lgb_params': {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]}
        }
        model_params.update(dfm_params)
        model = DFMModel(**model_params)
        
    # MIDAS 모델 특정 파라미터
    elif model_type == 'midas':
        midas_params = {
            'lookback_periods': 12,
            'poly_degree': 2,
            'max_lags': {'default': 30}, 
            'train_window_size': 730
        }
        model_params.update(midas_params)
        model = MIDASModel(**model_params)
    else:
        logger.error(f"지원되지 않는 모델 타입: {model_type}")
        return

    try:
        # 모델 학습
        logger.info(f"모델 학습 시작: {model_type}")
        model.fit()
        logger.info(f"모델 학습 완료: {model_type}")
        
        # 결과 시각화 및 CSV 저장
        logger.info("결과 시각화 및 CSV 저장 시작")
        model.plot_results() 
        logger.info("결과 시각화 및 CSV 저장 완료")
        
        # 특성 중요도 저장 (구현된 경우)
        try:
            model.export_feature_importance()
        except NotImplementedError:
            logger.info("특성 중요도 분석은 이 모델 타입에서 지원되지 않습니다.")
        except Exception as e:
             logger.warning(f"특성 중요도 분석 중 오류 발생: {e}")

        # 최종 월간 예측치 출력
        logger.info("최종 월간 예측치 계산 및 출력")
        monthly_nowcast = model.get_latest_monthly_nowcast()
        
        if monthly_nowcast is not None:
            final_nowcast_pct = round(monthly_nowcast, 2)
            logger.info(f"\nPredicted Monthly CPI YoY ({model.model_type.capitalize()}): {final_nowcast_pct:.2f}%")
        else:
            logger.warning("\n월간 예측치를 계산할 수 없습니다.")
            
    except Exception as e:
        logger.error(f"프로세스 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()