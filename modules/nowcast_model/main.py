"""
CPI Nowcasting 메인 실행 파일

이 파일은 CPI Nowcasting 파이프라인을 실행하는 메인 진입점입니다.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from modules.nowcast_model.utils import load_definitions, set_backtest_dates
from modules.nowcast_model.data_collector import collect_release_data, collect_historical_data
from modules.nowcast_model.data_processor import process_data
from modules.nowcast_model.model import run_state_space_model, generate_forecasts
from modules.nowcast_model.visualization import visualize_results

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

def run_pipeline():
    """전체 파이프라인 실행"""
    try:
        # 1. 변수 정의 및 릴리스 정보 로드
        variable_params, releases_info = load_definitions()
        
        # 2. 백테스트 날짜 설정
        bdates = set_backtest_dates()
        
        # 3. 릴리스 데이터 수집
        releases = collect_release_data(variable_params, releases_info)
        
        # 4. 역사적 데이터 수집
        hist = collect_historical_data(variable_params)
        
        # 5. 데이터 처리
        hist = process_data(hist, variable_params)
        
        # 6. 상태 공간 모델 실행
        models = run_state_space_model(hist, bdates)
        
        # 7. 예측 생성
        forecasts = generate_forecasts(models, hist, bdates)
        
        # 8. 결과 시각화
        visualize_results(forecasts)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def main():
    """메인 실행 함수"""
    try:
        # 환경 변수 확인
        if not os.getenv('EF_DIR'):
            raise ValueError("EF_DIR environment variable not set. Please set it to the root directory of the project.")
        
        if not os.getenv('FRED_API_KEY'):
            raise ValueError("FRED_API_KEY environment variable not set. Please set it to your FRED API key.")
        
        # 파이프라인 실행
        run_pipeline()
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 