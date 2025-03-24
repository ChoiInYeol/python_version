"""
실험 실행 관련 함수 정의
"""
import logging
from models import CPIPredictor
from data_processing import prepare_raw_data
from visualization import plot_experiment_results
from constants import EXPERIMENTS, ETF_EXPERIMENTS

logger = logging.getLogger(__name__)

def run_experiments(company_list=None, data_path='Daily물가지수.xlsx', use_etf=False):
    """
    실험 실행 및 결과 시각화
    
    Args:
        company_list: 사용할 기업 리스트
        data_path: 주가 데이터 파일 경로
        use_etf: ETF 섹터 데이터 사용 여부
        
    Returns:
        tuple: (figure, axes, results)
    """
    results = {}
    experiments = ETF_EXPERIMENTS if use_etf else EXPERIMENTS
    
    # 실험 수행
    for exp in experiments:
        logger.info(f"\n실행 중: {exp['name']}")
        
        # 각 실험별로 데이터 준비 (수익률 주기 반영)
        input_df, target_df, true_df = prepare_raw_data(
            company_list=company_list,
            data_path=data_path,
            return_period=exp['return_period'],
            use_etf=use_etf
        )
        
        predictor = CPIPredictor(n_pca_comp=exp['n_pca_comp'], lags=exp['lags'])
        data = predictor.prepare_data(input_df, target_df, lag_days=exp['lag_days'])
        
        if predictor.train_model():
            results[exp['name']] = {
                'predictor': predictor,
                'results': predictor.predict(),
                'eval': predictor.evaluate()
            }

    # 결과 시각화
    fig, axes = plot_experiment_results(results, experiments, true_df)
    
    # 성능 비교 출력
    logger.info("\n실험별 성능 비교:")
    for exp_name, exp_data in results.items():
        logger.info(f"\n{exp_name}:")
        for metric, value in exp_data['eval'].items():
            logger.info(f"  {metric}: {value:.6f}")
    
    return fig, axes, results 