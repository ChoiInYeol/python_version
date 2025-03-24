"""
메인 실행 파일
"""
import logging
import matplotlib.pyplot as plt
from experiments import run_experiments
from data_processing import save_experiment_results

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_company_experiments():
    """
    개별 기업 수익률을 사용한 실험 실행
    """
    try:
        # 실험 실행
        fig, axes, results = run_experiments(use_etf=False)
        
        # 결과 저장
        plt.savefig('Approach2_company.png')
        plt.close()
        
        # 실험 결과 저장
        results_df = save_experiment_results(results, 'experiment_results_company.csv')
        
        # 저장된 데이터 확인
        logger.info("\n저장된 데이터 미리보기 (개별 기업):")
        logger.info(results_df.head())
        
        # 실험별 통계 확인
        logger.info("\n실험별 평균 MAE (개별 기업):")
        logger.info(results_df.groupby('Experiment')['Model MAE'].mean())
        
    except Exception as e:
        logger.error(f"개별 기업 실험 실행 중 오류 발생: {str(e)}")
        raise

def run_etf_experiments():
    """
    ETF 섹터 수익률을 사용한 실험 실행
    """
    try:
        # 실험 실행
        fig, axes, results = run_experiments(use_etf=True)
        
        # 결과 저장
        plt.savefig('Approach2_etf.png')
        plt.close()
        
        # 실험 결과 저장
        results_df = save_experiment_results(results, 'experiment_results_etf.csv')
        
        # 저장된 데이터 확인
        logger.info("\n저장된 데이터 미리보기 (ETF):")
        logger.info(results_df.head())
        
        # 실험별 통계 확인
        logger.info("\n실험별 평균 MAE (ETF):")
        logger.info(results_df.groupby('Experiment')['Model MAE'].mean())
        
    except Exception as e:
        logger.error(f"ETF 실험 실행 중 오류 발생: {str(e)}")
        raise

def main():
    """
    메인 실행 함수
    """
    try:
        # 개별 기업 실험 실행
        logger.info("\n=== 개별 기업 실험 시작 ===")
        run_company_experiments()
        
        # ETF 실험 실행
        logger.info("\n=== ETF 실험 시작 ===")
        run_etf_experiments()
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 