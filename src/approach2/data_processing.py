"""
데이터 처리 관련 함수 정의
"""
import pandas as pd
import logging
from constants import DEFAULT_COMPANIES, SECTOR_ETF

logger = logging.getLogger(__name__)

def prepare_raw_data(company_list=None, data_path='Daily물가지수.xlsx', return_period=5, use_etf=False):
    """
    데이터 로드 및 전처리
    
    Args:
        company_list: 사용할 기업 리스트. None이면 기본 리스트 사용
        data_path: 주가 데이터 파일 경로
        return_period: 수익률 계산 기간 (일)
        use_etf: ETF 섹터 데이터 사용 여부
        
    Returns:
        tuple: (input_df, target_df, true_df)
    """
    # CPI 데이터 로드
    data = pd.read_csv('processed/Approach11.csv', index_col=0, parse_dates=True)
    data.dropna(inplace=True)
    
    if use_etf:
        # ETF 데이터 추출
        ret = data[SECTOR_ETF]
    else:
        # 기업 리스트 설정
        if company_list is None:
            company_list = [company for sector in DEFAULT_COMPANIES.values() 
                          for company in sector]

        # 주가 데이터 로드
        price_raw = pd.read_excel(data_path, sheet_name='p', skiprows=13, index_col=0).iloc[6:]
        price_raw.index = pd.to_datetime(price_raw.index)

        # 주가 수익률 계산 (return_period 반영)
        cp_list = [i+'-US' for i in company_list]
        ret = price_raw.loc[:, cp_list].pct_change(return_period, fill_method=None)
        ret.dropna(inplace=True)
        ret.index = pd.to_datetime(ret.index)

    # CPI 데이터 준비
    true_df = data.loc[:, ['Release', 'CPI_YOY']]
    true_df = true_df.loc[true_df.index.isin(ret.index)]
    
    # 타겟 변수 준비
    target_df = data.loc[:, '10MA']
    target_df.name = 'CPId'

    # 데이터 병합
    input_df = ret.join(target_df, how='inner')
    
    return input_df, target_df, true_df

def save_experiment_results(results, save_path='experiment_results.csv'):
    """
    실험 결과를 DataFrame으로 변환하고 CSV로 저장
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장할 CSV 파일 경로
        
    Returns:
        DataFrame: 저장된 결과 데이터
    """
    # 각 실험의 결과를 저장할 리스트
    data_list = []
    
    for exp_name, exp_data in results.items():
        # 예측 결과 데이터 가져오기
        pred_df = exp_data['results']
        
        # 실험 이름 컬럼 추가
        pred_df = pred_df.copy()
        pred_df['Experiment'] = exp_name
        
        # 평가 지표 추가
        for metric, value in exp_data['eval'].items():
            pred_df[metric] = value
            
        data_list.append(pred_df)
    
    # 모든 실험 결과를 하나의 DataFrame으로 통합
    combined_df = pd.concat(data_list, axis=0)
    
    # 인덱스를 컬럼으로 변환
    combined_df = combined_df.reset_index()
    
    # CSV로 저장
    combined_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    logger.info(f"실험 결과가 {save_path}에 저장되었습니다.")
    
    return combined_df 