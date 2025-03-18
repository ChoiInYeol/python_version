"""
유틸리티 모듈

이 모듈은 변수 정의 로드, 릴리스 정보 로드, 백테스트 날짜 설정 등 유틸리티 기능을 제공합니다.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

def load_definitions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    변수 정의와 릴리스 정보를 로드합니다.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (변수 정의 데이터프레임, 릴리스 정보 데이터프레임)
    """
    try:
        # 변수 정의 파일 경로 설정
        def_path = Path(os.getenv('EF_DIR')) / 'data' / 'definitions.xlsx'
        
        # 각 시트에서 데이터 로드
        variable_params = pd.read_excel(def_path, sheet_name='variables')
        releases_info = pd.read_excel(def_path, sheet_name='releases')
        
        # 변수 정의에 필요한 컬럼이 있는지 확인
        required_var_columns = ['varname', 'fullname', 'release', 'hist_source', 'hist_source_key', 
                             'hist_source_freq', 'st', 'd1', 'd2', 'nc_dfm_input', 'nc_method']
        missing_var_columns = [col for col in required_var_columns if col not in variable_params.columns]
        
        if missing_var_columns:
            raise ValueError(f"Missing required columns in variable definitions: {missing_var_columns}")
        
        # 릴리스 정보에 필요한 컬럼이 있는지 확인
        required_rel_columns = ['id', 'fullname', 'source', 'source_key']
        missing_rel_columns = [col for col in required_rel_columns if col not in releases_info.columns]
        
        if missing_rel_columns:
            raise ValueError(f"Missing required columns in release information: {missing_rel_columns}")
        
        logger.info(f"Loaded {len(variable_params)} variable definitions and {len(releases_info)} release information entries")
        
        return variable_params, releases_info
        
    except Exception as e:
        logger.error(f"Failed to load definitions: {str(e)}")
        raise

def set_backtest_dates() -> List[datetime]:
    """
    백테스트 날짜를 설정합니다.
    
    Returns:
        List[datetime]: 백테스트 날짜 목록
    """
    try:
        today = datetime.now()
        contiguous = pd.date_range(today - timedelta(days=90), today, freq='D')
        
        # 2021년부터의 랜덤 날짜 생성
        old_dates = []
        start_date = datetime(2021, 1, 1)
        end_date = (min(contiguous) - timedelta(days=1)).replace(day=1)
        
        current_date = start_date
        while current_date <= end_date:
            # 각 달에서 랜덤한 날짜 선택
            next_month = current_date + pd.DateOffset(months=1)
            last_day_of_next_month = (pd.Period(next_month, freq='M') + 1).start_time - pd.Timedelta(days=1)
            random_day = np.random.randint(1, last_day_of_next_month.day + 1)
            old_dates.append(current_date.replace(day=random_day))
            current_date += pd.DateOffset(months=1)
            
        bdates = sorted(old_dates + list(contiguous))
        
        logger.info(f"Set {len(bdates)} backtest dates")
        
        return bdates
        
    except Exception as e:
        logger.error(f"Failed to set backtest dates: {str(e)}")
        raise 