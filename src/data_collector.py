"""
데이터 수집 모듈

이 모듈은 FRED API 및 Yahoo Finance에서 데이터를 수집하는 기능을 제공합니다.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import concurrent.futures

import pandas as pd
import numpy as np
import requests
from pandas import DataFrame

# 로깅 설정
logger = logging.getLogger(__name__)

# 상수 정의
IMPORT_DATE_START = '2007-01-01'  # spdw, usd, metals, moo start in Q1-Q2 2007

def get_fred_release_dates(release_id: str, api_key: str) -> pd.DataFrame:
    """
    FRED API를 통해 특정 릴리스의 날짜 데이터를 가져옵니다.
    
    Args:
        release_id (str): FRED 릴리스 ID
        api_key (str): FRED API 키
        
    Returns:
        pd.DataFrame: 릴리스 날짜 데이터
    """
    url = f"https://api.stlouisfed.org/fred/release/dates"
    params = {
        'release_id': release_id,
        'realtime_start': '2010-01-01',
        'include_release_dates_with_no_data': 'true',
        'api_key': api_key,
        'file_type': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 릴리스 날짜 데이터프레임 생성
        release_dates = pd.DataFrame(data['release_dates'])
        release_dates['date'] = pd.to_datetime(release_dates['date'])
        
        return release_dates
        
    except Exception as e:
        logger.error(f"Failed to get FRED release dates for {release_id}: {str(e)}")
        raise

def get_fred_series(series_id: str, api_key: str, freq: str) -> pd.DataFrame:
    """
    FRED API를 통해 시계열 데이터를 가져옵니다.
    
    Args:
        series_id (str): FRED 시리즈 ID
        api_key (str): FRED API 키
        freq (str): 데이터 주기 ('d', 'w', 'm', 'q')
        
    Returns:
        pd.DataFrame: 시계열 데이터
    """
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': IMPORT_DATE_START
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 시계열 데이터프레임 생성
        series_data = pd.DataFrame(data['observations'])
        series_data['date'] = pd.to_datetime(series_data['date'])
        series_data['value'] = pd.to_numeric(series_data['value'], errors='coerce')
        
        # 주기에 따라 데이터 리샘플링
        if freq == 'd':
            resampled = series_data.set_index('date')['value'].resample('D').ffill()
        elif freq == 'w':
            resampled = series_data.set_index('date')['value'].resample('WE').ffill()
        elif freq == 'm':
            resampled = series_data.set_index('date')['value'].resample('ME').ffill()
        elif freq == 'q':
            resampled = series_data.set_index('date')['value'].resample('QE').ffill()
        else:
            raise ValueError(f"Invalid frequency: {freq}")
        
        return resampled.reset_index()
        
    except Exception as e:
        logger.error(f"Failed to get FRED series {series_id}: {str(e)}")
        raise

def get_yahoo_finance_data(symbol: str) -> pd.DataFrame:
    """
    Yahoo Finance에서 데이터를 가져옵니다.
    
    Args:
        symbol (str): Yahoo Finance 심볼
        
    Returns:
        pd.DataFrame: Yahoo Finance 데이터
    """
    try:
        # 시작 날짜를 타임스탬프로 변환
        start_timestamp = int(datetime.strptime(IMPORT_DATE_START, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.now().timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        params = {
            'period1': start_timestamp,
            'period2': end_timestamp,
            'interval': '1d',
            'events': 'history',
            'includeAdjustedClose': 'true'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # CSV 데이터를 데이터프레임으로 변환
        df = pd.read_csv(pd.StringIO(response.text))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'date', 'Adj Close': 'value'})
        
        # null 값 제거
        df = df[df['value'] != 'null']
        df['value'] = pd.to_numeric(df['value'])
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to get Yahoo Finance data for {symbol}: {str(e)}")
        raise

def collect_release_data(variable_params: pd.DataFrame) -> Dict:
    """
    릴리스 데이터를 수집합니다.
    
    Args:
        variable_params (pd.DataFrame): 변수 정의 데이터프레임
        
    Returns:
        Dict: 수집된 릴리스 데이터
    """
    logger.info("*** Getting Releases History")
    
    try:
        # FRED API 키 가져오기
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            raise ValueError("FRED API key not found in environment variables")
        
        # FRED 소스의 릴리스 파라미터 필터링
        fred_releases = variable_params[
            (variable_params['nc_dfm_input'] == 1) & 
            (variable_params['hist_source'] == 'fred')
        ]
        
        # 병렬로 릴리스 데이터 수집
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_release = {
                executor.submit(
                    get_fred_release_dates, 
                    row['hist_source_key'], 
                    api_key
                ): row['varname']
                for _, row in fred_releases.iterrows()
            }
            
            # 결과 수집
            fred_releases_data = []
            for future in concurrent.futures.as_completed(future_to_release):
                release_name = future_to_release[future]
                try:
                    release_dates = future.result()
                    release_dates['release'] = release_name
                    fred_releases_data.append(release_dates)
                except Exception as e:
                    logger.error(f"Failed to process release {release_name}: {str(e)}")
        
        # 모든 릴리스 데이터 결합
        if fred_releases_data:
            releases = {
                'raw': {
                    'fred': pd.concat(fred_releases_data, ignore_index=True)
                }
            }
            logger.info(f"Collected {len(fred_releases_data)} FRED releases")
            return releases
        else:
            logger.warning("No FRED releases were collected")
            return {'raw': {}}
        
    except Exception as e:
        logger.error(f"Failed to collect release data: {str(e)}")
        raise

def collect_historical_data(variable_params: pd.DataFrame) -> Dict:
    """
    역사적 데이터를 수집합니다.
    
    Args:
        variable_params (pd.DataFrame): 변수 정의 데이터프레임
        
    Returns:
        Dict: 수집된 역사적 데이터
    """
    logger.info("*** Importing Historical Data")
    
    try:
        # FRED API 키 가져오기
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            raise ValueError("FRED API key not found in environment variables")
        
        # FRED 데이터 수집
        fred_vars = variable_params[variable_params['hist_source'] == 'fred']
        fred_data = []
        
        for _, row in fred_vars.iterrows():
            try:
                series_data = get_fred_series(
                    row['hist_source_key'],
                    api_key,
                    row['hist_source_freq']
                )
                series_data['varname'] = row['varname']
                series_data['freq'] = row['hist_source_freq']
                fred_data.append(series_data)
            except Exception as e:
                logger.error(f"Failed to collect FRED data for {row['varname']}: {str(e)}")
        
        hist = {}
        if fred_data:
            hist['raw'] = {
                'fred': pd.concat(fred_data, ignore_index=True)
            }
            logger.info(f"Collected {len(fred_data)} FRED series")
        
        # Yahoo Finance 데이터 수집
        yahoo_vars = variable_params[variable_params['hist_source'] == 'yahoo']
        yahoo_data = []
        
        for _, row in yahoo_vars.iterrows():
            try:
                symbol_data = get_yahoo_finance_data(row['hist_source_key'])
                symbol_data['varname'] = row['varname']
                symbol_data['freq'] = row['hist_source_freq']
                yahoo_data.append(symbol_data)
            except Exception as e:
                logger.error(f"Failed to collect Yahoo Finance data for {row['varname']}: {str(e)}")
        
        if yahoo_data:
            if 'raw' not in hist:
                hist['raw'] = {}
            hist['raw']['yahoo'] = pd.concat(yahoo_data, ignore_index=True)
            logger.info(f"Collected {len(yahoo_data)} Yahoo Finance series")
        
        return hist
        
    except Exception as e:
        logger.error(f"Failed to collect historical data: {str(e)}")
        raise 