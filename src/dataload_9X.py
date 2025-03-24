"""
FRED 데이터 다운로드 스크립트
"""
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta

FRED_API_KEY = os.getenv('FRED_API_KEY')

# API 설정
API_KEY = FRED_API_KEY
BASE_URL = 'https://api.stlouisfed.org/fred'

# 필수 FRED series
fred_keys = [
    'CPIAUCSL',  # CPI headline
    'CPILFESL',  # Core CPI
    'CPIUFDSL',  # CPI Food
    'CPIHOSSL',  # CPI Food at home
    'CUSR0000SETB01',  # CPI Gasoline
    'PCEPI',  # PCE price index
    'PCEPILFE',  # Core PCE price index
    'DPCERD3A086NBEA',  # PCE food off-premises
    'GASREGW',  # Weekly gasoline price
    'DCOILWTICO'  # Crude oil (daily WTI 기준)
]

def fetch_fred_series(source_key: str, api_key: str, base_url: str) -> pd.DataFrame:
    """
    FRED 시계열 데이터 다운로드
    
    Args:
        source_key: FRED 시리즈 ID
        api_key: FRED API 키
        base_url: FRED API 기본 URL
        
    Returns:
        pd.DataFrame: 다운로드된 시계열 데이터
    """
    try:
        # 메타데이터 수집
        meta_url = f'{base_url}/series'
        meta_params = {'api_key': api_key, 'file_type': 'json', 'series_id': source_key}
        meta_r = requests.get(meta_url, params=meta_params).json().get('seriess', [])
        if not meta_r:
            print(f"skip {source_key}: No metadata available")
            return None
        meta_df = pd.DataFrame(meta_r)

        # release id
        release_url = f'{base_url}/series/release'
        release_params = {'api_key': api_key, 'file_type': 'json', 'series_id': source_key}
        release_response = requests.get(release_url, params=release_params).json()
        if 'releases' not in release_response or not release_response['releases']:
            print(f"skip {source_key}: No release info")
            return None
        release_id = release_response['releases'][0]['id']

        # 데이터 다운로드
        obs_url = f'{base_url}/series/observations'
        obs_params = {
            'api_key': api_key,
            'file_type': 'json',
            'series_id': source_key,
            'observation_start': '2000-01-01',  # 2000년부터 시작
            'observation_end': datetime.now().strftime('%Y-%m-%d')  # 현재 날짜까지
        }
        
        response = requests.get(obs_url, params=obs_params)
        data = response.json()
        
        if 'observations' not in data or not data['observations']:
            print(f"skip {source_key}: No observations")
            return None
            
        # 데이터프레임 생성
        obs_df = pd.DataFrame(data['observations'])
        obs_df['date'] = pd.to_datetime(obs_df['date'], format='%Y-%m-%d')
        obs_df[source_key] = pd.to_numeric(obs_df['value'].replace('.', pd.NA), errors='coerce')
        obs_df = obs_df[['date', source_key]]

        # 릴리즈 날짜 수집
        rel_url = f'{base_url}/release/dates'
        rel_params = {'api_key': api_key, 'file_type': 'json', 'release_id': release_id}
        rel_data = requests.get(rel_url, params=rel_params).json()['release_dates']
        if not rel_data:
            print(f"skip {source_key}: No release dates")
            return None

        # 릴리즈 날짜 데이터프레임 생성
        rel_df = pd.DataFrame(rel_data)
        rel_df['release_date'] = pd.to_datetime(rel_df['date'], format='%Y-%m-%d')
        
        # 데이터 병합 및 처리
        temp_df = pd.merge_asof(obs_df, rel_df, left_on='date', right_on='release_date', direction='forward')
        
        # 불필요한 컬럼 제거
        columns_to_drop = ['date_y']
        if 'realtime_start' in temp_df.columns:
            columns_to_drop.extend(['realtime_start', 'realtime_end'])
        if 'release_id' in temp_df.columns:
            columns_to_drop.append('release_id')
            
        temp_df = temp_df.drop(columns=columns_to_drop)
        
        # date를 인덱스로 설정
        temp_df = temp_df.set_index('date')
        
        # 중복 제거 및 일간 데이터로 리샘플링
        temp_df = temp_df.groupby(level=0).first()
        temp_df = temp_df.resample('D').asfreq()
        
        # 릴리즈 날짜 컬럼 추가
        temp_df[f'{source_key}_release_date'] = temp_df['release_date']
        temp_df = temp_df.drop(columns=['release_date'])
        
        return temp_df
        
    except Exception as e:
        print(f"Error fetching {source_key}: {str(e)}")
        return None

def fetch_fred_data(keys: list, api_key: str, base_url: str, delay: float = 0.1) -> pd.DataFrame:
    """
    여러 FRED 시계열 데이터 다운로드
    
    Args:
        keys: FRED 시리즈 ID 리스트
        api_key: FRED API 키
        base_url: FRED API 기본 URL
        delay: 요청 간 딜레이 (초)
        
    Returns:
        pd.DataFrame: 병합된 시계열 데이터
    """
    data_dict = {}
    
    for key in keys:
        print(f"Processing {key}...")
        df = fetch_fred_series(key, api_key, base_url)
        if df is not None:
            data_dict[key] = df
            print(f"Downloaded {key} with {len(df)} observations")
        time.sleep(delay)
    
    # 모든 데이터프레임 병합
    if data_dict:
        merged_df = pd.concat(data_dict.values(), axis=1, join='outer')
        # 날짜 기준으로 정렬
        merged_df = merged_df.sort_index()
        return merged_df
    return None

if __name__ == "__main__":
    # data 디렉토리 생성
    os.makedirs('data', exist_ok=True)

    # 데이터 다운로드
    df = fetch_fred_data(fred_keys, API_KEY, BASE_URL)

    if df is not None:
        # 결과 저장
        output_path = 'data/fred_data.csv'
        df.to_csv(output_path)
        print(f"\n데이터 저장 완료: {output_path}")
        print(f"데이터 크기: {df.shape}")
        print("\n데이터 샘플:")
        print(df.head())
        print("\n데이터 정보:")
        print(df.info())
        print("\n각 시리즈별 데이터 범위:")
        for col in df.columns:
            print(f"{col}: {df[col].first()} ~ {df[col].last()}")
    else:
        print("데이터 다운로드 실패")