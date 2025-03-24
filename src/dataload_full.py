"""
FRED 및 Yahoo Finance 데이터 다운로드 스크립트
"""
import pandas as pd
import requests
import time
import yfinance as yf
import os
from datetime import datetime, timedelta

FRED_API_KEY = os.getenv('FRED_API_KEY')

# API 설정
API_KEY = FRED_API_KEY
BASE_URL = 'https://api.stlouisfed.org/fred'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# 데이터 소스 설정
source_keys = pd.read_csv('../data/meta/nowcast-variables.csv')
fred_keys = source_keys[source_keys['hist_source'] == 'fred']['hist_source_key']
yf_tickers = source_keys[source_keys['hist_source'] == 'yahoo']['hist_source_key']

def fetch_yahoo_data(tickers, start='2000-01-01', end=END_DATE):
    """
    Yahoo Finance 데이터 다운로드
    
    Args:
        tickers: Yahoo Finance 티커 리스트
        start: 시작 날짜
        end: 종료 날짜
        
    Returns:
        pd.DataFrame: 다운로드된 시장 데이터
    """
    data_dict = {}
    for ticker in tickers:
        print(f"Fetching Yahoo data for {ticker}...")
        data = yf.download(ticker, start=start, end=end)
        data = data['Close']
        data.name = ticker
        data_dict[ticker] = data
    market_df = pd.concat(data_dict.values(), axis=1)
    market_df.columns = data_dict.keys()
    market_df.index.name = 'Date'
    return market_df

def fetch_fred_series(source_key, api_key, base_url, timeout=10):
    """
    FRED 시계열 데이터 다운로드
    
    Args:
        source_key: FRED 시리즈 ID
        api_key: FRED API 키
        base_url: FRED API 기본 URL
        timeout: 요청 타임아웃 (초)
        
    Returns:
        tuple: (메타데이터 DataFrame, 시계열 DataFrame, 에러 메시지)
    """
    try:
        # 메타데이터
        meta_url = f'{base_url}/series'
        meta_params = {'api_key': api_key, 'file_type': 'json', 'series_id': source_key}
        meta_response = requests.get(meta_url, params=meta_params, timeout=timeout)
        meta_json = meta_response.json()
        meta_r = meta_json.get('seriess', [])
        if not meta_r:
            return None, None, f"No metadata available - Response: {meta_json}"

        meta_df = pd.DataFrame(meta_r)
        frequency = meta_df['frequency_short'].iloc[0]  # 주기: D, W, M, Q 등

        release_url = f'{base_url}/series/release'
        release_params = {'api_key': api_key, 'file_type': 'json', 'series_id': source_key}
        release_response = requests.get(release_url, params=release_params, timeout=timeout)
        release_json = release_response.json()
        if 'releases' not in release_json or not release_json['releases']:
            return None, None, "No release info"
        release_id = release_json['releases'][0]['id']

        obs_url = f'{base_url}/series/observations'
        obs_params = {'api_key': api_key, 'file_type': 'json', 'series_id': source_key}
        obs_response = requests.get(obs_url, params=obs_params, timeout=timeout)
        obs_json = obs_response.json()
        if 'observations' not in obs_json or not obs_json['observations']:
            return None, None, "No observations or empty"

        obs_df = pd.DataFrame(obs_json['observations'])
        obs_df['date'] = pd.to_datetime(obs_df['date'], format='%Y-%m-%d', errors='coerce')
        obs_df[source_key] = obs_df['value']

        # 주기에 따라 날짜 조정
        if frequency == 'M':  # 월간
            obs_df['adjusted_date'] = obs_df['date'] + pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(1)
        elif frequency == 'Q':  # 분기
            obs_df['adjusted_date'] = obs_df['date'] + pd.offsets.QuarterEnd(0) + pd.offsets.MonthBegin(2)  # 2개월 뒤
        elif frequency == 'W':  # 주간
            obs_df['adjusted_date'] = obs_df['date'] + pd.offsets.Week(1)  # 1주 뒤
        else:
            obs_df['adjusted_date'] = obs_df['date']  # 기타 주기

        obs_df = obs_df[['adjusted_date', source_key]].set_index('adjusted_date')

        rel_url = f'{base_url}/release/dates'
        rel_params = {'api_key': api_key, 'file_type': 'json', 'release_id': release_id}
        rel_response = requests.get(rel_url, params=rel_params, timeout=timeout)
        rel_data = rel_response.json()['release_dates']
        if not rel_data:
            return None, None, "No release dates"

        rel_df = pd.DataFrame(rel_data)
        rel_df['release_date'] = pd.to_datetime(rel_df['date'], format='%Y-%m-%d', errors='coerce')
        rel_df = rel_df[['release_date']].sort_values('release_date')

        temp_df = obs_df.reset_index()
        temp_df = pd.merge_asof(temp_df, rel_df, left_on='adjusted_date', right_on='release_date', direction='forward')
        temp_df = temp_df.drop(columns=['adjusted_date'])
        temp_df = temp_df.groupby('release_date').first()
        temp_df = temp_df.resample('D').asfreq()

        return meta_df, temp_df, None

    except Exception as e:
        return None, None, str(e)

def fetch_fred_data(keys, api_key, base_url, delay_seconds=0.3):
    """
    여러 FRED 시계열 데이터 다운로드
    
    Args:
        keys: FRED 시리즈 ID 리스트
        api_key: FRED API 키
        base_url: FRED API 기본 URL
        delay_seconds: 요청 간 딜레이 (초)
        
    Returns:
        tuple: (메타데이터 딕셔너리, 데이터 딕셔너리, 스킵된 키 리스트)
    """
    meta_dict = {}
    data_dict = {}
    skipped_keys = []

    for source_key in keys:
        print(f"Processing {source_key}...")
        meta_df, series_df, error = fetch_fred_series(source_key, api_key, base_url)
        if error:
            print(f"skip {source_key}: {error}")
            skipped_keys.append((source_key, error))
        else:
            meta_dict[source_key] = meta_df
            data_dict[source_key] = series_df
        time.sleep(delay_seconds)

    return meta_dict, data_dict, skipped_keys

if __name__ == "__main__":
    # data 디렉토리 생성
    os.makedirs('data', exist_ok=True)

    # Yahoo Finance 데이터 다운로드
    print("Fetching Yahoo Finance data...")
    market_df = fetch_yahoo_data(yf_tickers)
    print(f"Market data shape: {market_df.shape}")

    # FRED 데이터 다운로드 (두 부분으로 나누어 실행)
    mid_point = len(fred_keys) // 2
    first_half = fred_keys[:mid_point]
    second_half = fred_keys[mid_point:]

    print("\nFetching first half of FRED keys...")
    meta_dict1, data_dict1, skipped1 = fetch_fred_data(first_half, API_KEY, BASE_URL)

    print("\nWaiting before second half...")
    time.sleep(60)

    print("\nFetching second half of FRED keys...")
    meta_dict2, data_dict2, skipped2 = fetch_fred_data(second_half, API_KEY, BASE_URL)

    # 결과 병합
    meta_dict = {**meta_dict1, **meta_dict2}
    data_dict = {**data_dict1, **data_dict2}
    skipped_keys = skipped1 + skipped2

    fred_merged_df = pd.concat(data_dict.values(), axis=1, join='outer')
    fred_meta_full_df = pd.concat(meta_dict.values(), axis=0, ignore_index=True)
    final_df = pd.concat([market_df, fred_merged_df], axis=1)

    # 결과 저장
    output_path = 'data/fred_data.csv'
    final_df.to_csv(output_path)
    
    # 메타데이터 저장
    meta_output_path = 'data/fred_meta.csv'
    fred_meta_full_df.to_csv(meta_output_path)

    # 결과 출력
    print("\n=== 다운로드 결과 ===")
    print(f"FRED 메타데이터 df shape: {fred_meta_full_df.shape}")
    print(f"FRED merged_df shape: {fred_merged_df.shape}")
    print(f"Market df shape: {market_df.shape}")
    print(f"Final df shape: {final_df.shape}")
    
    print("\n=== 메타데이터 샘플 ===")
    print(fred_meta_full_df.head())
    
    print("\n=== 데이터 샘플 ===")
    print(final_df.head())
    
    print("\n=== 스킵된 키 ===")
    for key, reason in skipped_keys:
        print(f"{key}: {reason}")
    
    print(f"\n데이터 저장 완료:")
    print(f"- 데이터: {output_path}")
    print(f"- 메타데이터: {meta_output_path}")