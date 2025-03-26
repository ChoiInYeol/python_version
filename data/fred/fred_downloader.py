import os
import time
import requests
import pandas as pd
from datetime import datetime

class FREDDownloader:
    def __init__(self, api_key: str = None, base_url: str = 'https://api.stlouisfed.org/fred'):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = base_url

    def fetch_series_metadata(self, series_id: str):
        url = f'{self.base_url}/series'
        params = {'api_key': self.api_key, 'file_type': 'json', 'series_id': series_id}
        res = requests.get(url, params=params).json()
        return res.get('seriess', [])

    def fetch_release_id(self, series_id: str):
        url = f'{self.base_url}/series/release'
        params = {'api_key': self.api_key, 'file_type': 'json', 'series_id': series_id}
        res = requests.get(url, params=params)
        releases = res.json().get('releases', [])
        return releases[0]['id'] if releases else None

    def fetch_observations(self, series_id: str):
        url = f'{self.base_url}/series/observations'
        params = {
            'api_key': self.api_key,
            'file_type': 'json',
            'series_id': series_id,
            'observation_start': '2000-01-01',
            'observation_end': datetime.now().strftime('%Y-%m-%d')
        }
        res = requests.get(url, params=params).json()
        return res.get('observations', [])

    def fetch_release_dates(self, release_id: str):
        url = f'{self.base_url}/release/dates'
        params = {'api_key': self.api_key, 'file_type': 'json', 'release_id': release_id}
        res = requests.get(url, params=params).json()
        return res.get('release_dates', [])

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        metadata = self.fetch_series_metadata(series_id)
        if not metadata:
            print(f"skip {series_id}: No metadata available")
            return None

        release_id = self.fetch_release_id(series_id)
        if not release_id:
            print(f"skip {series_id}: No release info")
            return None

        obs = self.fetch_observations(series_id)
        if not obs:
            print(f"skip {series_id}: No observations")
            return None

        obs_df = pd.DataFrame(obs)
        obs_df['date'] = pd.to_datetime(obs_df['date'])
        obs_df[series_id] = pd.to_numeric(obs_df['value'].replace('.', pd.NA), errors='coerce')
        obs_df = obs_df[['date', series_id]]

        release_dates = self.fetch_release_dates(release_id)
        if not release_dates:
            print(f"skip {series_id}: No release dates")
            return None

        release_df = pd.DataFrame(release_dates)
        release_df['release_date'] = pd.to_datetime(release_df['date'])

        df = pd.merge_asof(obs_df, release_df, left_on='date', right_on='release_date', direction='forward')

        # 🔁 인덱스를 관측일(date)에서 발표일(release_date)로 변경하여 Shift
        df = df.set_index('release_date')

        # 중복 제거 및 일간 리샘플링
        df = df.groupby(level=0).first().resample('D').asfreq()

        # 컬럼 이름 정리
        df.rename(columns={series_id: f"{series_id}"}, inplace=True)

        return df

    def fetch_multiple_series(self, series_ids: list, delay: float = 0.1) -> pd.DataFrame:
        data_frames = {}
        for sid in series_ids:
            print(f"Processing {sid}...")
            df = self.fetch_series(sid)
            if df is not None:
                data_frames[sid] = df
                print(f"Downloaded {sid}: {len(df)} rows")
            time.sleep(delay)

        if data_frames:
            merged = pd.concat(data_frames.values(), axis=1)
            return merged.sort_index()
        return None

    def save_to_csv(self, df: pd.DataFrame, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"Saved: {path}")
        print(f"Shape: {df.shape}")
        print(df.head())

