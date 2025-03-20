import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 float 타입으로 변환
    
    Args:
        file_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: float 타입으로 변환된 데이터프레임
    """
    df = pd.read_csv(file_path, index_col=0)
    
    # index를 datetime으로 변환
    df.index = pd.to_datetime(df.index)
    
    # '.' 문자를 NaN으로 변환
    df = df.replace('.', np.nan)
    
    # 문자열을 float로 변환 (소수점 자리수 유지)
    for col in df.columns:
        # 각 컬럼의 소수점 자리수 확인
        sample = df[col].dropna().iloc[0]
        if isinstance(sample, str):
            decimal_places = len(sample.split('.')[-1]) if '.' in sample else 0
            # float로 변환하고 원래 소수점 자리수로 반올림
            df[col] = pd.to_numeric(df[col], errors='coerce').round(decimal_places)
    
    return df

def set_cpi_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    CPI 발표일을 저장합니다.
    
    Args:
        df (pd.DataFrame or pd.Series): 입력 데이터프레임 또는 시리즈
        
    Returns:
        pd.DataFrame: CPI 발표일이 저장된 데이터프레임
    """
    # Series인 경우 DataFrame으로 변환
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    
    # CPI 발표일 추출 (NaN이 아닌 날짜들)
    cpi_release_dates = df.index[df.iloc[:, 0].notna()]
    
    # 데이터프레임 생성
    cpi_release_dates_df = pd.DataFrame({
        'release_date': cpi_release_dates,
        'cpi_value': df.iloc[:, 0].loc[cpi_release_dates]
    })
    
    return cpi_release_dates_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    return Y, X

def ffill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    df = df.ffill()
    return df

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    df = df.dropna()
    return df

def cpi_yoy_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    전년 대비 변화율 계산 (YoY)
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        
    Returns:
        pd.DataFrame: YoY 변화율이 계산된 데이터프레임
    """
    # CPI 값만 추출하여 YoY 변화율 계산
    cpi_values = df['cpi_value']
    
    # (현재 - 12개월 전 값 / 12개월 전 값) * 100
    yoy = (cpi_values - cpi_values.shift(12)) / cpi_values.shift(12) * 100
    
    # 결과를 DataFrame으로 변환
    yoy_df = yoy.to_frame('cpi_yoy')
    
    return yoy_df

def save_data(df: pd.DataFrame, file_path: str):
    """
    데이터프레임을 CSV 파일로 저장
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        file_path (str): 저장할 파일 경로
    """
    df.to_csv(file_path, index=True)

def load_meta_data(file_path: str) -> pd.DataFrame:
    """
    메타 데이터를 로드합니다.
    
    Args:
        file_path (str): 메타 데이터 CSV 파일 경로
        
    Returns:
        pd.DataFrame: 메타 데이터프레임
    """
    return pd.read_csv(file_path)

def transform_variable(df: pd.DataFrame, meta_df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    변수별 적절한 변환을 수행합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        meta_df (pd.DataFrame): 메타 데이터프레임
        col_name (str): 변환할 컬럼명
        
    Returns:
        pd.Series: 변환된 시리즈
    """
    # 메타 정보에서 해당 변수의 정보 추출
    var_meta = meta_df[meta_df['id'] == col_name].iloc[0]
    
    # 실제 값이 있는 인덱스만 추출
    valid_idx = df[col_name].notna()
    valid_data = df[col_name].copy()
    
    # 변수별 변환 수행
    if var_meta['frequency_short'] == 'D':  # 일별 데이터
        # 일별 변화율 (결측치 제외)
        valid_data[valid_idx] = valid_data[valid_idx].pct_change(fill_method=None) * 100
            
    elif var_meta['frequency_short'] == 'W':  # 주별 데이터
        # 주별 변화율 (결측치 제외)
        valid_data[valid_idx] = valid_data[valid_idx].pct_change(fill_method=None) * 100
            
    elif var_meta['frequency_short'] == 'M':  # 월별 데이터
        # 월별 데이터는 YoY 변화율로 변환 (결측치 제외)
        valid_data[valid_idx] = (valid_data[valid_idx] - valid_data[valid_idx].shift(12)) / valid_data[valid_idx].shift(12) * 100
        
    return valid_data

def main():
    """
    메인 함수
    """
    # 데이터 로드
    df = load_data('data/Only9X_DF.csv')
    meta_df = load_meta_data('data/Only9X_meta_DF.csv')
    
    # CPI 발표일 추출
    CPI = df.iloc[:, 0]
    cpi_release_date = set_cpi_release_date(CPI)
    
    # CPI YoY 변화율 계산
    cpi_yoy = cpi_yoy_transform(cpi_release_date)
    
    # 일별 리샘플링
    cpi_yoy = cpi_yoy.resample('D').ffill()
    
    # 나머지 변수들 변환
    X = df.iloc[:, 1:]
    transformed_X = pd.DataFrame()
    
    for col in X.columns:
        transformed_X[col] = transform_variable(X, meta_df, col)
    
    # 결측치 처리
    transformed_X = ffill_missing_values(transformed_X)
    
    # 변수들 merge
    df = pd.merge(transformed_X, cpi_yoy, left_index=True, right_index=True, how='left')
    
    # 최종 결측치 처리
    processed_df = df.ffill()
    processed_df = processed_df.dropna()
    # 인덱스 네임 date로 변경
    processed_df.index.name = 'date'

    # 결과 저장
    save_data(processed_df, 'data/processed/processed_data.csv')
    save_data(cpi_release_date, 'data/processed/cpi_release_date.csv')

if __name__ == "__main__":
    main()

