import pandas as pd
import numpy as np
from typing import Tuple
from enum import Enum
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 상수 정의
TARGET_COLUMN = 'CPIAUCSL'  # CPI 데이터 컬럼명
RELEASE_DATE_COLUMN = 'release_date'  # 발표일 컬럼명
VALUE_COLUMN = 'CPIAUCSL'  # CPI 값 컬럼명
TARGET_YOY_COLUMN = 'CPI_YOY'  # CPI YoY 변화율 컬럼명

class TransformationType(Enum):
    """변환 타입 정의"""
    YOY = 'yoy'  # 전년 대비 변화율
    QOQ = 'qoq'  # 전분기 대비 변화율
    MOM = 'mom'  # 전월 대비 변화율
    WOW = 'wow'  # 전주 대비 변화율
    DOD = 'dod'  # 전일 대비 변화율
    DIFF = 'diff'  # 차분
    LOG = 'log'  # 로그 변환
    LOG_DIFF = 'log_diff'  # 로그 차분
    LOG_YOY = 'log_yoy'  # 로그 전년 대비 변화율
    LOG_QOQ = 'log_qoq'  # 로그 전분기 대비 변화율
    LOG_MOM = 'log_mom'  # 로그 전월 대비 변화율
    SEAS_DIFF = 'seas_diff'  # 계절 차분
    LOG_SEAS_DIFF = 'log_seas_diff'  # 로그 계절 차분

class SeasonalAdjustment(Enum):
    """계절조정 타입 정의"""
    SAAR = 'SAAR'  # Seasonally Adjusted Annual Rate
    SA = 'SA'      # Seasonally Adjusted
    NSA = 'NSA'    # Not Seasonally Adjusted

def load_data(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 float 타입으로 변환
    
    Args:
        file_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: float 타입으로 변환된 데이터프레임
    """
    # low_memory=False로 설정하여 dtype 경고 방지
    df = pd.read_csv(file_path, index_col=0, low_memory=False)
    
    # index를 datetime으로 변환
    df.index = pd.to_datetime(df.index)
    
    # '.' 문자를 NaN으로 변환
    df = df.replace('.', np.nan)
    
    # 문자열을 float로 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_meta_data(file_path: str) -> pd.DataFrame:
    """
    메타데이터를 로드합니다.
    
    Args:
        file_path (str): 메타데이터 CSV 파일 경로
        
    Returns:
        pd.DataFrame: 메타데이터 데이터프레임
    """
    return pd.read_csv(file_path)

def determine_transform_type(meta_row: pd.Series) -> TransformationType:
    """
    변수의 특성에 따라 적절한 변환 방법을 결정합니다.
    
    Args:
        meta_row (pd.Series): 메타데이터 행
        
    Returns:
        TransformationType: 변환 타입
    """
    units = meta_row['units'].lower() if isinstance(meta_row['units'], str) else ''
    freq = meta_row['frequency_short']
    seas_adj = meta_row['seasonal_adjustment_short']
    
    # 계절조정이 되지 않은(NSA) 데이터 처리
    if seas_adj == SeasonalAdjustment.NSA.value:
        if 'dollars' in units or '$' in units:
            return TransformationType.LOG_SEAS_DIFF
        else:
            return TransformationType.SEAS_DIFF
    
    # 계절조정된(SA, SAAR) 데이터 처리
    # 금액 단위(달러) 데이터 처리
    if 'dollars' in units or '$' in units:
        if freq == 'D':  # 일별 데이터
            return TransformationType.LOG_DIFF
        elif freq == 'W':  # 주별 데이터
            return TransformationType.LOG_MOM
        elif freq == 'M':  # 월별 데이터
            return TransformationType.LOG_YOY
        elif freq == 'Q':  # 분기별 데이터
            return TransformationType.LOG_QOQ
    
    # 비율(%) 데이터 처리
    elif 'percent' in units or '%' in units:
        return TransformationType.DIFF
    
    # 주기별 기본 변환 방법 (계절조정된 데이터)
    elif freq == 'D':  # 일별 데이터
        return TransformationType.DOD
    elif freq == 'W':  # 주별 데이터
        return TransformationType.WOW
    elif freq == 'M':  # 월별 데이터
        return TransformationType.MOM if seas_adj == SeasonalAdjustment.SAAR.value else TransformationType.YOY
    elif freq == 'Q':  # 분기별 데이터
        return TransformationType.QOQ
    
    # 기본값
    return TransformationType.YOY

def set_cpi_release_date(series: pd.Series) -> pd.DataFrame:
    """
    CPI 발표일을 저장합니다.
    
    Args:
        series (pd.Series): CPI 시계열 데이터
        
    Returns:
        pd.DataFrame: CPI 발표일이 저장된 데이터프레임
    """
    # CPI 발표일 추출 (NaN이 아닌 날짜들)
    release_dates = series.index[series.notna()]
    
    # 데이터프레임 생성
    release_dates_df = pd.DataFrame({
        RELEASE_DATE_COLUMN: release_dates,
        VALUE_COLUMN: series.loc[release_dates]
    })
    
    return release_dates_df

def cpi_yoy_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    전년 대비 변화율 계산 (YoY)
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        
    Returns:
        pd.DataFrame: YoY 변화율이 계산된 데이터프레임
    """
    # CPI 값만 추출하여 YoY 변화율 계산
    values = df[VALUE_COLUMN]
    
    # (현재 - 12개월 전 값 / 12개월 전 값) * 100
    yoy = (values - values.shift(12)) / values.shift(12) * 100
    
    # 결과를 DataFrame으로 변환
    yoy_df = yoy.to_frame(TARGET_YOY_COLUMN)
    
    return yoy_df

def get_seasonal_period(freq: str) -> int:
    """
    주기에 따른 계절성 주기를 반환합니다.
    
    Args:
        freq (str): 데이터 주기 (D, W, M, Q)
        
    Returns:
        int: 계절성 주기
    """
    if freq == 'D':
        return 7  # 일주일
    elif freq == 'W':
        return 52  # 52주
    elif freq == 'M':
        return 12  # 12개월
    elif freq == 'Q':
        return 4  # 4분기
    return 1

def transform_single_column(series: pd.Series, transform_type: TransformationType, freq: str = 'M') -> pd.Series:
    """
    단일 시계열 변수를 변환합니다.
    
    Args:
        series (pd.Series): 변환할 시계열 데이터
        transform_type (TransformationType): 변환 타입
        freq (str): 데이터 주기 (기본값: 'M')
        
    Returns:
        pd.Series: 변환된 시계열 데이터
    """
    # 결측치 제거
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return pd.Series(index=series.index)
    
    # 계절성 주기 설정
    seasonal_period = get_seasonal_period(freq)
    
    if transform_type == TransformationType.YOY:
        transformed = (clean_series - clean_series.shift(12)) / clean_series.shift(12) * 100
    elif transform_type == TransformationType.QOQ:
        transformed = (clean_series - clean_series.shift(4)) / clean_series.shift(4) * 100
    elif transform_type == TransformationType.MOM:
        transformed = (clean_series - clean_series.shift(1)) / clean_series.shift(1) * 100
    elif transform_type == TransformationType.WOW:
        transformed = (clean_series - clean_series.shift(1)) / clean_series.shift(1) * 100
    elif transform_type == TransformationType.DOD:
        transformed = (clean_series - clean_series.shift(1)) / clean_series.shift(1) * 100
    elif transform_type == TransformationType.DIFF:
        transformed = clean_series.diff()
    elif transform_type == TransformationType.LOG:
        transformed = np.log(clean_series)
    elif transform_type == TransformationType.LOG_DIFF:
        transformed = np.log(clean_series).diff()
    elif transform_type == TransformationType.LOG_YOY:
        transformed = np.log(clean_series / clean_series.shift(12)) * 100
    elif transform_type == TransformationType.LOG_QOQ:
        transformed = np.log(clean_series / clean_series.shift(4)) * 100
    elif transform_type == TransformationType.LOG_MOM:
        transformed = np.log(clean_series / clean_series.shift(1)) * 100
    elif transform_type == TransformationType.SEAS_DIFF:
        transformed = clean_series.diff(seasonal_period)
    elif transform_type == TransformationType.LOG_SEAS_DIFF:
        transformed = np.log(clean_series).diff(seasonal_period)
    else:
        transformed = clean_series
        
    # 일간 데이터로 리샘플링
    return transformed.resample('D').ffill()

def process_data(df: pd.DataFrame, meta_df: pd.DataFrame, use_cpi_lags: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    데이터를 전처리하여 X, y, 발표일 데이터를 생성합니다.
    """
    try:
        # 오늘 날짜 추출
        today = pd.Timestamp.today()
        logger.info(f"데이터 처리 시작: {today}")
        
        # CPI 데이터 추출 및 처리
        cpi_data = df[TARGET_COLUMN]
        release_dates = set_cpi_release_date(cpi_data)
        y = cpi_yoy_transform(release_dates)
        
        # Y는 리샘플링하지 않고 발표일의 값만 유지
        y = y[TARGET_YOY_COLUMN]
        logger.info(f"CPI YoY 변환 완료: {len(y)} 개의 데이터 포인트")
        
        # CPI 관련 데이터 목록 (잠재적인 Y)
        cpi_columns = ['CPILFESL', 'CPIUFDSL', 'CPIHOSSL', 'CPIAUCSL']
        
        # 설명변수(X) 추출 (CPI 관련 데이터 제외)
        X_columns = [col for col in df.columns if col not in cpi_columns]
        logger.info(f"변환할 컬럼 수: {len(X_columns)}")
        transformed_series = []
        
        # 각 컬럼별 독립적 변환 수행
        for col in X_columns:
            try:
                series = df[col]
                meta_row = meta_df[meta_df['id'] == col]
                
                if not meta_row.empty:
                    meta_info = meta_row.iloc[0]
                    transform_type = determine_transform_type(meta_info)
                    logger.debug(f"{col} 변환: {transform_type.value}")
                    transformed = transform_single_column(
                        series,
                        transform_type,
                        meta_info['frequency_short']
                    )
                else:
                    logger.warning(f"{col}의 메타데이터가 없습니다. YoY 변환을 사용합니다.")
                    transformed = transform_single_column(series, TransformationType.YOY)
                
                transformed.name = col
                transformed_series.append(transformed)
                
            except Exception as e:
                logger.error(f"{col} 변환 중 오류 발생: {str(e)}")
                continue
        
        # 모든 변환된 시리즈를 데이터프레임으로 병합
        date_range = pd.date_range(start=df.index.min(), end=max(df.index.max(), today), freq='D')
        X = pd.DataFrame(index=date_range)
        
        # 변환된 시계열 데이터 추가
        for series in transformed_series:
            X = X.join(series)
        
        # CPI 관련 데이터 처리
        if use_cpi_lags:
            logger.info(f"CPI lag 생성 시작: {cpi_columns}")
            
            # CPI 관련 데이터의 lag 생성
            for col in cpi_columns:
                if col in df.columns:
                    logger.debug(f"{col} lag 생성 중...")
                    # CPI 데이터 추출 및 YoY 변환
                    cpi_series = df[col].dropna()
                    cpi_yoy = (cpi_series - cpi_series.shift(12)) / cpi_series.shift(12) * 100
                    cpi_yoy_daily = cpi_yoy.resample('D').ffill()

                    # 발표일 필터링
                    release_dates_filtered = release_dates[release_dates[RELEASE_DATE_COLUMN] >= cpi_yoy_daily.index.min()]
                    release_dates_sorted = sorted(release_dates_filtered[RELEASE_DATE_COLUMN])
                    release_dates_sorted.append(cpi_yoy_daily.index.max() + pd.Timedelta(days=1))  # 마지막 보정

                    lag1_series = pd.Series(index=cpi_yoy_daily.index, dtype='float64')

                    # 발표일 간 구간별 값 채우기 (lag1)
                    for i in range(len(release_dates_sorted) - 1):
                        current_release = release_dates_sorted[i]
                        next_release = release_dates_sorted[i + 1] - pd.Timedelta(days=1)
                        if current_release not in cpi_yoy_daily.index:
                            continue
                        lag1_series.loc[current_release + pd.Timedelta(days=1): next_release] = cpi_yoy_daily.loc[current_release]

                    # lag1 기준으로 lag2, lag3 생성
                    X[f'{col}_LAG_1'] = lag1_series
                    X[f'{col}_LAG_2'] = lag1_series.shift(30)
                    X[f'{col}_LAG_3'] = lag1_series.shift(60)
                else:
                    logger.warning(f"{col} 컬럼이 데이터프레임에 없습니다.")
        
        # 모든 행이 NaN인 컬럼 제거
        X = X.dropna(axis=1, how='all')
        logger.info(f"NaN 컬럼 제거 후 남은 컬럼 수: {len(X.columns)}")
        logger.debug(f"남은 컬럼 목록: {X.columns.tolist()}")
        
        # X만 전방향 결측치 보간
        X = X.ffill()
        
        X.index.name = 'date'
        y.index.name = 'date'
        
        logger.info(f"데이터 처리 완료: X shape={X.shape}, y shape={y.shape}")
        return X, y, release_dates
        
    except Exception as e:
        logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
        raise

def main():
    """
    메인 함수
    
    데이터를 로드하고 전처리하여 최종 데이터셋을 생성합니다.
    - CPI(CPIAUCSL)를 타겟 변수로 사용
    - 나머지 변수들은 각각의 특성에 맞게 변환
      1. 금액 단위: 로그 차분
      2. 비율: 차분
      3. 월별 데이터: YoY
      4. 일별 데이터: 로그 차분
    """
    try:
        # 데이터 로드
        DATA_NAME = '9X'
        META_NAME = '9X_meta'
        
        # CPI 관련 데이터의 lag 사용 여부 설정
        USE_CPI_LAGS = True  # True로 설정하면 CPI 관련 데이터의 lag를 사용
        
        logger.info(f"데이터 로드 시작: {DATA_NAME}, {META_NAME}")
        df = load_data(f'data/{DATA_NAME}.csv')
        meta_df = load_meta_data(f'data/{META_NAME}.csv')
        logger.info(f"데이터 로드 완료: df shape={df.shape}, meta_df shape={meta_df.shape}")
        
        # 데이터 전처리
        X, y, release_dates = process_data(df, meta_df, use_cpi_lags=USE_CPI_LAGS)
        
        # 최종 데이터셋 생성
        processed_df = pd.concat([X, y], axis=1)
        processed_df = processed_df.ffill()
        processed_df.index.name = 'date'
        
        # 결과 저장
        logger.info("결과 저장 시작")
        processed_df.to_csv(f'data/processed/{DATA_NAME}.csv', index=True)
        release_dates.to_csv(f'data/processed/{DATA_NAME}_cpi_release_date.csv', index=True)
        X.to_csv(f'data/processed/X_{DATA_NAME}.csv', index=True)
        y.to_frame().to_csv(f'data/processed/y_{DATA_NAME}.csv', index=True)
        logger.info("결과 저장 완료")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()

