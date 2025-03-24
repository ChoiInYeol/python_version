"""
상수 정의 모듈
"""

# 기본 기업 리스트 정의
DEFAULT_COMPANIES = {
    'Technology': ['MSFT', 'GOOG', 'META', 'AAPL', 'NVDA', 'CSCO'],
    'Finance': ['BRK.B', 'JPM', 'BAC'],
    'Healthcare': ['LLY', 'JNJ', 'ABBV', 'UNH', 'HCA', 'CI'],
    'Consumer': ['PG', 'KO', 'PM', 'WMT', 'COST', 'MCD', 'DIS'],
    'Energy': ['XOM', 'CVX', 'COP', 'NEE', 'SO', 'DUK'],
    'Industrial': ['FDX', 'UNP', 'UPS', 'MCK', 'COR', 'GWW'],
    'Materials': ['LIN', 'SHW', 'ECL', 'FCX', 'NEM', 'VMC'],
    'Real Estate': ['TSLA', 'GM', 'DHI'],
    'Telecom': ['TMUS', 'T', 'VZ'],
    'Services': ['AMZN', 'BKNG'],
    'Utilities': ['WM', 'RSG', 'WMB']
}

# ETF 섹터 리스트
SECTOR_ETF = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

# 실험 조건 정의
EXPERIMENTS = [
    {
        'name': 'Exp 1: 5-day Return',
        'return_period': 5,
        'lag_days': 2,
        'n_pca_comp': 5,
        'lags': 5
    },
    {
        'name': 'Exp 2: 20-day Return',
        'return_period': 20,
        'lag_days': 2,
        'n_pca_comp': 5,
        'lags': 5
    },
    {
        'name': 'Exp 3: No Lag',
        'return_period': 5,
        'lag_days': 0,
        'n_pca_comp': 5,
        'lags': 5
    },
    {
        'name': 'Exp 4: More PCA',
        'return_period': 5,
        'lag_days': 2,
        'n_pca_comp': 10,
        'lags': 5
    }
]

# ETF 실험 조건 정의
ETF_EXPERIMENTS = [
    {
        'name': 'ETF Exp 1: 5-day Return',
        'return_period': 5,
        'lag_days': 2,
        'n_pca_comp': 5,
        'lags': 5
    },
    {
        'name': 'ETF Exp 2: 20-day Return',
        'return_period': 20,
        'lag_days': 2,
        'n_pca_comp': 5,
        'lags': 5
    },
    {
        'name': 'ETF Exp 3: No Lag',
        'return_period': 5,
        'lag_days': 0,
        'n_pca_comp': 5,
        'lags': 5
    },
    {
        'name': 'ETF Exp 4: More PCA',
        'return_period': 5,
        'lag_days': 2,
        'n_pca_comp': 8,  # ETF가 11개이므로 8개로 조정
        'lags': 5
    }
] 