from fred_downloader import FREDDownloader

fred_keys = [
    'CPIAUCSL', 'CPILFESL', 'CPIUFDSL', 'CPIHOSSL',
    'CUSR0000SETB01', 'PCEPI', 'PCEPILFE', 'DPCERD3A086NBEA',
    'GASREGW', 'DCOILWTICO'
]

if __name__ == "__main__":
    fred = FREDDownloader()
    df = fred.fetch_multiple_series(fred_keys)

    if df is not None:
        fred.save_to_csv(df, 'data/fred_data.csv')
    else:
        print("FRED 데이터 수집 실패")
