"""
예측 테이블 생성을 위한 스크립트

이 스크립트는 예측 시스템에 필요한 데이터베이스 테이블들을 생성합니다.
"""

import logging
import os
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 상수 설정
JOB_NAME = 'create-sql-tables'
EF_DIR = os.getenv('EF_DIR')

def create_insert_query(df: pd.DataFrame, table_name: str, on_conflict: str) -> str:
    """
    DataFrame을 SQL INSERT 쿼리로 변환하는 함수

    Args:
        df (pd.DataFrame): 변환할 DataFrame
        table_name (str): 테이블 이름
        on_conflict (str): ON CONFLICT 절

    Returns:
        str: SQL INSERT 쿼리
    """
    columns = df.columns.tolist()
    values = []

    for _, row in df.iterrows():
        row_values = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                row_values.append('NULL')
            elif isinstance(val, str):
                row_values.append(f"'{val}'")
            else:
                row_values.append(str(val))
        values.append(f"({', '.join(row_values)})")

    query = f"""
    INSERT INTO {table_name} ({', '.join(columns)})
    VALUES {', '.join(values)}
    {on_conflict}
    """
    return query

def main():
    """메인 실행 함수"""
    try:
        # 데이터베이스 연결
        from model_inputs.constants import CONST
        engine = create_engine(
            f"postgresql://{CONST['DB_USERNAME']}:{CONST['DB_PASSWORD']}@"
            f"{CONST['DB_SERVER']}:5432/{CONST['DB_DATABASE']}"
        )

        # 테이블 구조 읽기
        forecasts = pd.read_excel(
            os.path.join(EF_DIR, 'deployment', 'forecast-tables.xlsx'),
            sheet_name='forecasts'
        )

        forecast_variables = pd.read_excel(
            os.path.join(EF_DIR, 'deployment', 'forecast-tables.xlsx'),
            sheet_name='variables'
        )

        forecast_hist_releases = pd.read_excel(
            os.path.join(EF_DIR, 'deployment', 'forecast-tables.xlsx'),
            sheet_name='releases'
        )

        # 릴리스와 변수 매핑 검증
        missing_releases = forecast_hist_releases[
            ~forecast_hist_releases['id'].isin(forecast_variables['release'])
        ]
        missing_variables = forecast_variables[
            ~forecast_variables['release'].isin(forecast_hist_releases['id'])
        ]

        if not missing_releases.empty:
            logger.warning("매핑되지 않은 릴리스:")
            logger.warning(missing_releases)

        if not missing_variables.empty:
            logger.warning("매핑되지 않은 변수:")
            logger.warning(missing_variables)

        # forecasts 테이블 생성
        with engine.connect() as conn:
            conn.execute(text('DROP TABLE IF EXISTS forecasts CASCADE'))
            conn.execute(text('''
                CREATE TABLE forecasts (
                    id VARCHAR(50) CONSTRAINT forecasts_pk PRIMARY KEY,
                    fullname VARCHAR(255) CONSTRAINT forecasts_fullname_uk UNIQUE NOT NULL,
                    shortname VARCHAR(255) CONSTRAINT forecasts_shortname_uk UNIQUE NOT NULL,
                    external BOOLEAN NOT NULL,
                    update_freq CHAR(1) NOT NULL,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            '''))

            insert_query = create_insert_query(
                forecasts,
                'forecasts',
                '''ON CONFLICT ON CONSTRAINT forecasts_pk DO UPDATE
                SET
                fullname=EXCLUDED.fullname,
                shortname=EXCLUDED.shortname,
                external=EXCLUDED.external,
                update_freq=EXCLUDED.update_freq,
                description=EXCLUDED.description'''
            )
            conn.execute(text(insert_query))

        # forecast_variables 테이블 생성
        with engine.connect() as conn:
            conn.execute(text('DROP TABLE IF EXISTS forecast_variables CASCADE'))
            conn.execute(text('''
                CREATE TABLE forecast_variables (
                    varname VARCHAR(50) CONSTRAINT forecast_variables_pk PRIMARY KEY,
                    fullname VARCHAR(255) CONSTRAINT forecast_variables_fullname_uk UNIQUE NOT NULL,
                    dispgroup VARCHAR(255) NOT NULL,
                    disporder INT NULL,
                    release VARCHAR(50) NOT NULL,
                    units VARCHAR(255) NOT NULL,
                    d1 VARCHAR(50) NOT NULL,
                    d2 VARCHAR(50) NOT NULL,
                    hist_source VARCHAR(255) NOT NULL,
                    hist_source_key VARCHAR(255) NOT NULL,
                    hist_source_freq CHAR(1) NOT NULL,
                    hist_source_transform VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT forecast_variables_release_fk FOREIGN KEY (release)
                        REFERENCES forecast_hist_releases (id) ON DELETE CASCADE ON UPDATE CASCADE
                )
            '''))

            insert_query = create_insert_query(
                forecast_variables,
                'forecast_variables',
                '''ON CONFLICT ON CONSTRAINT forecast_variables_pk DO UPDATE
                SET
                fullname=EXCLUDED.fullname,
                dispgroup=EXCLUDED.dispgroup,
                disporder=EXCLUDED.disporder,
                release=EXCLUDED.release,
                units=EXCLUDED.units,
                d1=EXCLUDED.d1,
                d2=EXCLUDED.d2,
                hist_source=EXCLUDED.hist_source,
                hist_source_key=EXCLUDED.hist_source_key,
                hist_source_freq=EXCLUDED.hist_source_freq,
                hist_source_transform=EXCLUDED.hist_source_transform'''
            )
            conn.execute(text(insert_query))

        # forecast_hist_values 테이블 생성
        with engine.connect() as conn:
            conn.execute(text('DROP TABLE IF EXISTS forecast_hist_values CASCADE'))
            conn.execute(text('''
                CREATE TABLE forecast_hist_values (
                    vdate DATE NOT NULL,
                    form VARCHAR(50) NOT NULL,
                    freq CHAR(1) NOT NULL,
                    varname VARCHAR(50) NOT NULL,
                    date DATE NOT NULL,
                    value NUMERIC(20, 4) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (vdate, form, freq, varname, date),
                    CONSTRAINT forecast_hist_values_fk FOREIGN KEY (varname)
                        REFERENCES forecast_variables (varname) ON DELETE CASCADE ON UPDATE CASCADE
                )
            '''))

            conn.execute(text('''
                SELECT create_hypertable(
                    relation => 'forecast_hist_values',
                    time_column_name => 'vdate'
                );
            '''))

        # forecast_values 테이블 생성
        with engine.connect() as conn:
            conn.execute(text('DROP TABLE IF EXISTS forecast_values CASCADE'))
            conn.execute(text('''
                CREATE TABLE forecast_values (
                    forecast VARCHAR(50) NOT NULL,
                    vdate DATE NOT NULL,
                    form VARCHAR(50) NOT NULL,
                    freq CHAR(1) NOT NULL,
                    varname VARCHAR(50) NOT NULL,
                    date DATE NOT NULL,
                    value NUMERIC(20, 4) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (forecast, vdate, form, freq, varname, date),
                    CONSTRAINT forecast_values_varname_fk FOREIGN KEY (varname)
                        REFERENCES forecast_variables (varname) ON DELETE CASCADE ON UPDATE CASCADE,
                    CONSTRAINT forecast_values_forecast_fk FOREIGN KEY (forecast)
                        REFERENCES forecasts (id) ON DELETE CASCADE ON UPDATE CASCADE
                )
            '''))

            conn.execute(text('''
                SELECT create_hypertable(
                    relation => 'forecast_values',
                    time_column_name => 'vdate'
                );
            '''))

        logger.info("모든 테이블이 성공적으로 생성되었습니다.")

    except Exception as e:
        logger.error(f"테이블 생성 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
