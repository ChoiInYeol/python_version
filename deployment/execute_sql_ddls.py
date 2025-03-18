"""
SQL DDL 파일 실행 스크립트

이 스크립트는 sql-ddls 폴더의 모든 SQL DDL 파일들을 실행합니다.
"""

import logging
import os
from pathlib import Path
from sqlalchemy import create_engine, text

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def execute_sql_file(engine, file_path: str) -> None:
    """
    SQL 파일을 실행합니다.

    Args:
        engine: SQLAlchemy 엔진 객체
        file_path (str): SQL 파일 경로
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        with engine.connect() as conn:
            conn.execute(text(sql_content))
            conn.commit()

        logger.info(f"SQL 파일 실행 완료: {file_path}")

    except Exception as e:
        logger.error(f"SQL 파일 실행 중 오류 발생 ({file_path}): {str(e)}")
        raise

def main():
    """메인 실행 함수"""
    try:
        # 환경 변수 확인
        ef_dir = os.getenv('EF_DIR')
        if not ef_dir:
            raise ValueError("EF_DIR 환경 변수가 설정되지 않았습니다.")

        # 데이터베이스 연결
        from model_inputs.constants import CONST
        engine = create_engine(
            f"postgresql://{CONST['DB_USERNAME']}:{CONST['DB_PASSWORD']}@"
            f"{CONST['DB_SERVER']}:5432/{CONST['DB_DATABASE']}"
        )

        # SQL DDL 파일 경로
        sql_ddls_dir = os.path.join(ef_dir, 'deployment', 'sql-ddls')

        # SQL 파일 실행 순서 정의
        sql_files = [
            'job-logs.sql',
            'jobscripts.sql',
            'forecast-hist-values-v2.sql',
            'forecast-hist-values-v2-latest.sql',
            'forecast-values-v2.sql',
            'forecast-values-v2-latest.sql',
            'forecast-values-v2-all.sql'
        ]

        # 각 SQL 파일 실행
        for sql_file in sql_files:
            file_path = os.path.join(sql_ddls_dir, sql_file)
            if os.path.exists(file_path):
                execute_sql_file(engine, file_path)
            else:
                logger.warning(f"SQL 파일을 찾을 수 없습니다: {file_path}")

        logger.info("모든 SQL DDL 파일이 성공적으로 실행되었습니다.")

    except Exception as e:
        logger.error(f"SQL DDL 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
