# 경제 예측 시스템 (Python 버전)

이 프로젝트는 [econforecasting.com](https://econforecasting.com)에 표시되는 예측을 생성하는 데 사용되는 데이터 스크래핑, 데이터 정제 및 모델링 코드의 Python 버전입니다.

## 프로젝트 구조

```
python_version/
├── modules/           # 주요 코드 모듈
├── deployment/        # 배포 관련 파일
├── tests/            # 테스트 코드
├── config/           # 설정 파일
├── data/             # 데이터 파일
└── utils/            # 유틸리티 함수
```

## 설치 방법

1. Python 3.8 이상 설치
2. 가상환경 생성 및 활성화:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```
3. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 주요 기능

- 데이터 스크래핑 및 수집
- 데이터 정제 및 전처리
- 시계열 분석 및 예측
- 감성 분석
- 구조적 모델링
- 복합 모델링
- 이자율 모델링
- 경기침체 확률 지수 계산

## 데이터베이스 구조

PostgreSQL/TimescaleDB 서버를 사용하여 데이터와 모델 특성을 저장합니다.

## 개발 환경

- Python >= 3.8
- PostgreSQL/TimescaleDB
- Windows/Linux 환경 지원

## 라이선스

원본 R 프로젝트의 라이선스를 따릅니다. 