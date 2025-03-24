"""
동적요인모형(DFM) 기반 nowcasting/forecasting 시스템
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from statsmodels.tsa.api import VAR
from pykalman import KalmanFilter
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dfm_kalman_model")

class DFMModel:
    def __init__(
        self, 
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        n_factors: Optional[int] = None,
        model_path: Optional[str] = None,
        forecast_horizon: int = 30
    ):
        """
        DFM 모델 초기화
        
        Args:
            X_path: 고빈도 지표 데이터 경로
            y_path: 목표 변수 데이터 경로
            target: 목표 변수 컬럼명
            n_factors: 추출할 요인 수
            model_path: 모델 저장/로드 경로
            forecast_horizon: 예측 기간
        """
        self.X_path = X_path
        self.y_path = y_path
        self.target = target
        self.n_factors = n_factors
        self.model_path = model_path
        self.forecast_horizon = forecast_horizon

        self.X = None
        self.y = None
        self.pca_model = None
        self.Z_smoothed = None
        self.var_model = None
        self.kalman_filter = None
        self.arima_model = None

        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def load_data(self) -> None:
        """
        데이터 로드 및 전처리
        - ragged edge 문제를 고려한 데이터 처리
        """
        # 고빈도 지표 데이터 로드
        self.X = pd.read_csv(self.X_path, parse_dates=['date'], index_col='date')
        self.y = pd.read_csv(self.y_path, parse_dates=['date'], index_col='date')
        
        # 결측치 및 이상치 처리
        self.X = self.X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
        
        # 최신 데이터까지 인덱스 확장
        today = pd.Timestamp.today()
        if today > self.X.index.max():
            new_idx = pd.date_range(start=self.X.index.min(), end=today, freq='D')
            self.X = self.X.reindex(new_idx)
            self.y = self.y.reindex(new_idx)
            
        # Period 인덱스로 변환
        self.X.index = pd.DatetimeIndex(self.X.index).to_period('D')
        self.y.index = pd.DatetimeIndex(self.y.index).to_period('D')
        
        logger.info(f"데이터 로드 완료: {len(self.X)} 행, {len(self.X.columns)} 열")

    def select_num_factors(self, X_scaled: np.ndarray) -> int:
        """
        Bai-Ng IC를 사용한 요인 수 선택
        
        Args:
            X_scaled: 스케일링된 입력 데이터
            
        Returns:
            선택된 요인 수
        """
        eigenvalues = np.linalg.eigvals(np.cov(X_scaled.T))
        ic_values = []
        for k in range(1, min(10, X_scaled.shape[1]) + 1):
            unexplained_var = np.sum(eigenvalues[k:])
            penalty = k * np.log(X_scaled.shape[0]) / X_scaled.shape[0]
            ic = np.log(unexplained_var / X_scaled.shape[1]) + penalty
            ic_values.append(ic)
        return np.argmin(ic_values) + 1

    def extract_factors(self) -> pd.DataFrame:
        """
        PCA를 사용한 요인 추출
        
        Returns:
            추출된 요인 데이터프레임
        """
        # 결측치가 적은 변수만 선택
        valid_cols = self.X.columns[self.X.isna().mean() < 0.2]
        X_valid = self.X[valid_cols].ffill()
        
        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        # 요인 수 선택
        if self.n_factors is None:
            self.n_factors = self.select_num_factors(X_scaled)
            logger.info(f"Bai-Ng IC로 선택된 요인 수: {self.n_factors}")

        # PCA 수행
        self.pca_model = PCA(n_components=self.n_factors)
        Z = self.pca_model.fit_transform(X_scaled)
        Z_df = pd.DataFrame(Z, index=self.X.index, 
                          columns=[f'Factor_{i+1}' for i in range(self.n_factors)])
        
        return Z_df

    def fit_var(self, Z: pd.DataFrame) -> None:
        """
        VAR 모델 적합
        
        Args:
            Z: 요인 데이터프레임
        """
        self.var_model = VAR(Z)
        self.var_result = self.var_model.fit(1)
        logger.info("VAR 모델 적합 완료")

    def setup_kalman_filter(self, Z: pd.DataFrame) -> None:
        """
        Kalman Filter 설정
        
        Args:
            Z: 요인 데이터프레임
        """
        # ragged edge 마스크 생성
        ragged_mask = self.X.isna()
        R_diag = np.where(ragged_mask.any(axis=1), 1e6, 1e-3)
        
        # Kalman Filter 초기화
        self.kalman_filter = KalmanFilter(
            transition_matrices=self.var_result.coefs[0],
            transition_offsets=self.var_result.intercept,
            observation_matrices=np.eye(Z.shape[1]),
            observation_covariance=np.diag(R_diag),
            transition_covariance=np.diag(self.var_result.resid.cov().mean(axis=0))
        )
        logger.info("Kalman Filter 설정 완료")

    def smooth_factors(self, Z: pd.DataFrame) -> pd.DataFrame:
        """
        Kalman Smoother를 사용한 요인 보정
        
        Args:
            Z: 요인 데이터프레임
            
        Returns:
            보정된 요인 데이터프레임
        """
        smoothed_state, _ = self.kalman_filter.smooth(Z.values)
        return pd.DataFrame(smoothed_state, index=Z.index, columns=Z.columns)

    def fit_arima(self, y: pd.Series, Z: pd.DataFrame) -> None:
        """
        ARIMA-DFM 모델 적합
        
        Args:
            y: 목표 변수
            Z: 보정된 요인
        """
        # Ridge 회귀로 요인 가중치 추정
        y_train = y.dropna()
        X_train = Z.loc[y_train.index]
        
        model = Ridge()
        grid = GridSearchCV(model, param_grid={'alpha': np.logspace(-4, 2, 10)}, cv=3)
        grid.fit(X_train, y_train)
        
        # ARIMA 모델 적합
        self.arima_model = ARIMA(y_train, order=(1,1,1))
        self.arima_result = self.arima_model.fit()
        
        logger.info("ARIMA-DFM 모델 적합 완료")

    def fit(self) -> None:
        """
        전체 모델 적합 프로세스
        """
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 요인 추출
        Z = self.extract_factors()
        
        # 3. VAR 모델 적합
        self.fit_var(Z)
        
        # 4. Kalman Filter 설정
        self.setup_kalman_filter(Z)
        
        # 5. 요인 보정
        self.Z_smoothed = self.smooth_factors(Z)
        
        # 6. ARIMA-DFM 모델 적합
        self.fit_arima(self.y[self.target], self.Z_smoothed)

    def forecast_target(self, steps: int, alpha: float = 0.05) -> pd.DataFrame:
        """
        목표 변수 예측
        
        Args:
            steps: 예측 기간
            alpha: 신뢰구간 유의수준
            
        Returns:
            예측 결과 데이터프레임
        """
        # 1. 요인 예측
        Z_forecast = self.var_result.forecast(
            self.Z_smoothed.values[-self.var_result.k_ar:], 
            steps
        )
        Z_forecast = pd.DataFrame(
            Z_forecast, 
            columns=self.Z_smoothed.columns,
            index=pd.period_range(
                start=self.Z_smoothed.index[-1] + 1, 
                periods=steps, 
                freq='D'
            )
        )

        # 2. ARIMA 예측
        arima_forecast = self.arima_result.forecast(steps)
        
        # 3. 요인 기반 예측
        y_train = self.y[self.target].dropna()
        X_train = self.Z_smoothed.loc[y_train.index]
        
        model = Ridge()
        grid = GridSearchCV(model, param_grid={'alpha': np.logspace(-4, 2, 10)}, cv=3)
        grid.fit(X_train, y_train)
        
        factor_forecast = grid.predict(Z_forecast)
        
        # 4. 예측 결합
        forecast_mean = 0.5 * arima_forecast + 0.5 * factor_forecast
        
        # 5. 신뢰구간 계산
        residuals = y_train - grid.predict(X_train)
        sigma_hat = np.std(residuals)
        z_score = norm.ppf(1 - alpha / 2)
        
        lower_bound = forecast_mean - z_score * sigma_hat
        upper_bound = forecast_mean + z_score * sigma_hat

        # 6. 결과 데이터프레임 생성
        forecast_df = pd.DataFrame({
            'forecast': forecast_mean,
            'lower': lower_bound,
            'upper': upper_bound
        }, index=Z_forecast.index)

        return forecast_df

    def save_model(self, path: str) -> None:
        """
        모델 저장
        
        Args:
            path: 저장 경로
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'pca_model': self.pca_model,
            'n_factors': self.n_factors,
            'var_result': self.var_result,
            'kalman_filter': self.kalman_filter,
            'arima_result': self.arima_result
        }
        joblib.dump(model_data, path)
        logger.info(f"모델 저장 완료: {path}")

    def load_model(self, path: str) -> None:
        """
        모델 로드
        
        Args:
            path: 로드 경로
        """
        model_data = joblib.load(path)
        self.pca_model = model_data['pca_model']
        self.n_factors = model_data['n_factors']
        self.var_result = model_data['var_result']
        self.kalman_filter = model_data['kalman_filter']
        self.arima_result = model_data['arima_result']
        logger.info(f"모델 로드 완료: {path}")
