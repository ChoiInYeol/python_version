"""
CPI 예측 모델 클래스 정의
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class CPIPredictor:
    def __init__(self, n_pca_comp=5, lags=10, start_date='2020-01-01'):
        """
        CPI 예측 모델 초기화
        
        Args:
            n_pca_comp: PCA 컴포넌트 수
            lags: VAR 모델의 시차
            start_date: 예측 시작 날짜
        """
        self.n_pca_comp = n_pca_comp
        self.lags = lags
        self.start_date = start_date
        self.pca = PCA(n_components=n_pca_comp)
        
    def prepare_data(self, input_df, target_df, lag_days=2):
        """
        데이터 전처리 및 PCA 변환
        
        Args:
            input_df: 입력 데이터
            target_df: 타겟 데이터
            lag_days: 지연 일수
            
        Returns:
            전처리된 데이터
        """
        # PCA 변환
        reduced_input = self.pca.fit_transform(input_df)
        reduced_df = pd.DataFrame(reduced_input, 
                                index=input_df.index,
                                columns=[f'PC{i+1}' for i in range(self.n_pca_comp)])
        
        # 타겟 데이터 래깅
        target_df = target_df.shift(lag_days)
        
        # 데이터 병합
        self.data = pd.concat([target_df, reduced_df], axis=1).dropna()
        self.data.index.freq = 'B'
        self.start_idx = self.data.index.get_loc(self.start_date)
        
        return self.data

    def train_model(self):
        """
        VAR 모델 학습
        
        Returns:
            bool: 학습 성공 여부
        """
        train_data = self.data.iloc[:self.start_idx]
        train_data.index.freq = 'B'
        
        try:
            var_model = VAR(train_data, freq='B')
            self.var_results = var_model.fit(maxlags=self.lags)
            logger.info(f"모델 학습 완료: AIC = {self.var_results.aic}")
            return True
        except Exception as e:
            logger.error(f"VAR 모델 학습 실패: {e}")
            return False

    def predict(self):
        """
        예측 수행
        
        Returns:
            DataFrame: 예측 결과
        """
        predictions = []
        actuals = []
        dates = []

        for i in range(self.start_idx, len(self.data)):
            forecast_input = self.data.values[i-self.lags:i]
            var_forecast = self.var_results.forecast(forecast_input, steps=1)
            cpi_pred = var_forecast[0, 0]

            # 변동성 보정
            residuals = self.var_results.resid['CPId']
            vol = residuals.std()
            cpi_pred_adjusted = cpi_pred + np.random.normal(0, vol)

            actual_value = self.data['CPId'].iloc[i]
            pred_date = self.data.index[i]
            
            predictions.append(cpi_pred_adjusted)
            actuals.append(actual_value)
            dates.append(pred_date)

        self.results_df = pd.DataFrame({
            'Date': dates,
            'Actual_CPI': actuals,
            'Predicted_CPI': predictions
        }).set_index('Date')
        
        return self.results_df

    def evaluate(self):
        """
        모델 성능 평가
        
        Returns:
            dict: 평가 지표
        """
        total_mae = mean_absolute_error(self.results_df['Actual_CPI'], 
                                      self.results_df['Predicted_CPI'])
        
        naive_pred = self.data['CPId'].shift(1).iloc[self.start_idx:]
        naive_actual = self.data['CPId'].iloc[self.start_idx:]
        mae_naive = mean_absolute_error(naive_actual, naive_pred)
        
        eval_results = {
            'Model MAE': total_mae,
            'Naive MAE': mae_naive,
            'CPI Std': self.data['CPId'].std(),
            'CPI Std * 0.2': self.data['CPId'].std() * 0.2
        }
        
        return eval_results 