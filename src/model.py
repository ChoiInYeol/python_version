"""
모델 모듈

이 모듈은 상태 공간 모델 및 예측 관련 기능을 제공합니다.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, ElasticNetCV

# 로깅 설정
logger = logging.getLogger(__name__)

def prepare_model_data(hist_transformed: pd.DataFrame, target_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    모델 학습을 위한 데이터를 준비합니다.
    
    Args:
        hist_transformed (pd.DataFrame): 변환된 역사적 데이터
        target_date (datetime): 목표 날짜
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X, y) 데이터
    """
    try:
        # 목표 변수 데이터 준비
        target_data = hist_transformed[
            (hist_transformed['varname'] == 'cpi') & 
            (hist_transformed['form'] == 'd1') &
            (hist_transformed['date'] <= target_date)
        ].copy()
        
        # 예측 변수 데이터 준비
        feature_data = hist_transformed[
            (hist_transformed['nc_dfm_input'] == 1) &
            (hist_transformed['date'] <= target_date)
        ].copy()
        
        # 데이터 피벗
        X = feature_data.pivot(
            index='date',
            columns=['varname', 'form'],
            values='value'
        )
        
        y = target_data.set_index('date')['value']
        
        # 결측치 처리
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')
        
        return X, y
        
    except Exception as e:
        logger.error(f"Failed to prepare model data: {str(e)}")
        raise

def train_pca_model(X: pd.DataFrame, n_components: int = 10) -> Tuple[PCA, pd.DataFrame]:
    """
    PCA 모델을 학습하고 변환된 데이터를 반환합니다.
    
    Args:
        X (pd.DataFrame): 입력 데이터
        n_components (int): PCA 컴포넌트 수
        
    Returns:
        Tuple[PCA, pd.DataFrame]: (PCA 모델, 변환된 데이터)
    """
    try:
        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA 모델 학습
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 변환된 데이터를 데이터프레임으로 변환
        X_pca_df = pd.DataFrame(
            X_pca,
            index=X.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return pca, X_pca_df
        
    except Exception as e:
        logger.error(f"Failed to train PCA model: {str(e)}")
        raise

def train_elastic_net(X: pd.DataFrame, y: pd.Series) -> ElasticNet:
    """
    ElasticNet 모델을 학습합니다.
    
    Args:
        X (pd.DataFrame): 입력 데이터
        y (pd.Series): 목표 변수
        
    Returns:
        ElasticNet: 학습된 모델
    """
    try:
        # 교차 검증을 통한 하이퍼파라미터 선택
        cv_model = ElasticNetCV(
            l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            eps=1e-3,
            n_alphas=100,
            cv=5,
            random_state=42
        )
        
        cv_model.fit(X, y)
        
        # 최적 하이퍼파라미터로 모델 학습
        model = ElasticNet(
            alpha=cv_model.alpha_,
            l1_ratio=cv_model.l1_ratio_,
            random_state=42
        )
        
        model.fit(X, y)
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to train ElasticNet model: {str(e)}")
        raise

def run_state_space_model(hist: Dict, bdates: List[datetime]) -> Dict:
    """
    상태 공간 모델을 실행합니다.
    
    Args:
        hist (Dict): 역사적 데이터
        bdates (List[datetime]): 백테스트 날짜 목록
        
    Returns:
        Dict: 학습된 모델들
    """
    logger.info("*** Running State-Space Model")
    
    try:
        models = {}
        
        # 각 백테스트 날짜에 대해 모델 학습 및 예측
        for target_date in bdates:
            logger.info(f"Processing target date: {target_date}")
            
            # 데이터 준비
            X, y = prepare_model_data(hist['transformed'], target_date)
            
            # PCA 모델 학습
            pca, X_pca = train_pca_model(X)
            
            # ElasticNet 모델 학습
            elastic_net = train_elastic_net(X_pca, y)
            
            # 모델 저장
            models[target_date] = {
                'pca': pca,
                'elastic_net': elastic_net,
                'X': X,
                'y': y,
                'X_pca': X_pca
            }
            
            logger.info(f"Model trained successfully for {target_date}")
        
        return models
            
    except Exception as e:
        logger.error(f"Failed to run state-space model: {str(e)}")
        raise

def generate_monthly_forecast(target_date: datetime, models: Dict, hist: Dict) -> Dict[str, float]:
    """
    월간 예측을 생성합니다.
    
    Args:
        target_date (datetime): 목표 날짜
        models (Dict): 학습된 모델들
        hist (Dict): 역사적 데이터
        
    Returns:
        Dict[str, float]: 예측 결과
    """
    try:
        # 해당 날짜의 모델 가져오기
        model_data = models[target_date]
        
        # 예측 변수 데이터 준비
        feature_data = hist['transformed'][
            (hist['transformed']['nc_dfm_input'] == 1) &
            (hist['transformed']['date'] <= target_date)
        ].copy()
        
        # 데이터 피벗
        X = feature_data.pivot(
            index='date',
            columns=['varname', 'form'],
            values='value'
        )
        
        # 결측치 처리
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # PCA 변환
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = model_data['pca'].transform(X_scaled)
        
        # 예측 수행
        forecast = model_data['elastic_net'].predict(X_pca[-1:])
        
        return {
            'date': target_date,
            'forecast': forecast[0],
            'actual': model_data['y'].iloc[-1] if target_date in model_data['y'].index else None
        }
        
    except Exception as e:
        logger.error(f"Failed to generate monthly forecast for {target_date}: {str(e)}")
        raise

def generate_quarterly_forecast(target_date: datetime, models: Dict, hist: Dict) -> Dict[str, float]:
    """
    분기 예측을 생성합니다.
    
    Args:
        target_date (datetime): 목표 날짜
        models (Dict): 학습된 모델들
        hist (Dict): 역사적 데이터
        
    Returns:
        Dict[str, float]: 예측 결과
    """
    try:
        # 해당 분기의 마지막 날짜 찾기
        quarter_end = target_date + pd.DateOffset(months=3) - pd.DateOffset(days=1)
        
        # 해당 날짜의 모델 가져오기
        model_data = models[target_date]
        
        # 예측 변수 데이터 준비
        feature_data = hist['transformed'][
            (hist['transformed']['nc_dfm_input'] == 1) &
            (hist['transformed']['date'] <= quarter_end)
        ].copy()
        
        # 데이터 피벗
        X = feature_data.pivot(
            index='date',
            columns=['varname', 'form'],
            values='value'
        )
        
        # 결측치 처리
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # PCA 변환
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = model_data['pca'].transform(X_scaled)
        
        # 예측 수행
        forecast = model_data['elastic_net'].predict(X_pca[-1:])
        
        return {
            'date': quarter_end,
            'forecast': forecast[0],
            'actual': model_data['y'].iloc[-1] if quarter_end in model_data['y'].index else None
        }
        
    except Exception as e:
        logger.error(f"Failed to generate quarterly forecast for {target_date}: {str(e)}")
        raise

def generate_forecasts(models: Dict, hist: Dict, bdates: List[datetime]) -> Dict:
    """
    예측을 생성합니다.
    
    Args:
        models (Dict): 학습된 모델들
        hist (Dict): 역사적 데이터
        bdates (List[datetime]): 백테스트 날짜 목록
        
    Returns:
        Dict: 생성된 예측 결과
    """
    logger.info("*** Generating Forecasts")
    
    try:
        forecasts = {}
        
        # 월간 예측 생성
        monthly_forecasts = []
        for target_date in bdates:
            try:
                forecast = generate_monthly_forecast(target_date, models, hist)
                monthly_forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Failed to generate monthly forecast for {target_date}: {str(e)}")
        
        if monthly_forecasts:
            forecasts['monthly'] = pd.DataFrame(monthly_forecasts)
            logger.info(f"Generated {len(monthly_forecasts)} monthly forecasts")
        
        # 분기 예측 생성
        quarterly_forecasts = []
        for target_date in bdates:
            try:
                forecast = generate_quarterly_forecast(target_date, models, hist)
                quarterly_forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Failed to generate quarterly forecast for {target_date}: {str(e)}")
        
        if quarterly_forecasts:
            forecasts['quarterly'] = pd.DataFrame(quarterly_forecasts)
            logger.info(f"Generated {len(quarterly_forecasts)} quarterly forecasts")
        
        return forecasts
        
    except Exception as e:
        logger.error(f"Failed to generate forecasts: {str(e)}")
        raise 