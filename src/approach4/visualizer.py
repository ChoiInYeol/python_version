"""
DFM 모델 결과 시각화 모듈
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots
from typing import Tuple, List

def plot_forecast(forecast_df: pd.DataFrame, true_df: pd.Series, 
                 factors: pd.DataFrame = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    DFM 모델의 예측 결과와 요인 시각화
    
    Args:
        forecast_df: forecast, lower, upper 포함된 DataFrame
        true_df: 실제 CPI YOY 시계열
        factors: 요인 데이터프레임 (선택사항)
    
    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: 생성된 그래프와 축 객체
    """
    plt.style.use(['science', 'ieee', 'no-latex'])
    
    # 서브플롯 개수 결정
    n_subplots = 2 if factors is not None else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4*n_subplots), dpi=150)
    if n_subplots == 1:
        axes = [axes]
    
    # 1. 예측 결과 플롯
    ax = axes[0]
    
    # Forecast 및 신뢰구간
    ax.plot(forecast_df.index.to_timestamp(), forecast_df['forecast'], 
            label='Forecast', color='blue')
    ax.fill_between(forecast_df.index.to_timestamp(),
                    forecast_df['lower'], forecast_df['upper'],
                    color='blue', alpha=0.2, label='95% CI')

    # 실제 값 (공식 발표일 기준)
    actual = true_df[true_df.index >= forecast_df.index.min()]
    ax.plot(actual.index.to_timestamp(), actual.values, 
            label='Actual CPI YoY', linestyle='--', color='orange')

    # 스타일링
    ax.legend()
    ax.set_title('CPI YoY Forecast with 95% Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('CPI YoY (%)')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 요인 플롯 (있는 경우)
    if factors is not None:
        ax = axes[1]
        
        # 요인 시계열 플롯
        for col in factors.columns:
            ax.plot(factors.index.to_timestamp(), factors[col], 
                   label=col, alpha=0.7)
        
        # 스타일링
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title('Extracted Factors')
        ax.set_xlabel('Date')
        ax.set_ylabel('Factor Value')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig, axes

def plot_factor_contribution(factors: pd.DataFrame, 
                           pca_model: object) -> plt.Figure:
    """
    요인의 원본 변수 기여도 시각화
    
    Args:
        factors: 요인 데이터프레임
        pca_model: PCA 모델 객체
    
    Returns:
        plt.Figure: 생성된 그래프
    """
    plt.style.use(['science', 'ieee', 'no-latex'])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 요인 기여도 계산
    contribution = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'Factor_{i+1}' for i in range(pca_model.n_components_)],
        index=pca_model.feature_names_in_
    )
    
    # 히트맵 플롯
    im = ax.imshow(contribution, aspect='auto', cmap='RdBu_r')
    
    # 컬러바
    plt.colorbar(im, ax=ax)
    
    # 축 레이블
    ax.set_xticks(np.arange(len(contribution.columns)))
    ax.set_yticks(np.arange(len(contribution.index)))
    ax.set_xticklabels(contribution.columns)
    ax.set_yticklabels(contribution.index)
    
    # 제목
    ax.set_title('Factor Contribution to Original Variables')
    
    # 레이블 회전
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig
