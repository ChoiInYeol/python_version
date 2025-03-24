"""
시각화 관련 함수 정의
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
import scienceplots

def plot_experiment_results(results, experiments, true_df):
    """
    실험 결과 시각화
    
    Args:
        results: 실험 결과 딕셔너리
        experiments: 실험 조건 리스트
        true_df: 실제 데이터
        
    Returns:
        tuple: (figure, axes)
    """
    plt.style.use(['science', 'ieee', 'no-latex'])
    years = results[experiments[0]['name']]['results'].index.year.unique()
    n_years = len(years)
    n_experiments = len(experiments)
    
    fig, axes = plt.subplots(n_years, n_experiments, 
                            figsize=(6*n_experiments, 5*n_years),
                            dpi=150)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, year in enumerate(years):
        for j, exp in enumerate(experiments):
            if exp['name'] not in results:
                continue
                
            plot_single_experiment(axes[i, j], results[exp['name']]['results'],
                                true_df, year, exp, i == 0)

    plt.suptitle('CPI Prediction Experiments Comparison\nVAR with PCA', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    return fig, axes

def plot_single_experiment(ax, exp_results, true_df, year, exp, is_first_row):
    """
    단일 실험 결과 플롯
    
    Args:
        ax: matplotlib axes 객체
        exp_results: 실험 결과 데이터
        true_df: 실제 데이터
        year: 연도
        exp: 실험 조건
        is_first_row: 첫 번째 행 여부
    """
    year_data = exp_results[exp_results.index.year == year]
    year_true = true_df[true_df.index.year == year]
    
    # 병합
    year_data = year_data.join(year_true, how='inner')
    
    # 라인 플롯
    ax.plot(year_data.index, year_data['Actual_CPI'], 
           label='Actual', zorder=3, linestyle='--', color='orange', linewidth=2)
    ax.plot(year_data.index, year_data['Predicted_CPI'], 
           label='Predicted', zorder=2, linestyle='-', color='blue', linewidth=1)
    ax.plot(year_data.index, year_true['CPI_YOY'],
           label='Official', zorder=1, linestyle='-.', color='red', linewidth=1)
    
    # 공시 포인트
    release_points = year_data[year_data['Release']]
    ax.scatter(release_points.index, release_points['CPI_YOY'],
              color='red', s=25, label='Release', zorder=5, marker='*')
    ax.scatter(release_points.index, release_points['Predicted_CPI'],
            color='blue', s=25, label='Release', zorder=5, marker='*')
    
    # 스타일링
    style_subplot(ax, year, exp, is_first_row)
    
    # MAE 표시
    year_mae = mean_absolute_error(year_data['Actual_CPI'], 
                                 year_data['Predicted_CPI'])
    ax.text(0.02, 0.95, f'MAE: {year_mae:.4f}', 
           transform=ax.transAxes,
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
           fontsize=8)

def style_subplot(ax, year, exp, is_first_row):
    """
    서브플롯 스타일링
    
    Args:
        ax: matplotlib axes 객체
        year: 연도
        exp: 실험 조건
        is_first_row: 첫 번째 행 여부
    """
    ax.legend(loc='upper right', fontsize=8)
    if is_first_row:
        ax.set_title(f"{exp['name']}\nReturn: {exp['return_period']}d, Lag: {exp['lag_days']}d", 
                   pad=20, fontsize=10)
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('CPI YoY (%)', fontsize=8)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=8) 