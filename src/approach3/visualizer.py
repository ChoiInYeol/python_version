"""
visualizer.py
시각화 관련 기능을 담당하는 모듈
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
from typing import Optional
from sklearn.metrics import mean_squared_error
import logging
from scipy.stats import norm

logger = logging.getLogger("visualizer")

class DFMVisualizer:
    def __init__(self, model_type: str, output_dir: str):
        """
        DFM 모델의 시각화를 담당하는 클래스
        
        Args:
            model_type: 모델 타입 ('elasticnet', 'xgboost', 'lightgbm')
            output_dir: 출력 디렉토리 경로
        """
        self.model_type = model_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_prediction_accuracy(self, nowcast_df: pd.DataFrame) -> None:
        """
        예측 오차 분포를 분석하고 히스토그램으로 시각화
        
        Args:
            nowcast_df: Nowcast 결과가 담긴 DataFrame
        """
        logger.info(f"[{self.model_type}] Starting plot_prediction_accuracy...")
        # 시각화 스타일 설정
        plt.style.use(['science', 'ieee', 'no-latex'])
        
        # 데이터 유효성 검사
        if nowcast_df is None or nowcast_df.empty or 'actual' not in nowcast_df.columns or 'predicted' not in nowcast_df.columns:
            logger.error(f"[{self.model_type}] plot_prediction_accuracy: Invalid input dataframe.")
            return
            
        # 실제 발표일 데이터 선택
        valid_data = nowcast_df[nowcast_df['actual'].notna() & nowcast_df['predicted'].notna()].copy()
        
        if valid_data.empty:
            logger.warning(f"[{self.model_type}] No valid data points with both actual and predicted values found. Skipping error analysis.")
            
            # 빈 그래프 생성 및 안내 메시지 표시
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No actual release dates found. Cannot compute prediction errors.", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Prediction Error Analysis (No Data)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'prediction_error_distribution.png'))
            plt.close()
            return
            
        logger.debug(f"[{self.model_type}] plot_prediction_accuracy valid_data shape: {valid_data.shape}")
        
        try:
            # 일별 예측 오차와 period_avg 오차 계산 (실제치 - 예측치)
            daily_errors = valid_data['actual'] - valid_data['predicted']
            
            # 서브플롯 생성 (2개의 히스토그램)
            fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=300)
            plt.subplots_adjust(hspace=0.3)
            
            # 1. 일별 예측 오차 히스토그램
            logger.debug(f"[{self.model_type}] Plotting daily error histogram...")
            self._plot_error_histogram(axes[0], daily_errors, "Daily Nowcast", "blue")
            
            # 2. period_avg 오차 히스토그램 (period_avg 열이 있는 경우)
            if 'period_avg' in valid_data.columns:
                logger.debug(f"[{self.model_type}] Plotting period average error histogram...")
                period_avg_mask = valid_data['period_avg'].notna()
                if period_avg_mask.any():
                    period_avg_errors = valid_data.loc[period_avg_mask, 'actual'] - valid_data.loc[period_avg_mask, 'period_avg']
                    self._plot_error_histogram(axes[1], period_avg_errors, "Period Average", "green")
                    
                    # 유효한 오차 데이터만 필터링
                    valid_period_avg_errors = period_avg_errors.dropna()
                    if len(valid_period_avg_errors) >= 2:
                        # 오차 데이터를 CSV 파일로 저장
                        error_stats_period = pd.DataFrame({
                            'Metric': ['Mean Error', 'Std Dev Error', 'RMSE', 'Num Samples'],
                            'Value': [
                                valid_period_avg_errors.mean(), 
                                valid_period_avg_errors.std(), 
                                np.sqrt(np.mean(valid_period_avg_errors**2)), 
                                len(valid_period_avg_errors)
                            ]
                        })
                        error_stats_period.to_csv(os.path.join(self.output_dir, 'prediction_error_stats_period_avg.csv'), index=False)
                        valid_period_avg_errors.to_frame(name='Period Avg Error').to_csv(os.path.join(self.output_dir, 'prediction_errors_period_avg.csv'))
                else:
                    axes[1].text(0.5, 0.5, "No valid period average data available", 
                                ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title("Period Average Error Distribution - No Valid Data")
            else:
                axes[1].text(0.5, 0.5, "Period average column not found in data", 
                            ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title("Period Average Error Distribution - Missing Data")
            
            plt.suptitle(f'Prediction Error Distributions (Release Dates Only) - {self.model_type.upper()}', 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            
            # 저장
            save_path_png = os.path.join(self.output_dir, 'prediction_error_distribution.png')
            save_path_svg = os.path.join(self.output_dir, 'prediction_error_distribution.svg')
            plt.savefig(save_path_png)
            plt.savefig(save_path_svg, format='svg')
            plt.close(fig)
            logger.info(f"[{self.model_type}] Prediction accuracy plots saved to {save_path_png} and {save_path_svg}")
            
            # 일별 예측 오차 데이터 저장 (유효한 데이터가 있는 경우에만)
            valid_daily_errors = daily_errors.dropna()
            if len(valid_daily_errors) >= 2:
                error_stats_daily = pd.DataFrame({
                    'Metric': ['Mean Error', 'Std Dev Error', 'RMSE', 'Num Samples'],
                    'Value': [
                        valid_daily_errors.mean(), 
                        valid_daily_errors.std(), 
                        np.sqrt(np.mean(valid_daily_errors**2)), 
                        len(valid_daily_errors)
                    ]
                })
                error_stats_daily.to_csv(os.path.join(self.output_dir, 'prediction_error_stats_daily.csv'), index=False)
                valid_daily_errors.to_frame(name='Daily Prediction Error').to_csv(os.path.join(self.output_dir, 'prediction_errors_daily.csv'))
        
        except Exception as e:
            logger.error(f"[{self.model_type}] Error plotting prediction accuracy: {e}", exc_info=True)
            if 'fig' in locals() and fig is not None:
                 plt.close(fig)

    def _plot_error_histogram(self, ax, errors, label, color):
        """
        오차 히스토그램을 그리는 헬퍼 함수
        
        Args:
            ax: matplotlib 축 객체
            errors: 오차 시리즈
            label: 그래프 레이블
            color: 그래프 색상
        """
        # NaN 값 필터링
        valid_errors = errors.dropna()
        
        if len(valid_errors) < 2:
            # 데이터가 부족하면 메시지 표시
            ax.text(0.5, 0.5, f"Insufficient data for {label} error histogram", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Error Distribution - No Valid Data')
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Density')
            return
        
        # 통계량 계산 (NaN 제외)
        mean_error = valid_errors.mean()
        std_error = valid_errors.std()
        rmse = np.sqrt(np.mean(valid_errors**2))
        
        # 히스토그램 그리기
        try:
            n, bins, patches = ax.hist(valid_errors, bins=50, density=True, alpha=0.7, 
                                      color=color, label=f'{label} Error')
            
            # 정규분포 곡선 추가 (표준편차가 0 이상일 때만)
            if std_error > 0:
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mean_error, std_error)
                ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
            else:
                 logger.warning(f"Standard deviation is zero for {label} errors. Skipping normal distribution curve.")
            
            # 플롯 설정
            ax.set_title(f'{label} Error Distribution\nMean={mean_error:.3f}, StdDev={std_error:.3f}, RMSE={rmse:.3f}')
        except ValueError as ve:
             # Range 오류 처리 (모든 값이 동일한 경우 등)
            logger.error(f"ValueError creating histogram for {label}: {str(ve)}. Data might be constant.")
            ax.text(0.5, 0.5, f"Could not plot histogram: {str(ve)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Error Distribution - Plotting Error')
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"Error creating histogram for {label}: {str(e)}", exc_info=True)
            ax.text(0.5, 0.5, f"Error creating histogram: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Error Distribution - Error')
            
        ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=8)

    def plot_nowcast_results(self, nowcast_df: pd.DataFrame) -> None:
        """Nowcast와 Actual 비교 플롯 생성."""
        logger.info(f"[{self.model_type}] Starting plot_nowcast_results...")
        # 시각화 스타일 설정
        plt.style.use(['science', 'ieee', 'no-latex'])
        
        # 데이터 유효성 검사
        if nowcast_df is None or nowcast_df.empty:
            logger.error(f"[{self.model_type}] plot_nowcast_results: Input dataframe is empty or None.")
            return
        logger.debug(f"[{self.model_type}] plot_nowcast_results input df shape: {nowcast_df.shape}, head:\n{nowcast_df.head()}")

        # 2020년 이후의 데이터만 선택
        start_date = pd.Period('2020-01-01', freq='D')
        filtered_df = nowcast_df[nowcast_df.index >= start_date].copy()
        
        # 필터링 후 데이터가 없는 경우
        if filtered_df.empty:
            logger.warning("2020년 이후 데이터가 없습니다.")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No data available after 2020 for plotting.", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Nowcast Results - No Data After 2020")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'nowcast_results.png'))
            plt.close()
            return
        
        # 예측값이 모두 NaN인지 확인
        if 'predicted' not in filtered_df.columns or filtered_df['predicted'].isna().all():
            logger.warning(f"[{self.model_type}] No valid 'predicted' data after filtering. Skipping plot.")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "All prediction values are NaN.", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Nowcast Results - No Valid Predictions")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'nowcast_results.png'))
            plt.close()
            return
        
        try:
            # 연도별로 데이터 분리
            years = filtered_df.index.year.unique()
            n_years = len(years)
            
            # 서브플롯 생성
            fig, axes = plt.subplots(n_years, 1, figsize=(15, 5*n_years), dpi=300)
            if n_years == 1:  # 단일 연도인 경우 axes를 리스트로 변환
                axes = [axes]
            plt.subplots_adjust(hspace=0.4)
            
            # 각 연도별로 플롯
            for i, year in enumerate(years):
                ax = axes[i]
                year_data = filtered_df[filtered_df.index.year == year]
                dates = year_data.index.to_timestamp()
                
                # 유효한 예측값 확인
                valid_pred = year_data['predicted'].replace([np.inf, -np.inf], np.nan).dropna()
                if valid_pred.empty:
                    ax.text(0.5, 0.5, f"No valid predictions for {year}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'CPI YoY Nowcast Results ({self.model_type.upper()}) - {year} - No Data')
                    continue
                
                logger.debug(f"[{self.model_type}] Plotting year {year}...")
                # 일별 예측치 플롯 (유효한 데이터만)
                ax.plot(dates[year_data['predicted'].notna()], year_data['predicted'].dropna(), 
                        label='Daily Nowcast', color='blue', alpha=0.5, linewidth=1, zorder=1)
                
                # 실제값 처리
                valid_actual = year_data['actual'].replace([np.inf, -np.inf], np.nan).dropna()
                if not valid_actual.empty:
                    actual_ffill = year_data['actual'].ffill()
                    # 실제값 플롯 (ffill된 전체 데이터를 그림)
                    ax.plot(dates, actual_ffill, 
                            label='Actual', color='red', linewidth=2, zorder=4, drawstyle='steps-post')
                    
                    # 발표일 포인트 표시 (기존 로직 유지)
                    release_mask = year_data['actual'].notna()
                    if release_mask.any():
                        release_dates_ts = year_data[release_mask].index.to_timestamp()
                        release_values = year_data.loc[release_mask, 'actual']
                        ax.scatter(release_dates_ts, release_values, 
                                    color='red', s=50, marker='*', zorder=5, label='Release Dates')
                
                # Period Average 처리
                if 'period_avg' in year_data.columns:
                     valid_period_avg = year_data['period_avg'].replace([np.inf, -np.inf], np.nan).dropna()
                     if not valid_period_avg.empty and 'actual' in year_data.columns:
                        actual_mask = year_data['actual'].notna()
                        if actual_mask.any():
                            period_values_on_release = year_data.loc[actual_mask, 'period_avg']
                            valid_period_mask = period_values_on_release.notna()
                            if valid_period_mask.any():
                                period_dates = year_data[actual_mask].index[valid_period_mask].to_timestamp()
                                period_values = period_values_on_release[valid_period_mask]
                                ax.scatter(period_dates, period_values, 
                                        label='Period Average', color='green', 
                                        s=30, zorder=4, alpha=1.0, marker='o')
                
                # MAE 계산 및 표시
                valid_data = year_data.dropna(subset=['actual', 'predicted'])
                if len(valid_data) >= 2:
                    mae = np.mean(np.abs(valid_data['actual'] - valid_data['predicted']))
                    ax.text(0.02, 0.95, f'MAE: {mae:.3f}', 
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                           fontsize=8)
                
                # 스타일링
                ax.set_title(f'CPI YoY Nowcast Results ({self.model_type.upper()}) - {year}')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('CPI YoY (%)', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(loc='upper right', fontsize=8)
                
                # x축 날짜 포맷 및 간격 설정
                ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # 최대 12개 틱 표시
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
            
            plt.suptitle('CPI YoY Nowcast Results by Year', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # 저장
            save_path_png = os.path.join(self.output_dir, 'nowcast_results.png')
            save_path_svg = os.path.join(self.output_dir, 'nowcast_results.svg')
            plt.savefig(save_path_png)
            plt.savefig(save_path_svg, format='svg')
            plt.close(fig) # 명시적으로 figure 닫기
            logger.info(f"[{self.model_type}] Nowcast results plot saved to {save_path_png} and {save_path_svg}")

        except Exception as e:
            logger.error(f"[{self.model_type}] Error plotting nowcast results: {e}", exc_info=True)
            if 'fig' in locals() and fig is not None:
                 plt.close(fig) # 오류 발생 시 figure 닫기 시도

    def plot_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """
        변수 중요도를 시각화
        
        Args:
            importance_df: 변수 중요도가 담긴 DataFrame
        """
        if importance_df.empty:
            logger.warning("Feature importance 데이터가 없습니다.")
            return
            
        importance_df.index = importance_df.index.str.replace('%', 'pct')
        importance_df.to_frame(name='Importance').to_csv(os.path.join(self.output_dir, 'feature_importance.csv'))
        
        plt.style.use(['science', 'ieee'])
        plt.figure(figsize=(10,6), dpi=300)
        importance_df.plot(kind='bar')
        plt.title(f"Feature Importance ({self.model_type.capitalize()})")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.svg'))
        plt.close()
        logger.info(f"Feature importance plot saved to {self.output_dir}") 