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
        예측치와 실제치의 차이를 분석하고 시각화
        
        Args:
            nowcast_df: Nowcast 결과가 담긴 DataFrame
        """
        # 실제치와 예측치가 있는 데이터만 선택
        valid_data = nowcast_df.dropna(subset=['actual', 'predicted'])
        
        # 소수점 2째 자리에서 반올림
        actual_rounded = valid_data['actual'].round(2)
        predicted_rounded = valid_data['predicted'].round(2)
        
        # 차이의 절대값 계산
        differences = (actual_rounded - predicted_rounded).abs()
        
        # 0.05 미만을 성공으로 분류
        successful_predictions = (differences < 0.05).sum()
        failed_predictions = (differences >= 0.05).sum()
        
        # 차이 분포 계산 (성공/실패 구분)
        diff_ranges = pd.cut(differences, bins=[0, 0.05, 0.1, 0.2, 0.3, float('inf')], 
                           labels=['Success (<0.05)', '0.05-0.1', '0.1-0.2', '0.2-0.3', '>0.3'])
        diff_distribution = diff_ranges.value_counts().sort_index()
        
        # 시각화
        plt.style.use(['science', 'ieee', 'no-latex'])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
        
        # 1. 성공/실패 비율 파이 차트
        labels = ['Successful Predictions', 'Failed Predictions']
        sizes = [successful_predictions, failed_predictions]
        colors = ['#44AA44', '#FF4444']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Prediction Success Rate (Threshold: 0.05)')
        
        # 2. 차이 분포 막대 그래프
        diff_distribution.plot(kind='bar', ax=ax2, color='#4444FF')
        ax2.set_title('Distribution of Prediction Differences')
        ax2.set_xlabel('Difference Range')
        ax2.set_ylabel('Frequency')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_accuracy.png'), dpi=300)
        plt.savefig(os.path.join(self.output_dir, 'prediction_accuracy.svg'))
        plt.close()
        
        # 결과를 CSV 파일로 저장
        results_df = pd.DataFrame({
            'Category': ['Successful Predictions', 'Failed Predictions', 'Total'],
            'Frequency': [successful_predictions, failed_predictions, len(valid_data)],
            'Percentage(%)': [successful_predictions/len(valid_data)*100, failed_predictions/len(valid_data)*100, 100]
        })
        results_df.to_csv(os.path.join(self.output_dir, 'prediction_accuracy.csv'), index=False)
        
        # 차이 분포도 CSV로 저장
        diff_distribution.to_frame(name='Frequency').to_csv(os.path.join(self.output_dir, 'prediction_difference_distribution.csv'))
        
        logger.info(f"Prediction accuracy analysis saved to {self.output_dir}")

    def plot_nowcast_results(self, nowcast_df: pd.DataFrame) -> None:
        """
        Nowcast 결과와 Rolling RMSE를 시각화
        
        Args:
            nowcast_df: Nowcast 결과가 담긴 DataFrame
        """
        start_date = pd.Period('2020-01-01', freq='D')
        nowcast = nowcast_df[nowcast_df.index >= start_date]
        
        valid_mask = nowcast['actual'].notna()
        rmse_series = pd.Series(index=nowcast.index[valid_mask])
        for date in nowcast.index[valid_mask]:
            past_mask = (nowcast.index <= date) & valid_mask
            if past_mask.sum() >= 2:
                rmse = np.sqrt(mean_squared_error(
                    nowcast.loc[past_mask, 'actual'],
                    nowcast.loc[past_mask, 'predicted']
                ))
                rmse_series[date] = rmse
        
        plt.style.use(['science', 'ieee'])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], dpi=300)
        
        ax1.plot(nowcast.index.to_timestamp(), nowcast['predicted'], 
                label='Nowcast', color='#4444FF', linewidth=1)
        actual_data = nowcast['actual'].ffill()
        ax1.plot(nowcast.index.to_timestamp(), actual_data, 
                color='#FF4444', linewidth=1, linestyle='--', alpha=0.5)
        actual_points = nowcast[nowcast['actual'].notna()]
        ax1.scatter(actual_points.index.to_timestamp(), actual_points['actual'],
                   color='#FF4444', s=2, zorder=5, label='Actual', marker='^')
        nowcast_scatter_points = nowcast[nowcast['actual'].notna()]
        ax1.scatter(nowcast_scatter_points.index.to_timestamp(), nowcast_scatter_points['predicted'],
                   color='black', s=2, zorder=5, label='Nowcast', marker='*')
        
        ax1.legend(loc='upper right')
        ax1.set_title(f"Real-Time CPI Nowcasting ({self.model_type.capitalize()})")
        ax1.set_ylabel("CPI YoY Change (%)")
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.plot(rmse_series.index.to_timestamp(), rmse_series.values,
                color='#44AA44', linewidth=1.5, label='Rolling RMSE')
        ax2.legend(loc='upper right')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("RMSE")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'nowcast_plot.png'), dpi=300)
        plt.savefig(os.path.join(self.output_dir, 'nowcast_plot.svg'))
        plt.close()
        logger.info(f"Nowcast plot saved to {self.output_dir}")

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