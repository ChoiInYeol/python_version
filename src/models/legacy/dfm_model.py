import pandas as pd
import numpy as np
import logging
import os
from typing import Optional
import matplotlib.pyplot as plt
import scienceplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DFMModel:
    def __init__(
        self, 
        data_path: str,
        cpi_release_path: str,
        target: str = 'cpi_yoy',
        window_size: int = 24,
        forecast_horizon: int = 40,
        start_date: Optional[str] = None
    ):
        self.data_path = data_path
        self.cpi_release_path = cpi_release_path
        self.target = target
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.df = None
        self.cpi_release_dates = None
        
    def load_data(self) -> None:
        try:
            self.df = pd.read_csv(self.data_path, parse_dates=['date'], index_col='date')
            cpi_release_df = pd.read_csv(self.cpi_release_path, parse_dates=['release_date'])
            cpi_release_df = cpi_release_df[cpi_release_df['release_date'] >= self.df.index.min()]
            self.cpi_release_dates = cpi_release_df['release_date'].values
            self.df = self.df.dropna()
            self.df.index = pd.DatetimeIndex(self.df.index).to_period('D')
            logger.info(f"데이터 로드 완료: {len(self.df)} 행, 기간: {self.df.index.min()} ~ {self.df.index.max()}")
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            raise
            
    def prepare_data(self, current_date: pd.Timestamp) -> pd.Series:
        if self.df is None:
            self.load_data()
            
        start_date = current_date - pd.offsets.MonthEnd(self.window_size)
        mask = (self.df.index >= start_date.to_period('D')) & (self.df.index <= current_date.to_period('D'))
        current_data = self.df[mask]
        
        logger.debug(f"prepare_data: current_date={current_date}, start_date={start_date}, 데이터 크기={len(current_data)}")
        
        if len(current_data) < 12:
            logger.warning(f"데이터 부족: current_date={current_date}, start_date={start_date}, 크기={len(current_data)}")
            earliest_date = self.df.index.min().to_timestamp()
            if current_date < (earliest_date + pd.offsets.MonthEnd(self.window_size)):
                current_date = earliest_date + pd.offsets.MonthEnd(self.window_size)
                start_date = current_date - pd.offsets.MonthEnd(self.window_size)
                mask = (self.df.index >= start_date.to_period('D')) & (self.df.index <= current_date.to_period('D'))
                current_data = self.df[mask]
                logger.info(f"current_date 조정: {current_date}, 데이터 크기={len(current_data)}")
        
        if len(current_data) < 12:
            raise ValueError(f"여전히 데이터 부족: current_date={current_date}, 크기={len(current_data)}")
            
        return current_data[self.target]
    
    def train(self, current_date: pd.Timestamp) -> None:
        self.train_y = self.prepare_data(current_date)
        logger.info(f"데이터 준비 완료: current_date={current_date}, 데이터 크기={len(self.train_y)}")
    
    def predict_daily(self, target_date: pd.Timestamp, current_date: pd.Timestamp) -> float:
        self.train(current_date)
        steps = (target_date - current_date).days
        if steps <= 0:
            steps = 1
        
        if len(self.train_y) >= 12:
            forecast = np.mean(self.train_y[-12:])
        else:
            forecast = np.mean(self.train_y)
        
        if np.isnan(forecast):
            logger.warning(f"예측값 NaN 발생: target_date={target_date}, current_date={current_date}, steps={steps}")
            forecast = self.train_y[-1] if len(self.train_y) > 0 else 0.0
        
        return forecast
    
    def nowcast_pipeline(self, target_date: pd.Timestamp) -> pd.DataFrame:
        results = []
        start_date = target_date - pd.Timedelta(days=self.forecast_horizon)
        target_date_period = target_date.to_period('D')
        actual_value = self.df.loc[target_date_period, self.target] if target_date_period in self.df.index else np.nan
        
        # target_date에 해당하는 CPI 발표일 찾기
        target_release_date = min([d for d in self.cpi_release_dates if d >= target_date], default=None)

        for current_date in pd.date_range(start_date, target_date, freq='D'):
            # target_date에 대한 CPI 발표 여부
            cpi_released_target = target_release_date is not None and current_date >= target_release_date
            prediction_target = self.predict_daily(target_date, current_date)
            actual_target = actual_value if cpi_released_target else np.nan
            
            results.append({
                'date': current_date,
                'target_date': target_date,
                'actual_target': actual_target,
                'predicted_target': prediction_target,
                'cpi_released_target': cpi_released_target
            })
        
        return pd.DataFrame(results)
    
    def evaluate(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
            
        min_date = self.df.index.min().to_timestamp() + pd.offsets.MonthEnd(self.window_size)
        if self.start_date:
            min_date = max(min_date, self.start_date)
        valid_release_dates = [d for d in self.cpi_release_dates if d >= min_date]
        
        results = []
        for release_date in valid_release_dates[1:]:
            release_date = pd.Timestamp(release_date)
            logger.info(f"평가 진행 중: release_date={release_date}")
            nowcast_df = self.nowcast_pipeline(release_date)
            results.append(nowcast_df)
        
        results_df = pd.concat(results)
        valid_df = results_df.dropna(subset=['actual_target'])
        rmse_target = np.sqrt(np.mean((valid_df['actual_target'] - valid_df['predicted_target']) ** 2))
        logger.info(f"Target RMSE: {rmse_target:.4f}")
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame) -> None:
        
        # SciencePlot 스타일 설정
        plt.style.use(['science', 'ieee'])
        
        # 각 날짜별로 가장 최근의 예측치만 선택
        latest_predictions = results_df.sort_values('date').groupby('date').last().reset_index()
        
        # 그래프 크기 및 DPI 설정
        plt.figure(figsize=(12, 6), dpi=300)
        
        # y축 범위 계산
        y_min = latest_predictions['predicted_target'].min()
        y_max = latest_predictions['predicted_target'].max()
        y_range = y_max - y_min
        y_min = y_min - y_range * 0.1  # 여유 공간 추가
        y_max = y_max + y_range * 0.1
        
        # 날짜별 예측치 플롯
        plt.plot(latest_predictions['date'], latest_predictions['predicted_target'], 
                label='Nowcast', color='#4444FF', alpha=0.7, linewidth=2)
        
        # 발표일(cpi_released_target이 True인 날)에 별표 표시
        release_dates = results_df[results_df['cpi_released_target']]['date'].unique()
        for date in release_dates:
            pred_value = latest_predictions[latest_predictions['date'] == date]['predicted_target'].iloc[0]
            plt.scatter(date, pred_value, 
                      color='#4444FF', s=100, zorder=5, marker='*')
            plt.annotate(f'Pred: {pred_value:.2f}%',
                       (date, pred_value),
                       xytext=(10, -10), textcoords='offset points',
                       fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 실제값 표시
        for _, row in results_df.iterrows():
            if row['date'] == row['target_date'] and not pd.isna(row['actual_target']):
                plt.scatter(row['date'], row['actual_target'], 
                          color='#FF4444', s=120, zorder=5)
                plt.annotate(f'Actual: {row["actual_target"]:.2f}%',
                           (row['date'], row['actual_target']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # y축 범위 설정
        plt.ylim(y_min, y_max)
        
        # 그리드 추가
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 범례 설정
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # 제목 및 레이블 설정
        plt.title("Real-Time CPI Nowcasting", pad=20, fontsize=12)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("CPI YoY Change (\%)", fontsize=10)
        
        # x축 레이블 회전 및 간격 조정
        plt.xticks(rotation=45, ha='right')
        
        # 여백 조정
        plt.tight_layout()
        
        # 결과 저장
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/nowcast_plot.svg', format='svg', bbox_inches='tight')
        plt.savefig('output/nowcast_plot.png', format='png', bbox_inches='tight', dpi=300)
        results_df.to_csv('output/nowcasts.csv', index=False)
        logger.info("예측 결과가 output/nowcasts.csv와 output/nowcast_plot.svg, output/nowcast_plot.png에 저장되었습니다.")
        plt.close()

def main():
    model = DFMModel(
        data_path='data/processed/processed_data.csv',
        cpi_release_path='data/processed/cpi_release_date.csv',
        target='cpi_yoy',
        start_date='2020-06-01',
        forecast_horizon=30,
        window_size=12
    )
    results_df = model.evaluate()
    model.plot_results(results_df)

if __name__ == "__main__":
    main()