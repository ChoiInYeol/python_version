"""
base_model.py
모든 Nowcasting 모델의 기본 추상 클래스
"""
import pandas as pd
import numpy as np
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
from visualizer import DFMVisualizer # Visualizer는 공통적으로 사용될 수 있음

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("base_model")

class NowcastingBaseModel(ABC):
    """Nowcasting 모델의 추상 베이스 클래스."""
    def __init__(
        self,
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        model_type: str = 'base',
        data_name: str = 'unknown',
        output_dir: Optional[str] = None
    ):
        self.X_path = X_path
        self.y_path = y_path
        self.target = target
        self.model_type = model_type.lower()
        self.data_name = data_name
        self.output_dir = output_dir or f'output/nowcasts_{self.model_type}_{self.data_name}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.DataFrame] = None
        
        logger.info(f"Initializing {self.__class__.__name__} ({self.model_type}, {self.data_name}) with output directory: {self.output_dir}")

    def load_data(self) -> None:
        """X와 y 데이터를 로드하고, 기본 전처리 수행."""
        try:
            self.X = pd.read_csv(self.X_path, parse_dates=['date'], index_col='date')
            self.y = pd.read_csv(self.y_path, parse_dates=['date'], index_col='date')
            self.X = self.X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
            
            # 인덱스를 PeriodIndex로 변환 (일별 기준)
            self.X.index = pd.DatetimeIndex(self.X.index).to_period('D')
            self.y.index = pd.DatetimeIndex(self.y.index).to_period('D')
            
            # y 데이터에 target 컬럼이 있는지 확인
            if self.target not in self.y.columns:
                raise ValueError(f"Target column '{self.target}' not found in y data.")

            logger.info(f"Data loaded for {self.model_type}: X={self.X.shape}, y={self.y.shape}")
            
        except Exception as e:
            logger.error(f"데이터 로드 실패 ({self.model_type}): {str(e)}")
            raise

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """모델 학습."""
        pass

    @abstractmethod
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        예측 수행. 입력 X가 없으면 self.X 사용.
        반드시 일별 PeriodIndex를 가진 Series를 반환해야 함.
        """
        pass

    def export_nowcast_csv(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        전체 기간의 Nowcast와 Actual 값을 CSV로 저장하고, 발표일 사이의 평균값도 계산.
        
        Args:
            output_path: CSV 저장 경로 (None이면 self.output_dir 사용)
            
        Returns:
            결과 DataFrame
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'nowcasts.csv')
            
        if self.X is None or self.y is None:
            self.load_data()
        
        predictions = self.predict(self.X)
        
        nowcast = pd.DataFrame(index=self.X.index)
        nowcast['predicted'] = predictions.reindex(self.X.index)
        nowcast['actual'] = self.y[self.target].reindex(self.X.index)
        
        release_dates = nowcast[nowcast['actual'].notna()].index.sort_values()
        logger.info(f"[{self.model_type}] 식별된 릴리즈 날짜 수: {len(release_dates)}")
        
        period_avgs = pd.Series(index=nowcast.index, dtype=float)
        
        for i in range(len(release_dates)):
            current_release = release_dates[i]
            
            if i > 0:
                prev_release = release_dates[i-1]
                assign_mask = (nowcast.index > prev_release) & (nowcast.index <= current_release)
                calc_mask = (nowcast.index > prev_release) & (nowcast.index < current_release)
                period_predictions = nowcast.loc[calc_mask, 'predicted']
                
                if not period_predictions.empty and not period_predictions.isna().all():
                    avg_prediction = period_predictions.mean()
                    period_avgs.loc[assign_mask] = avg_prediction
                    logger.debug(f"[{self.model_type}] 기간 ({prev_release}, {current_release}) 의 평균 예측값 (계산 범위: {prev_release+1}~{current_release-1}): {avg_prediction:.4f}")
                else:
                    logger.debug(f"[{self.model_type}] 기간 ({prev_release}, {current_release}) 에 대한 평균 계산 데이터 없음.")
        
        if len(release_dates) > 0:
            last_release = release_dates[-1]
            future_mask = (nowcast.index > last_release)
            future_predictions = nowcast.loc[future_mask, 'predicted']
            
            if not future_predictions.empty and not future_predictions.isna().all():
                future_avg = future_predictions.mean()
                period_avgs.loc[future_mask] = future_avg
                logger.debug(f"[{self.model_type}] 마지막 발표일 {last_release} 이후 기간의 평균 예측값: {future_avg:.4f}")
            else:
                logger.debug(f"[{self.model_type}] 마지막 발표일 {last_release} 이후 기간 데이터 없음.")
        
        nowcast['period_avg'] = period_avgs
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nowcast.to_csv(output_path)
        logger.info(f"[{self.model_type}] Nowcast results saved to {output_path}")
            
        return nowcast

    def get_latest_monthly_nowcast(self) -> Optional[float]:
        """
        가장 최근 발표일 이후부터 최신 데이터까지의 일별 예측 평균을 계산하여
        현재 (또는 다음) 발표 기간에 대한 월간 Nowcast 값을 반환합니다.
        """
        if self.X is None or self.y is None:
            self.load_data()
            if self.X is None or self.y is None: return None

        daily_predictions = self.predict(self.X)
        if daily_predictions.empty:
            logger.warning(f"[{self.model_type}] 일별 예측값을 얻지 못했습니다.")
            return None

        valid_y = self.y[self.target].dropna()
        if valid_y.empty:
            logger.warning(f"[{self.model_type}] y 데이터에서 실제 발표일을 찾을 수 없습니다.")
            # 실제 발표일이 없으면 전체 예측 기간의 평균 반환 또는 마지막 값 반환
            return daily_predictions.mean() if not daily_predictions.empty else None

        last_release_date = valid_y.index.max()

        start_avg_date = last_release_date + 1
        end_avg_date = daily_predictions.index.max()

        if start_avg_date > end_avg_date:
            logger.warning(f"[{self.model_type}] 평균 계산 시작일({start_avg_date})이 마지막 예측일({end_avg_date})보다 늦습니다. 마지막 일별 예측값을 반환합니다.")
            return daily_predictions.iloc[-1] if not daily_predictions.empty else None

        period_predictions = daily_predictions.loc[start_avg_date:end_avg_date]

        if period_predictions.empty or period_predictions.isna().all():
            logger.warning(f"[{self.model_type}] {start_avg_date}부터 {end_avg_date} 사이의 유효한 예측값을 찾을 수 없습니다. 마지막 일별 예측값을 반환합니다.")
            return daily_predictions.iloc[-1] if not daily_predictions.empty else None

        monthly_avg_nowcast = period_predictions.mean()
        logger.info(f"[{self.model_type}] {start_avg_date}부터 {end_avg_date}까지의 일별 예측 평균 (월간 Nowcast): {monthly_avg_nowcast:.4f}")

        return monthly_avg_nowcast

    def plot_results(self, output_dir: Optional[str] = None) -> None:
        """
        Nowcast 결과 시각화 및 CSV 저장
        
        Args:
            output_dir: 출력 디렉토리 (None이면 self.output_dir 사용)
        """
        plot_output_dir = output_dir or self.output_dir
        os.makedirs(plot_output_dir, exist_ok=True)
        logger.info(f"[{self.model_type}] Starting plot_results in {plot_output_dir}")
        
        try:
            # CSV 저장 및 데이터 로드
            logger.info(f"[{self.model_type}] Exporting nowcast CSV...")
            nowcast_df = self.export_nowcast_csv(os.path.join(plot_output_dir, 'nowcasts.csv'))
            logger.info(f"[{self.model_type}] Nowcast CSV exported. Shape: {nowcast_df.shape}")
            
            # 데이터 유효성 검사 추가
            if nowcast_df.empty:
                logger.warning(f"[{self.model_type}] Exported nowcast_df is empty. Skipping plotting.")
                return
            if nowcast_df['predicted'].isna().all():
                 logger.warning(f"[{self.model_type}] All 'predicted' values in nowcast_df are NaN. Plotting might be limited.")
                 
            # 시각화 객체 생성 및 실행
            logger.info(f"[{self.model_type}] Initializing DFMVisualizer...")
            visualizer = DFMVisualizer(self.model_type, plot_output_dir)
            
            logger.info(f"[{self.model_type}] Calling plot_nowcast_results...")
            visualizer.plot_nowcast_results(nowcast_df.copy()) # 데이터 복사본 전달
            logger.info(f"[{self.model_type}] Finished plot_nowcast_results.")
            
            logger.info(f"[{self.model_type}] Calling plot_prediction_accuracy...")
            visualizer.plot_prediction_accuracy(nowcast_df.copy()) # 데이터 복사본 전달
            logger.info(f"[{self.model_type}] Finished plot_prediction_accuracy.")
            
            logger.info(f"[{self.model_type}] Result plots should be saved to {plot_output_dir}")
            
        except Exception as e:
            logger.error(f"[{self.model_type}] Error during plot_results: {e}", exc_info=True)

    def export_feature_importance(self, output_dir: Optional[str] = None) -> None:
        """
        (선택 사항) 변수 중요도 시각화 및 저장.
        모델별로 구현하거나, 중요도를 계산할 수 없으면 경고 메시지 출력.
        """
        plot_output_dir = output_dir or self.output_dir
        logger.warning(f"[{self.model_type}] Feature importance export not implemented for this model type.")
        # 모델별로 이 메서드를 오버라이드하여 구현 