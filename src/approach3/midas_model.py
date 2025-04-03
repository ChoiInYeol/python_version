"""
midas_model.py
MIDAS (Mixed-Data Sampling) approach for CPI Nowcasting
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import logging
from typing import Optional, Dict, List, Tuple
from base_model import NowcastingBaseModel
from visualizer import DFMVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("midas_model")

class MIDASModel(NowcastingBaseModel):
    def __init__(
        self,
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        lookback_periods: int = 12,  # Monthly periods to look back
        poly_degree: int = 2,        # Polynomial degree for weight function
        max_lags: Dict[str, int] = None,
        train_window_size: int = 730,  # Days
        data_name: str = 'unknown',
        output_dir: Optional[str] = None
    ):
        super().__init__(
            X_path=X_path,
            y_path=y_path,
            target=target,
            model_type='midas',
            data_name=data_name,
            output_dir=output_dir
        )
        
        self.lookback_periods = lookback_periods
        self.poly_degree = poly_degree
        self.max_lags = max_lags or {'default': 30}
        self.train_window_size = train_window_size
        self.X_freq = 'D'
        self.y_freq = 'D'
        self.coefficients: Dict = {}
        self.selected_features: List[str] = []
        self.scaler = StandardScaler()
        
    def load_data(self) -> None:
        super().load_data()
        self.y_freq = 'D'
        self.X_freq = 'D'
        logger.info(f"Data loaded for MIDAS: X({self.X_freq})={self.X.shape}, y({self.y_freq})={self.y.shape}")
    
    def _almon_weights(self, params: np.ndarray, lags: int) -> np.ndarray:
        weights = np.zeros(lags)
        for i in range(lags):
            norm_i = i / (lags - 1) if lags > 1 else 0
            weight = 0
            for j, param in enumerate(params):
                weight += param * (norm_i ** j)
            weights[i] = np.exp(weight)
        
        return weights / weights.sum() if weights.sum() > 0 else weights
    
    def _aggregate_high_freq_data(self, X_high_freq: pd.DataFrame, 
                                 target_dates: pd.DatetimeIndex,
                                 feature: str,
                                 params: np.ndarray) -> pd.Series:
        max_lag = None
        for prefix, lag in self.max_lags.items():
            if feature.startswith(prefix):
                max_lag = lag
                break
        if max_lag is None:
            max_lag = self.max_lags.get('default', 30)
        
        weights = self._almon_weights(params, max_lag)
        
        result = pd.Series(index=target_dates, dtype=float)
        
        for target_date in target_dates:
            data_before = X_high_freq[X_high_freq.index <= target_date].iloc[-max_lag:]
            
            if len(data_before) > 0:
                weighted_values = data_before.values[::-1][:len(weights)] * weights[:len(data_before)]
                result[target_date] = weighted_values.sum()
            else:
                result[target_date] = np.nan
                
        return result
    
    def _objective_func(self, params: np.ndarray, X_train: pd.DataFrame, 
                       y_train: pd.Series, feature: str) -> float:
        aggregated = self._aggregate_high_freq_data(
            X_train[[feature]], y_train.index, feature, params
        )
        
        valid_indices = ~aggregated.isna() & ~y_train.isna()
        if valid_indices.sum() < 2:
            return np.inf
            
        mse = ((aggregated[valid_indices] - y_train[valid_indices]) ** 2).mean()
        return mse
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> List[str]:
        correlations = {}
        
        for feature in X.columns:
            if X[feature].isna().mean() > 0.2:
                continue
                
            try:
                initial_params = np.zeros(self.poly_degree + 1)
                initial_params[0] = 0
                
                result = minimize(
                    self._objective_func,
                    initial_params,
                    args=(X[[feature]], y, feature),
                    method='BFGS',
                    options={'maxiter': 100}
                )
                
                if result.success:
                    aggregated = self._aggregate_high_freq_data(
                        X[[feature]], y.index, feature, result.x
                    )
                    
                    valid_indices = ~aggregated.isna() & ~y.isna()
                    if valid_indices.sum() >= 5:
                        corr = np.abs(np.corrcoef(
                            aggregated[valid_indices], 
                            y[valid_indices]
                        )[0, 1])
                        
                        if not np.isnan(corr):
                            correlations[feature] = corr
            except Exception as e:
                logger.warning(f"Error processing feature {feature}: {str(e)}")
                continue
        
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Top features by correlation: {top_features[:n_features]}")
        
        return [f[0] for f in top_features[:n_features]]
    
    def fit(self, n_features: int = 10, **kwargs) -> None:
        if self.X is None or self.y is None:
            self.load_data()
            if self.X is None or self.y is None: 
                logger.error("Data loading failed, cannot fit model.")
                return

        end_date = self.y[self.target].dropna().index.max()
        train_start = end_date - pd.Timedelta(days=self.train_window_size)
        
        mask_X = (self.X.index >= train_start) & (self.X.index <= end_date)
        X_train_full = self.X.loc[mask_X].copy()
        mask_y = (self.y.index >= train_start) & (self.y.index <= end_date)
        y_train_full = self.y[self.target].loc[mask_y].copy().dropna()
        X_train_full = X_train_full.loc[y_train_full.index]

        if X_train_full.empty or y_train_full.empty:
            logger.error("학습 데이터 부족")
            raise ValueError("학습 데이터 부족")

        self.selected_features = self._select_features(X_train_full, y_train_full, n_features)
        logger.info(f"Selected features: {self.selected_features}")

        X_scaled = self.scaler.fit_transform(X_train_full[self.selected_features].fillna(0))
        X_scaled_df = pd.DataFrame(X_scaled, index=X_train_full.index, columns=self.selected_features)

        self.coefficients = {}
        for feature in self.selected_features:
            try:
                initial_params = np.zeros(self.poly_degree + 1)
                result = minimize(
                    self._objective_func,
                    initial_params,
                    args=(X_scaled_df[[feature]], y_train_full, feature),
                    method='BFGS', options={'maxiter': 200}
                )
                if result.success:
                    self.coefficients[feature] = result.x
            except Exception as e:
                logger.warning(f"Error training weights for {feature}: {str(e)}")

        aggregated_features = pd.DataFrame(index=y_train_full.index)
        for feature in self.selected_features:
            if feature in self.coefficients:
                target_dates_dt = pd.DatetimeIndex(y_train_full.index.to_timestamp())
                aggregated = self._aggregate_high_freq_data(
                    X_scaled_df[[feature]], target_dates_dt, feature, self.coefficients[feature]
                )
                aggregated.index = pd.PeriodIndex(aggregated.index, freq='D')
                aggregated_features[feature] = aggregated.reindex(y_train_full.index)
        
        valid_indices = aggregated_features.dropna().index
        if len(valid_indices) < len(self.selected_features) + 1:
             logger.error("Not enough valid aggregated data for regression.")
             raise ValueError("회귀 분석 위한 데이터 부족")

        X_reg = aggregated_features.loc[valid_indices]
        y_reg = y_train_full.loc[valid_indices]
        
        X_reg_sm = sm.add_constant(X_reg)
        model = sm.OLS(y_reg, X_reg_sm)
        results = model.fit()
        
        self.coefficients['regression'] = {
            'const': results.params['const'],
            'features': {feature: results.params[feature] for feature in self.selected_features 
                         if feature in results.params}
        }
        logger.info(f"MIDAS Regression results:\n{results.summary().tables[1]}")

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        if X is None:
            if self.X is None: self.load_data()
            X = self.X

        if not isinstance(X.index, pd.PeriodIndex):
            X.index = pd.DatetimeIndex(X.index).to_period('D')

        X_scaled = self.scaler.transform(X[self.selected_features].fillna(0))
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=self.selected_features)

        target_dates = X.index
        aggregated_features = pd.DataFrame(index=target_dates)
        
        for feature in self.selected_features:
            if feature in self.coefficients:
                target_dates_dt = pd.DatetimeIndex(target_dates.to_timestamp())
                aggregated = self._aggregate_high_freq_data(
                    X_scaled_df[[feature]], target_dates_dt, feature, self.coefficients[feature]
                )
                aggregated.index = pd.PeriodIndex(aggregated.index, freq='D')
                aggregated_features[feature] = aggregated.reindex(target_dates)
                
        predictions = pd.Series(index=target_dates, dtype=float)
        if 'regression' in self.coefficients:
             predictions[:] = self.coefficients['regression']['const']
             for feature, coef in self.coefficients['regression']['features'].items():
                 if feature in aggregated_features.columns:
                     predictions += coef * aggregated_features[feature].fillna(0)
        else:
             logger.warning("Regression coefficients not found. Returning NaNs.")
             predictions[:] = np.nan
             
        return predictions

    def export_nowcast_csv(self, output_path: Optional[str] = None) -> pd.DataFrame:
        if self.X is None or self.y is None:
            self.load_data()
        
        predictions = self.predict()
        
        X_idx = self.X.index
        if not isinstance(X_idx, pd.PeriodIndex):
            X_idx = pd.DatetimeIndex(X_idx).to_period('D')
            
        predictions = predictions.reindex(X_idx)
        
        nowcast = pd.DataFrame({'predicted': predictions}, index=X_idx)
        
        nowcast['actual'] = None
        
        y_actual = self.y[self.target].copy()
        
        for date, value in y_actual.items():
            if pd.notna(value):
                date_period = pd.Period(date, freq='D')
                if date_period in nowcast.index:
                    nowcast.loc[date_period, 'actual'] = value
        
        release_dates = nowcast[nowcast['actual'].notna()].index.sort_values()
        
        logger.info(f"식별된 릴리즈 날짜 수: {len(release_dates)}")
        if len(release_dates) > 0:
            logger.info(f"첫 번째 릴리즈 날짜: {release_dates[0]}")
            logger.info(f"마지막 릴리즈 날짜: {release_dates[-1]}")
        
        period_avgs = pd.Series(index=nowcast.index, dtype=float)
        
        for i in range(len(release_dates)):
            current_release = release_dates[i]
            
            if i > 0:
                prev_release = release_dates[i-1]
                mask = (nowcast.index > prev_release) & (nowcast.index <= current_release)
                period_predictions = nowcast.loc[(nowcast.index > prev_release) & (nowcast.index < current_release), 'predicted']
                
                if not period_predictions.empty and not period_predictions.isna().all():
                    avg_prediction = period_predictions.mean()
                    period_avgs.loc[mask] = avg_prediction
                    logger.debug(f"기간 {prev_release} ~ {current_release} 의 평균 예측값: {avg_prediction:.4f}")
        
        if len(release_dates) > 0:
            last_release = release_dates[-1]
            future_mask = (nowcast.index > last_release)
            future_predictions = nowcast.loc[future_mask, 'predicted']
            
            if not future_predictions.empty and not future_predictions.isna().all():
                future_avg = future_predictions.mean()
                period_avgs.loc[future_mask] = future_avg
                logger.debug(f"마지막 발표일 {last_release} 이후 기간의 평균 예측값: {future_avg:.4f}")
        
        nowcast['period_avg'] = period_avgs
        
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nowcast.to_csv(output_path)
            logger.info(f"Nowcast results saved to {output_path}")
            
        return nowcast
        
    def plot_results(self, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        nowcast_df = self.export_nowcast_csv(os.path.join(output_dir, 'nowcasts.csv'))
        
        visualizer = DFMVisualizer('midas', output_dir)
        visualizer.plot_nowcast_results(nowcast_df)
        visualizer.plot_prediction_accuracy(nowcast_df)
        
        if 'regression' in self.coefficients:
            coefs = self.coefficients['regression']['features']
            importance = pd.Series(coefs).abs().sort_values(ascending=False)
            
            visualizer.plot_feature_importance(importance)

    def export_feature_importance(self, output_dir: Optional[str] = None) -> None:
        plot_output_dir = output_dir or self.output_dir
        if 'regression' in self.coefficients and 'features' in self.coefficients['regression']:
            coefs = self.coefficients['regression']['features']
            importance_df = pd.Series(coefs).abs().sort_values(ascending=False)
            
            if not importance_df.empty:
                visualizer = DFMVisualizer(self.model_type, plot_output_dir)
                visualizer.plot_feature_importance(importance_df)
            else:
                logger.warning("No feature importance data to export.")
        else:
            logger.warning("Regression coefficients not found, cannot export feature importance.")