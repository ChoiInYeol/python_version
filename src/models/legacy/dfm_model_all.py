import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DFMModel:
    def __init__(
        self, 
        X_path: str,
        y_path: str,
        cpi_release_path: str,
        target: str = 'CPI_YOY',
        train_window_size: int = 730,  # Train 창 크기 (약 2년)
        val_window_size: int = 90,     # Validation 창 크기 (약 3개월)
        forecast_horizon: int = 40,
        start_date: Optional[str] = None,
        n_factors: int = 1,
        l1_ratio_range: list = [0.3, 0.5, 0.7],
        alpha_range: list = [1e-4, 1e-3, 1e-2],  # 정규화 완화
        model_path: Optional[str] = None
    ):
        self.X_path = X_path
        self.y_path = y_path
        self.cpi_release_path = cpi_release_path
        self.target = target
        self.train_window_size = train_window_size
        self.val_window_size = val_window_size
        self.forecast_horizon = forecast_horizon
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.n_factors = n_factors
        self.l1_ratio_range = l1_ratio_range
        self.alpha_range = alpha_range
        self.model_path = model_path
        
        self.X = None
        self.y = None
        self.cpi_release_dates = None
        self.factor_models = []
        self.scalers = []
        self.best_params = []
        self.valid_columns = []

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_data(self) -> None:
        try:
            self.X = pd.read_csv(self.X_path, parse_dates=['date'], index_col='date')
            self.y = pd.read_csv(self.y_path, parse_dates=['date'], index_col='date')
            self.X = self.X.dropna(axis=1, how='all').replace([np.inf, -np.inf], np.nan)
            
            today = pd.Timestamp.today()
            if today > self.X.index.max():
                new_idx = pd.date_range(start=self.X.index.min(), end=today, freq='D')
                self.X = self.X.reindex(new_idx)
                self.y = self.y.reindex(new_idx)
            
            cpi_release_df = pd.read_csv(self.cpi_release_path, parse_dates=['release_date'])
            self.cpi_release_dates = [pd.Timestamp(d) for d in cpi_release_df['release_date'].values]  # numpy.datetime64 -> pd.Timestamp
            
            self.X.index = pd.DatetimeIndex(self.X.index).to_period('D')
            self.y.index = pd.DatetimeIndex(self.y.index).to_period('D')
            
            logger.info(f"데이터 로드 완료: X={self.X.shape}, y={self.y.shape}")
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            raise

    def _grid_search_factor(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[ElasticNet, StandardScaler, Dict]:
        cpi_dates = pd.Index([d.to_period('D') for d in self.cpi_release_dates])
        train_idx = X_train.index.isin(cpi_dates) & y_train.notna()
        val_idx = X_val.index.isin(cpi_dates) & y_val.notna()
        
        X_train_valid = X_train.loc[train_idx]
        y_train_valid = y_train.loc[train_idx]
        X_val_valid = X_val.loc[val_idx]
        y_val_valid = y_val.loc[val_idx]
        
        if len(X_train_valid) < 10 or len(X_val_valid) < 2:
            raise ValueError(f"데이터 부족: Train={len(X_train_valid)}, Val={len(X_val_valid)}")
        
        valid_cols = X_train_valid.columns[X_train_valid.isna().mean() < 0.2]
        X_train_valid = X_train_valid[valid_cols].fillna(0)
        X_val_valid = X_val_valid[valid_cols].fillna(0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_valid)
        X_val_scaled = scaler.transform(X_val_valid)
        
        best_model = None
        best_rmse = float('inf')
        best_params = {}
        
        for alpha in self.alpha_range:
            for l1_ratio in self.l1_ratio_range:
                model = ElasticNet(
                    alpha=alpha, 
                    l1_ratio=l1_ratio, 
                    max_iter=10000, 
                    tol=1e-3, 
                    random_state=42
                )
                model.fit(X_train_scaled, y_train_valid)
                Z_val_pred = model.predict(X_val_scaled)
                rmse = mean_squared_error(y_val_valid, Z_val_pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        
        logger.info(f"Validation 최적 RMSE: {best_rmse:.4f}")
        return best_model, scaler, best_params, valid_cols.tolist()

    def _compute_Z(self, X: pd.DataFrame, models=None, scalers=None, valid_cols=None) -> pd.DataFrame:
        Z = pd.DataFrame(index=X.index)
        models = models if models is not None else self.factor_models
        scalers = scalers if scalers is not None else self.scalers
        valid_cols = valid_cols if valid_cols is not None else self.valid_columns
        
        for i, (model, scaler, cols) in enumerate(zip(models, scalers, valid_cols)):
            X_valid = X[cols].fillna(0)
            X_scaled = scaler.transform(X_valid)
            Z[f'Factor_{i+1}'] = model.predict(X_scaled)
        return Z

    def fit_sliding_window(self, end_date: pd.Timestamp = None) -> None:
        if self.X is None or self.y is None:
            self.load_data()
        
        self.factor_models = []
        self.scalers = []
        self.best_params = []
        self.valid_columns = []
        
        end_date = end_date or pd.Timestamp.today()
        start_date = self.X.index.min().to_timestamp()
        val_windows = []
        
        current_end = start_date + pd.Timedelta(days=self.train_window_size)
        while current_end + pd.Timedelta(days=self.val_window_size) <= end_date:
            train_start = current_end - pd.Timedelta(days=self.train_window_size)
            train_end = current_end
            val_end = current_end + pd.Timedelta(days=self.val_window_size)
            val_windows.append((train_start, train_end, val_end))
            current_end += pd.Timedelta(days=self.val_window_size)
        
        best_val_rmse = float('inf')
        best_model_data = None
        
        for train_start, train_end, val_end in val_windows:
            logger.info(f"Train: {train_start} ~ {train_end}, Val: {train_end} ~ {val_end}")
            train_mask = (self.X.index >= train_start.to_period('D')) & (self.X.index <= train_end.to_period('D'))
            val_mask = (self.X.index > train_end.to_period('D')) & (self.X.index <= val_end.to_period('D'))
            
            X_train = self.X.loc[train_mask].copy()
            y_train = self.y[self.target].loc[train_mask].copy()
            X_val = self.X.loc[val_mask].copy()
            y_val = self.y[self.target].loc[val_mask].copy()
            
            try:
                model, scaler, params, valid_cols = self._grid_search_factor(X_train, y_train, X_val, y_val)
                temp_models = [model]
                temp_scalers = [scaler]
                temp_params = [params]
                temp_cols = [valid_cols]
                
                Z_val = self._compute_Z(X_val, temp_models, temp_scalers, temp_cols)
                val_rmse = mean_squared_error(y_val[y_val.notna()], Z_val.reindex(y_val.index).interpolate()[y_val.notna()])
                
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model_data = (temp_models, temp_scalers, temp_params, temp_cols)
                    logger.info(f"최적 모델 갱신 - Val RMSE: {best_val_rmse:.4f}")
            
            except ValueError as e:
                logger.warning(f"슬라이딩 윈도우 학습 실패: {str(e)}")
                continue
        
        if best_model_data:
            self.factor_models, self.scalers, self.best_params, self.valid_columns = best_model_data
            if self.model_path:
                self.save_model(self.model_path)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        Z = self._compute_Z(X)
        return Z.sum(axis=1)

    def predict_daily(self, target_date: pd.Timestamp, current_date: pd.Timestamp) -> float:
        if self.X is None:
            self.load_data()
        
        mask = (self.X.index <= current_date.to_period('D'))
        pred_X = self.X[self.X.index == target_date.to_period('D')].copy()
        
        if pred_X.empty or len(self.X.loc[mask]) < self.train_window_size:
            logger.warning(f"데이터 부족: target_date={target_date}, current_date={current_date}")
            return np.nan
        
        return self.predict(pred_X).iloc[0]

    def nowcast_pipeline(self, current_date: pd.Timestamp) -> pd.DataFrame:
        if self.X is None or self.y is None:
            self.load_data()
        
        results = []
        future_release_dates = [d for d in self.cpi_release_dates if d > current_date]
        if not future_release_dates:
            logger.warning("미래 CPI 발표일 없음")
            return pd.DataFrame()
        
        target_date = min(future_release_dates)
        start_date = current_date - pd.Timedelta(days=self.forecast_horizon)
        
        for pred_date in pd.date_range(start_date, current_date, freq='D'):
            pred = self.predict_daily(target_date, pred_date)
            cpi_released = target_date <= current_date
            actual = self.y[self.target].loc[target_date.to_period('D')] if cpi_released else np.nan
            
            results.append({
                'date': pred_date,
                'target_date': target_date,
                'actual_target': actual,
                'predicted_target': pred,
                'cpi_released_target': cpi_released
            })
        
        return pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)

    def evaluate_historical(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        if self.X is None or self.y is None:
            self.load_data()
        
        results = []
        valid_release_dates = [d for d in self.cpi_release_dates if start_date <= d <= end_date]
        
        for release_date in valid_release_dates:
            start = release_date - pd.Timedelta(days=self.forecast_horizon)
            for current_date in pd.date_range(start, release_date, freq='D'):
                pred = self.predict_daily(release_date, current_date)
                actual = self.y[self.target].loc[pd.Timestamp(release_date).to_period('D')] if current_date >= release_date else np.nan
                results.append({
                    'date': current_date,
                    'target_date': release_date,
                    'actual_target': actual,
                    'predicted_target': pred,
                    'cpi_released_target': current_date >= release_date
                })
        
        results_df = pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)
        valid_df = results_df[results_df['cpi_released_target']].dropna(subset=['actual_target'])
        rmse = mean_squared_error(valid_df['actual_target'], valid_df['predicted_target'])
        logger.info(f"과거 데이터 RMSE: {rmse:.4f}")
        return results_df

    def analyze_factor_importance(self) -> pd.DataFrame:
        if not self.factor_models:
            logger.warning("모델이 학습되지 않았습니다.")
            return pd.DataFrame()
        
        importance = {}
        for i, (model, valid_cols) in enumerate(zip(self.factor_models, self.valid_columns)):
            coefs = pd.Series(model.coef_, index=valid_cols)
            importance[f'Factor_{i+1}'] = coefs.abs()
        
        importance_df = pd.DataFrame(importance)
        logger.info("팩터별 변수 중요도:\n" + importance_df.to_string())
        
        lag_cols = [col for col in importance_df.index if 'lag' in col.lower() or 'cpi' in col.lower()]
        if lag_cols:
            logger.info(f"Y Lag 변수 중요도:\n{importance_df.loc[lag_cols].to_string()}")
        
        return importance_df

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'factor_models': self.factor_models,
            'scalers': self.scalers,
            'best_params': self.best_params,
            'valid_columns': self.valid_columns
        }
        joblib.dump(model_data, path)
        logger.info(f"모델이 {path}에 저장되었습니다.")

    def load_model(self, path: str) -> None:
        model_data = joblib.load(path)
        self.factor_models = model_data['factor_models']
        self.scalers = model_data['scalers']
        self.best_params = model_data['best_params']
        self.valid_columns = model_data['valid_columns']
        logger.info(f"모델을 {path}에서 로드했습니다.")

    def plot_results(self, results_df: pd.DataFrame) -> None:
        plt.style.use(['science', 'ieee'])
        plt.figure(figsize=(12, 6), dpi=300)
        
        latest_predictions = results_df.sort_values('date').groupby('date').last().reset_index()
        y_min = min(latest_predictions['predicted_target'].min(), results_df['actual_target'].min(skipna=True))
        y_max = max(latest_predictions['predicted_target'].max(), results_df['actual_target'].max(skipna=True))
        y_range = y_max - y_min
        plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        plt.plot(latest_predictions['date'], latest_predictions['predicted_target'], 
                label='Nowcast', color='#4444FF', alpha=0.7, linewidth=2)
        
        release_dates = results_df[results_df['cpi_released_target']]['date'].unique()
        for date in release_dates:
            pred_value = latest_predictions[latest_predictions['date'] == date]['predicted_target'].iloc[0]
            plt.scatter(date, pred_value, color='#4444FF', s=100, zorder=5, marker='*')
            plt.annotate(f'Pred: {pred_value:.2f}%', (date, pred_value), xytext=(10, -10), 
                        textcoords='offset points', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        actual_df = results_df[results_df['cpi_released_target'] & results_df['actual_target'].notna()]
        for _, row in actual_df.iterrows():
            plt.scatter(row['date'], row['actual_target'], color='#FF4444', s=120, zorder=5)
            plt.annotate(f'Actual: {row["actual_target"]:.2f}%', (row['date'], row['actual_target']),
                        xytext=(10, 10), textcoords='offset points', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.title("Real-Time CPI Nowcasting", pad=20, fontsize=12)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("CPI YoY Change (%)", fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/nowcast_plot.svg', format='svg', bbox_inches='tight')
        plt.savefig('output/nowcast_plot.png', format='png', bbox_inches='tight', dpi=300)
        results_df.to_csv('output/nowcasts.csv', index=False)
        logger.info("결과가 output 디렉토리에 저장되었습니다.")
        plt.close()

def main():
    model = DFMModel(
        X_path='data/processed/X_processed.csv',
        y_path='data/processed/y_processed.csv',
        cpi_release_path='data/processed/cpi_release_date_full.csv',
        start_date='2024-06-01',
        train_window_size=730,
        val_window_size=90,
        forecast_horizon=25,
        n_factors=20,
        model_path='src/models/dfm_model.joblib'
    )
    
    # 오늘까지의 데이터로 Walking Forward 학습
    model.fit_sliding_window()
    
    # 오늘 기준 실시간 Nowcast
    # today = pd.Timestamp.today()
    # results_df = model.nowcast_pipeline(today)
    # model.plot_results(results_df)
    
    # 과거 성능 평가 (디버깅용, 선택적)
    historical_results = model.evaluate_historical(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))
    model.plot_results(historical_results)

if __name__ == "__main__":
    main()