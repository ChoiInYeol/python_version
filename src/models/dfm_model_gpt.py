import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dfm_model_gpt")

class DFMModel:
    def __init__(
        self, 
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        train_window_size: int = 730,  # 약 2년
        forecast_horizon: int = 90,     # 발표일 기준 N일 전부터 예측
        n_factors: int = 1,
        l1_ratio_range: List[float] = [0.05, 0.1, 0.3, 0.5],
        alpha_range: List[float] = [1e-4, 1e-3, 1e-2],
        model_path: Optional[str] = None
    ):
        self.X_path = X_path
        self.y_path = y_path
        self.target = target
        self.train_window_size = train_window_size
        self.forecast_horizon = forecast_horizon
        self.n_factors = n_factors
        self.l1_ratio_range = l1_ratio_range
        self.alpha_range = alpha_range
        self.model_path = model_path

        self.X = None
        self.y = None
        self.factor_models = []
        self.scalers = []
        self.best_params = []
        self.valid_columns = []

        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)

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
            self.X.index = pd.DatetimeIndex(self.X.index).to_period('D')
            self.y.index = pd.DatetimeIndex(self.y.index).to_period('D')
            logger.info(f"데이터 로드 완료: X={self.X.shape}, y={self.y.shape}")
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            raise

    def _grid_search_factor(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series
                              ) -> Tuple[ElasticNet, StandardScaler, Dict, List]:
        # y의 NaN 제거
        train_mask = y_train.notna()
        val_mask = y_val.notna()
        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]
        X_val = X_val.loc[val_mask]
        y_val = y_val.loc[val_mask]
        
        valid_cols = X_train.columns[X_train.isna().mean() < 0.2]
        if len(valid_cols) < 10:
            raise ValueError(f"사용 가능한 변수 부족: {len(valid_cols)}")
        X_train_valid = X_train[valid_cols].fillna(0)
        X_val_valid = X_val[valid_cols].fillna(0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_valid)
        X_val_scaled = scaler.transform(X_val_valid)

        best_model = None
        best_rmse = float('inf')
        best_params = {}
        for alpha in self.alpha_range:
            for l1_ratio in self.l1_ratio_range:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000,
                                   tol=1e-3, random_state=42)
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_val_scaled)
                rmse = mean_squared_error(y_val, pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        logger.info(f"Validation 최적 RMSE: {best_rmse:.4f}")
        return best_model, scaler, best_params, valid_cols.tolist()

    def _compute_Z(self, X: pd.DataFrame, models=None, scalers=None,
                   valid_cols=None) -> pd.DataFrame:
        Z = pd.DataFrame(index=X.index)
        models = models if models is not None else self.factor_models
        scalers = scalers if scalers is not None else self.scalers
        valid_cols = valid_cols if valid_cols is not None else self.valid_columns
        for i, (model, scaler, cols) in enumerate(zip(models, scalers, valid_cols)):
            X_valid = X[cols].fillna(0)
            Z[f'Factor_{i+1}'] = model.predict(scaler.transform(X_valid))
        return Z

    def fit(self) -> None:
        self.load_data()
        end_date = pd.Timestamp.today()
        train_start = end_date - pd.Timedelta(days=self.train_window_size)
        # 학습기간 내에서 실제 CPI 발표일(비NaN)만 선택
        mask = (self.X.index >= train_start.to_period('D')) & (self.X.index <= end_date.to_period('D')) & (self.y[self.target].notna())
        X_train_full = self.X.loc[mask].copy()
        y_train_full = self.y[self.target].loc[mask].copy()

        if X_train_full.empty or len(X_train_full) < 10:
            raise ValueError("학습 데이터 부족")

        # 시간순 80/20 분할
        split_idx = int(len(X_train_full) * 0.8)
        X_train = X_train_full.iloc[:split_idx]
        y_train = y_train_full.iloc[:split_idx]
        X_val = X_train_full.iloc[split_idx:]
        y_val = y_train_full.iloc[split_idx:]
        
        residual = y_train.copy()
        self.factor_models = []
        self.scalers = []
        self.best_params = []
        self.valid_columns = []
        for i in range(self.n_factors):
            try:
                model, scaler, params, valid_cols = self._grid_search_factor(X_train, residual, X_val, y_val)
            except ValueError as e:
                logger.warning(f"Factor {i+1} 학습 실패: {str(e)}")
                break
            self.factor_models.append(model)
            self.scalers.append(scaler)
            self.best_params.append(params)
            self.valid_columns.append(valid_cols)
            Z_train = self._compute_Z(X_train, models=[model], scalers=[scaler], valid_cols=[valid_cols])
            residual = residual - Z_train.iloc[:, 0]
            logger.info(f"Factor {i+1} 학습 완료: {params}, 사용 변수 수: {len(valid_cols)}")
        if self.model_path:
            self.save_model(self.model_path)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        Z = self._compute_Z(X)
        return Z.sum(axis=1)

    def export_nowcast_csv(self, output_path: str = 'output/nowcasts.csv') -> None:
        nowcast = pd.DataFrame(index=self.X.index)
        nowcast['predicted'] = self.predict(self.X)
        nowcast['actual'] = self.y[self.target]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nowcast.to_csv(output_path)
        logger.info(f"Nowcast results saved to {output_path}")

    def plot_results(self, output_dir: str = 'output') -> None:
        os.makedirs(output_dir, exist_ok=True)
        nowcast = pd.DataFrame(index=self.X.index)
        
        # 2015년 이전 데이터는 제외
        start_date = pd.Period('2015-01-01', freq='D')
        nowcast = nowcast[nowcast.index >= start_date]
        
        nowcast['predicted'] = self.predict(self.X)
        nowcast['actual'] = self.y[self.target]
        
        # CSV 파일로 저장
        nowcast.to_csv(os.path.join(output_dir, 'nowcasts.csv'))
        
        # RMSE 계산
        valid_mask = nowcast['actual'].notna()
        rmse_series = pd.Series(index=nowcast.index[valid_mask])
        for date in nowcast.index[valid_mask]:
            past_mask = (nowcast.index <= date) & valid_mask
            if past_mask.sum() >= 2:  # 최소 2개 이상의 데이터 포인트 필요
                rmse = np.sqrt(mean_squared_error(
                    nowcast.loc[past_mask, 'actual'],
                    nowcast.loc[past_mask, 'predicted']
                ))
                rmse_series[date] = rmse
        
        # 플롯 생성
        plt.style.use(['science', 'ieee'])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], dpi=300)
        
        # 상단 그래프: Nowcast vs Actual
        ax1.plot(nowcast.index.to_timestamp(), nowcast['predicted'], 
                label='Nowcast', color='#4444FF', linewidth=2)
        
        # 실제값 플롯 (점과 선으로 표시)
        actual_data = nowcast['actual'].copy()
        actual_data = actual_data.ffill()  # NaN 값을 이전 값으로 채우기
        
        ax1.plot(nowcast.index.to_timestamp(), actual_data, 
                color='#FF4444', linewidth=2, linestyle='--', alpha=0.5)
        
        # 실제값이 있는 지점에 큰 점으로 표시
        actual_points = nowcast[nowcast['actual'].notna()]
        ax1.scatter(actual_points.index.to_timestamp(), actual_points['actual'],
                   color='#FF4444', s=5, zorder=5, label='Actual')
        
        ax1.legend(loc='upper right')
        ax1.set_title("Real-Time CPI Nowcasting")
        ax1.set_ylabel("CPI YoY Change (%)")
        ax1.tick_params(axis='x', rotation=45)
        
        # 하단 그래프: RMSE
        ax2.plot(rmse_series.index.to_timestamp(), rmse_series.values,
                color='#44AA44', linewidth=1.5, label='Rolling RMSE')
        ax2.legend(loc='upper right')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("RMSE")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nowcast_plot.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'nowcast_plot.svg'))
        plt.close()
        logger.info(f"Nowcast plot and CSV saved to {output_dir}")

    def analyze_factor_importance(self) -> pd.DataFrame:
        importance = {}
        for i, (model, cols) in enumerate(zip(self.factor_models, self.valid_columns)):
            if model is None or not cols:
                continue
            coefs = pd.Series(model.coef_, index=cols)
            importance[f'Factor_{i+1}'] = coefs.abs()
        
        # 중요도를 DataFrame으로 변환하고 평균 계산
        importance_df = pd.DataFrame(importance)
        if importance_df.empty:
            return pd.DataFrame()
            
        # 각 변수별 평균 중요도 계산 후 상위 20개 선택
        mean_importance = importance_df.mean(axis=1)
        top_features = mean_importance.sort_values(ascending=False).head(20)
        logger.info("팩터별 변수 중요도:\n" + top_features.to_string())
        return top_features

    def export_feature_importance(self, output_dir: str = 'output') -> None:
        os.makedirs(output_dir, exist_ok=True)
        importance_df = self.analyze_factor_importance()
        
        if importance_df.empty:
            logger.warning("Feature importance 데이터가 없습니다.")
            return
            
        # % 기호가 포함된 칼럼명 처리
        importance_df.index = importance_df.index.str.replace('%', 'pct')
        
        # Series를 DataFrame으로 변환하여 저장
        importance_df.to_frame(name='Importance').to_csv(os.path.join(output_dir, 'feature_importance.csv'))
        
        plt.style.use(['science', 'ieee'])
        plt.figure(figsize=(10,6), dpi=300)
        importance_df.plot(kind='bar')
        plt.title("Feature Importance")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'feature_importance.svg'))
        plt.close()
        logger.info(f"Feature importance saved to {output_dir}")

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
