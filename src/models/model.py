"""
model_gpt.py
Dynamic Factor Model for CPI Nowcasting with ElasticNet, XGBoost, or LightGBM
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict, List
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import xgboost as xgb
import lightgbm as lgb
import itertools
from visualizer import DFMVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_gpt")

class DFMModel:
    def __init__(
        self, 
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        train_window_size: int = 730,  # 약 2년
        n_factors: int = 1,
        model_type: str = 'elasticnet',  # 'elasticnet', 'xgboost', 'lightgbm'
        DATA_NAME: str = '9X',
        l1_ratio_range: List[float] = [0.05, 0.1, 0.3, 0.5],
        alpha_range: List[float] = [1e-4, 1e-3, 1e-2],
        xgb_params: Optional[Dict] = None,  # XGBoost 기본 파라미터
        lgb_params: Optional[Dict] = None,  # LightGBM 기본 파라미터
        model_path: Optional[str] = None
    ):
        self.X_path = X_path
        self.y_path = y_path
        self.target = target
        self.train_window_size = train_window_size
        self.n_factors = n_factors
        self.model_type = model_type.lower()
        self.DATA_NAME = DATA_NAME
        self.l1_ratio_range = l1_ratio_range
        self.alpha_range = alpha_range
        self.xgb_params = xgb_params or {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
        self.lgb_params = lgb_params or {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
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
        """X와 y 데이터를 로드하고, 오늘까지 확장하며 인덱스를 Period로 변환."""
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
                            ) -> Tuple[object, StandardScaler, Dict, List]:
        """선택한 모델 타입에 따라 요인을 추출하며, Grid Search로 최적 파라미터 탐색."""
        train_mask = y_train.notna()
        val_mask = y_val.notna()
        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]
        X_val = X_val.loc[val_mask]
        y_val = y_val.loc[val_mask]
        
        # 결측치가 적은 특성만 선택
        valid_cols = X_train.columns[X_train.isna().mean() < 0.2]
        if len(valid_cols) < 10:
            raise ValueError(f"사용 가능한 변수 부족: {len(valid_cols)}")
        
        # 결측치 처리 및 스케일링
        X_train_valid = X_train[valid_cols].fillna(X_train[valid_cols].mean())
        X_val_valid = X_val[valid_cols].fillna(X_val[valid_cols].mean())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_valid)
        X_val_scaled = scaler.transform(X_val_valid)

        best_model = None
        best_rmse = float('inf')
        best_params = {}

        if self.model_type == 'elasticnet':
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

        elif self.model_type == 'xgboost':
            param_combinations = [dict(zip(self.xgb_params.keys(), v)) 
                                for v in itertools.product(*self.xgb_params.values())]
            
            for params in param_combinations:
                model = xgb.XGBRegressor(**params, random_state=42)
                model.fit(X_train_scaled, y_train, 
                         eval_set=[(X_val_scaled, y_val)], 
                         verbose=False)
                pred = model.predict(X_val_scaled)
                rmse = mean_squared_error(y_val, pred)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = params

        elif self.model_type == 'lightgbm':
            param_combinations = [dict(zip(self.lgb_params.keys(), v)) 
                                for v in itertools.product(*self.lgb_params.values())]
            
            for params in param_combinations:
                # LightGBM 데이터셋 생성
                train_data = lgb.Dataset(X_train_scaled, label=y_train)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                
                # 기본 파라미터 설정
                base_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'random_state': 42,
                    'verbose': -1
                }
                
                # 사용자 정의 파라미터와 기본 파라미터 병합
                params.update(base_params)
                
                try:
                    model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[val_data],
                        num_boost_round=params['n_estimators'],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=10, verbose=False),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                    
                    pred = model.predict(X_val_scaled)
                    rmse = mean_squared_error(y_val, pred)
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_params = params
                        
                except Exception as e:
                    logger.warning(f"LightGBM 학습 실패 (파라미터: {params}): {str(e)}")
                    continue

        else:
            raise ValueError(f"지원되지 않는 model_type: {self.model_type}")

        logger.info(f"Validation 최적 RMSE: {best_rmse:.4f}")
        logger.info(f"최적 파라미터: {best_params}")
        return best_model, scaler, best_params, valid_cols.tolist()

    def _compute_Z(self, X: pd.DataFrame, models=None, scalers=None,
                   valid_cols=None) -> pd.DataFrame:
        """입력 X를 요인 시계열 Z로 변환."""
        Z = pd.DataFrame(index=X.index)
        models = models if models is not None else self.factor_models
        scalers = scalers if scalers is not None else self.scalers
        valid_cols = valid_cols if valid_cols is not None else self.valid_columns
        for i, (model, scaler, cols) in enumerate(zip(models, scalers, valid_cols)):
            X_valid = X[cols].fillna(0)
            X_scaled = scaler.transform(X_valid)
            if isinstance(model, lgb.Booster):
                Z[f'Factor_{i+1}'] = model.predict(X_scaled)
            else:
                Z[f'Factor_{i+1}'] = model.predict(X_scaled)
        return Z

    def fit(self) -> None:
        """최신 데이터를 사용해 모델을 학습."""
        self.load_data()
        end_date = pd.Timestamp.today()
        train_start = end_date - pd.Timedelta(days=self.train_window_size)
        mask = (self.X.index >= train_start.to_period('D')) & (self.X.index <= end_date.to_period('D')) & (self.y[self.target].notna())
        X_train_full = self.X.loc[mask].copy()
        y_train_full = self.y[self.target].loc[mask].copy()

        if X_train_full.empty or len(X_train_full) < 10:
            raise ValueError("학습 데이터 부족")

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
        """입력 X를 요인으로 변환 후 예측값 반환."""
        Z = self._compute_Z(X)
        return Z.sum(axis=1)

    def export_nowcast_csv(self, output_path: str = f'output/nowcasts.csv') -> None:
        """전체 기간의 Nowcast와 Actual 값을 CSV로 저장."""
        nowcast = pd.DataFrame(index=self.X.index)
        nowcast['predicted'] = self.predict(self.X)
        nowcast['actual'] = self.y[self.target]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nowcast.to_csv(output_path)
        logger.info(f"Nowcast results saved to {output_path}")

    def plot_results(self, output_dir: str = f'output/nowcasts') -> None:
        """Nowcast와 Actual 비교 플롯 및 Rolling RMSE 플롯 생성."""
        nowcast = pd.DataFrame(index=self.X.index)
        nowcast['predicted'] = self.predict(self.X)
        nowcast['actual'] = self.y[self.target]
        nowcast.to_csv(os.path.join(output_dir, 'nowcasts.csv'))
        
        visualizer = DFMVisualizer(self.model_type, output_dir)
        visualizer.plot_nowcast_results(nowcast)
        visualizer.plot_prediction_accuracy(nowcast)

    def analyze_factor_importance(self) -> pd.DataFrame:
        """각 변수의 요인별 중요도를 계산하고 상위 20개 반환."""
        importance = {}
        for i, (model, cols) in enumerate(zip(self.factor_models, self.valid_columns)):
            if model is None or not cols:
                continue
            if self.model_type == 'elasticnet':
                coefs = pd.Series(model.coef_, index=cols)
            elif self.model_type == 'xgboost':
                coefs = pd.Series(model.feature_importances_, index=cols)
            elif self.model_type == 'lightgbm':
                coefs = pd.Series(model.feature_importance(), index=cols)
            importance[f'Factor_{i+1}'] = coefs.abs()
        
        importance_df = pd.DataFrame(importance)
        if importance_df.empty:
            return pd.DataFrame()
        
        mean_importance = importance_df.mean(axis=1)
        top_features = mean_importance.sort_values(ascending=False).head(20)
        logger.info("팩터별 변수 중요도:\n" + top_features.to_string())
        return top_features

    def export_feature_importance(self, output_dir: str = 'output') -> None:
        """변수 중요도를 CSV와 플롯으로 저장."""
        importance_df = self.analyze_factor_importance()
        
        if importance_df.empty:
            logger.warning("Feature importance 데이터가 없습니다.")
            return
            
        visualizer = DFMVisualizer(self.model_type, output_dir)
        visualizer.plot_feature_importance(importance_df)

    def save_model(self, path: str) -> None:
        """모델과 관련 데이터를 저장."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'factor_models': self.factor_models,
            'scalers': self.scalers,
            'best_params': self.best_params,
            'valid_columns': self.valid_columns,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)
        logger.info(f"모델이 {path}에 저장되었습니다.")

    def load_model(self, path: str) -> None:
        """저장된 모델을 로드."""
        model_data = joblib.load(path)
        self.factor_models = model_data['factor_models']
        self.scalers = model_data['scalers']
        self.best_params = model_data['best_params']
        self.valid_columns = model_data['valid_columns']
        self.model_type = model_data['model_type']
        logger.info(f"모델을 {path}에서 로드했습니다.")