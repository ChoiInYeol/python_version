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
from base_model import NowcastingBaseModel
from visualizer import DFMVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dfm_model")

class DFMModel(NowcastingBaseModel):
    def __init__(
        self, 
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        train_window_size: int = 730,  # 약 2년
        n_factors: int = 1,
        model_type: str = 'elasticnet',  # 'elasticnet', 'xgboost', 'lightgbm'
        data_name: str = 'unknown',
        l1_ratio_range: List[float] = [0.05, 0.1, 0.3, 0.5],
        alpha_range: List[float] = [1e-4, 1e-3, 1e-2],
        xgb_params: Optional[Dict] = None,  # XGBoost 기본 파라미터
        lgb_params: Optional[Dict] = None,  # LightGBM 기본 파라미터
        model_path: Optional[str] = None,
        smoothing_window: Optional[int] = None,  # 예측값 스무딩 윈도우
        avg_before_release: bool = True, # 발표 전날 평균값 대체 여부
        output_dir: Optional[str] = None
    ):
        super().__init__(
            X_path=X_path,
            y_path=y_path,
            target=target,
            model_type=model_type,
            data_name=data_name,
            output_dir=output_dir
        )
        
        self.train_window_size = train_window_size
        self.n_factors = n_factors
        self.l1_ratio_range = l1_ratio_range
        self.alpha_range = alpha_range
        self.xgb_params = xgb_params or {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [100]}
        self.lgb_params = lgb_params or {'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [100]}
        self.model_path = model_path
        self.smoothing_window = smoothing_window
        self.avg_before_release = avg_before_release

        self.factor_models = []
        self.scalers = []
        self.best_params = []
        self.valid_columns = []

        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            pass

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
            param_combinations = [dict(zip(self.xgb_params['max_depth'], v)) 
                                for v in itertools.product(*self.xgb_params['max_depth'])]
            
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
            param_combinations = [dict(zip(self.lgb_params['max_depth'], v)) 
                                for v in itertools.product(*self.lgb_params['max_depth'])]
            
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

    def fit(self, **kwargs) -> None:
        """최신 데이터를 사용해 모델을 학습."""
        if self.X is None or self.y is None:
            self.load_data()
            if self.X is None or self.y is None: 
                logger.error("Data loading failed, cannot fit model.")
                return
        
        # 학습 종료 날짜 설정 (y의 마지막 유효한 날짜 기준)
        end_date = self.y[self.target].dropna().index.max()
        if not isinstance(end_date, pd.Period):
            end_date = pd.Period(end_date, freq='D') # Ensure Period type
            
        train_start = end_date - pd.Timedelta(days=self.train_window_size)
        if not isinstance(train_start, pd.Period):
             train_start = pd.Period(train_start, freq='D') # Ensure Period type
        
        # 학습 데이터 필터링 (각 데이터프레임에 맞는 마스크 사용)
        x_mask = (self.X.index >= train_start) & (self.X.index <= end_date)
        X_train_full = self.X.loc[x_mask].copy()
        
        y_mask = (self.y.index >= train_start) & (self.y.index <= end_date)
        y_train_full = self.y[self.target].loc[y_mask].copy()
        
        # 타겟 값이 있는 날짜만 선택 (dropna)
        y_train_full = y_train_full.dropna()
        
        # y의 유효한 인덱스에 맞춰 X 필터링
        valid_y_index = y_train_full.index
        X_train_full = X_train_full.loc[X_train_full.index.isin(valid_y_index)]
        
        # 최종적으로 X와 y의 인덱스를 일치시킴 (dropna된 y 기준)
        common_index = X_train_full.index.intersection(valid_y_index)
        X_train_full = X_train_full.loc[common_index]
        y_train_full = y_train_full.loc[common_index]

        if X_train_full.empty or len(X_train_full) < 10:
            logger.error(f"학습 데이터 부족 (시작: {train_start}, 끝: {end_date}, 공통 개수: {len(X_train_full)})")
            raise ValueError("학습 데이터 부족")

        # 검증 데이터 분리
        split_idx = int(len(X_train_full) * 0.8)
        X_train = X_train_full.iloc[:split_idx]
        y_train = y_train_full.iloc[:split_idx]
        X_val = X_train_full.iloc[split_idx:]
        y_val = y_train_full.iloc[split_idx:]
        
        # Factor 모델 학습
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

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        입력 X를 요인으로 변환 후 예측값 반환.
        Base 클래스의 요구사항을 만족하는 PeriodIndex Series 반환.
        """
        if X is None:
            if self.X is None:
                self.load_data()
                if self.X is None: raise ValueError("X data not loaded.")
            X = self.X
        
        if not isinstance(X.index, pd.PeriodIndex):
            X.index = pd.DatetimeIndex(X.index).to_period('D')
            
        Z = self._compute_Z(X)
        raw_prediction = Z.sum(axis=1)

        if not raw_prediction.index.is_monotonic_increasing:
             raw_prediction = raw_prediction.sort_index()
             logger.debug("Raw prediction index sorted.")

        if self.smoothing_window and self.smoothing_window > 1:
            final_prediction = raw_prediction.rolling(window=self.smoothing_window, min_periods=1).mean()
            logger.info(f"Applying SMA smoothing with window {self.smoothing_window}")
            if not raw_prediction.empty:
                 logger.debug(f"Raw prediction sample (tail) before smoothing:\n{raw_prediction.tail()}")
                 logger.debug(f"Smoothed prediction sample (tail):\n{final_prediction.tail()}")
        else:
            final_prediction = raw_prediction.copy()
            if not raw_prediction.empty:
                 logger.debug(f"Raw prediction sample (no smoothing, tail):\n{raw_prediction.tail()}")

        if self.avg_before_release:
            if self.y is None: self.load_data()
            if self.y is not None:
                release_dates = self.y[self.target].dropna().index.sort_values()
                if len(release_dates) > 1:
                    logger.info("Applying averaging for the day before release dates.")
                    for i in range(1, len(release_dates)):
                        current_release_date = release_dates[i]
                        prev_release_date = release_dates[i-1]
                        target_date = current_release_date - 1
                        start_avg_date = prev_release_date + 1
                        if target_date in final_prediction.index and start_avg_date <= target_date:
                            preds_to_avg = raw_prediction.loc[start_avg_date:target_date]
                            if not preds_to_avg.empty:
                                avg_pred = preds_to_avg.mean()
                                original_pred = final_prediction.loc[target_date]
                                final_prediction.loc[target_date] = avg_pred
                                logger.debug(f"Prediction for {target_date} (day before {current_release_date}) "
                                             f"replaced with average ({avg_pred:.4f}) "
                                             f"of period {start_avg_date} to {target_date}. "
                                             f"Original prediction was: {original_pred:.4f}")
                            else:
                                 logger.debug(f"No raw predictions found between {start_avg_date} and {target_date} "
                                              f"to average for target date {target_date}.")

        if not isinstance(final_prediction.index, pd.PeriodIndex):
            final_prediction.index = pd.DatetimeIndex(final_prediction.index).to_period('D')
            
        return final_prediction.reindex(X.index)

    def get_latest_monthly_nowcast(self) -> Optional[float]:
        """
        가장 최근 발표일 이후부터 최신 데이터까지의 일별 예측 평균을 계산하여
        현재 (또는 다음) 발표 기간에 대한 월간 Nowcast 값을 반환합니다.
        """
        if self.X is None or self.y is None:
            logger.warning("데이터가 로드되지 않아 월간 Nowcast를 계산할 수 없습니다.")
            try:
                self.load_data()
                logger.info("데이터를 로드했습니다.")
            except Exception as e:
                logger.error(f"데이터 로드 중 오류 발생: {e}")
                return None

        daily_predictions = self.predict(self.X)
        if daily_predictions.empty:
            logger.warning("일별 예측값을 얻지 못했습니다.")
            return None

        valid_y = self.y[self.target].dropna()
        if valid_y.empty:
            logger.warning("y 데이터에서 실제 발표일을 찾을 수 없습니다.")
            return None
        last_release_date = valid_y.index.max()

        start_avg_date = last_release_date + 1
        end_avg_date = daily_predictions.index.max()

        if start_avg_date > end_avg_date:
            logger.warning(f"평균 계산 시작일({start_avg_date})이 마지막 예측일({end_avg_date})보다 늦습니다. "
                           f"마지막 일별 예측값을 반환합니다: {daily_predictions.iloc[-1]:.4f}")
            return daily_predictions.iloc[-1]

        period_predictions = daily_predictions.loc[start_avg_date:end_avg_date]

        if period_predictions.empty:
            logger.warning(f"{start_avg_date}부터 {end_avg_date} 사이의 예측값을 찾을 수 없습니다. "
                           f"마지막 일별 예측값을 반환합니다: {daily_predictions.iloc[-1]:.4f}")
            return daily_predictions.iloc[-1]

        monthly_avg_nowcast = period_predictions.mean()
        logger.info(f"{start_avg_date}부터 {end_avg_date}까지의 일별 예측 평균 (월간 Nowcast): {monthly_avg_nowcast:.4f}")

        return monthly_avg_nowcast

    def export_nowcast_csv(self, output_path: str = f'output/nowcasts.csv') -> None:
        """전체 기간의 Nowcast와 Actual 값을 CSV로 저장하고, 발표일 사이의 평균값도 계산하여 저장."""
        nowcast = pd.DataFrame(index=self.X.index)
        nowcast['predicted'] = self.predict(self.X)
        nowcast['actual'] = self.y[self.target]
        
        release_dates = nowcast[nowcast['actual'].notna()].index.sort_values()
        
        period_avgs = pd.Series(index=nowcast.index, dtype=float)
        
        for i in range(len(release_dates)):
            current_release = release_dates[i]
            
            if i > 0:
                prev_release = release_dates[i-1]
                # 평균 계산용 데이터: 이전 발표일 포함, 현재 발표일 제외
                period_predictions = nowcast.loc[(nowcast.index >= prev_release) & (nowcast.index < current_release), 'predicted']
                
                if not period_predictions.empty:
                    avg_prediction = period_predictions.mean()
                    # 평균값은 현재 발표일에만 적용
                    period_avgs.loc[current_release] = avg_prediction
                    logger.debug(f"기간 {prev_release} ~ {current_release} 의 평균 예측값: {avg_prediction:.4f}")
        
        if len(release_dates) > 0:
            last_release = release_dates[-1]
            future_mask = (nowcast.index > last_release)
            future_predictions = nowcast.loc[future_mask, 'predicted']
            
            if not future_predictions.empty:
                future_avg = future_predictions.mean()
                # 마지막 발표일 이후 기간에는 평균값 적용 유지
                period_avgs.loc[future_mask] = future_avg
                logger.debug(f"마지막 발표일 {last_release} 이후 기간의 평균 예측값: {future_avg:.4f}")
        
        nowcast['period_avg'] = period_avgs
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nowcast.to_csv(output_path)
        logger.info(f"Nowcast results saved to {output_path} with period averages")

    def plot_results(self, output_dir: str = f'output/nowcasts') -> None:
        """Nowcast와 Actual 비교 플롯 및 Rolling RMSE 플롯 생성."""
        output_path = os.path.join(output_dir, 'nowcasts.csv')
        self.export_nowcast_csv(output_path)
        
        nowcast = pd.read_csv(output_path, index_col=0, parse_dates=True)
        nowcast.index = pd.DatetimeIndex(nowcast.index).to_period('D')
        
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

    def export_feature_importance(self, output_dir: Optional[str] = None) -> None:
        """변수 중요도를 CSV와 플롯으로 저장."""
        plot_output_dir = output_dir or self.output_dir
        importance_df = self.analyze_factor_importance()
        
        if importance_df.empty:
            logger.warning("Feature importance 데이터가 없습니다.")
            return
            
        visualizer = DFMVisualizer(self.model_type, plot_output_dir)
        visualizer.plot_feature_importance(importance_df)

    def save_model(self, path: Optional[str] = None) -> None:
        """모델과 관련 데이터를 저장."""
        save_path = path or self.model_path
        if not save_path:
            save_path = os.path.join(self.output_dir, f'{self.model_type}_model.joblib')
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {
            'factor_models': self.factor_models,
            'scalers': self.scalers,
            'best_params': self.best_params,
            'valid_columns': self.valid_columns,
            'X_path': self.X_path,
            'y_path': self.y_path,
            'target': self.target,
            'model_type': self.model_type,
            'output_dir': self.output_dir,
            'train_window_size': self.train_window_size,
            'n_factors': self.n_factors,
            'l1_ratio_range': self.l1_ratio_range,
            'alpha_range': self.alpha_range,
            'xgb_params': self.xgb_params,
            'lgb_params': self.lgb_params,
            'smoothing_window': self.smoothing_window,
            'avg_before_release': self.avg_before_release
        }
        joblib.dump(model_data, save_path)
        logger.info(f"모델이 {save_path}에 저장되었습니다.")

    def load_model(self, path: str) -> None:
        """저장된 모델을 로드."""
        try:
            model_data = joblib.load(path)
            # 모델 관련 속성 로드
            self.factor_models = model_data['factor_models']
            self.scalers = model_data['scalers']
            self.best_params = model_data['best_params']
            self.valid_columns = model_data['valid_columns']
            
            # Base 클래스 속성 로드 (이전 버전 호환성 처리)
            # 기존 모델 파일에 해당 키가 없을 수 있으므로, self의 현재 값으로 대체
            self.X_path = model_data.get('X_path', self.X_path)
            self.y_path = model_data.get('y_path', self.y_path)
            self.target = model_data.get('target', self.target)
            self.model_type = model_data.get('model_type', self.model_type)
            self.output_dir = model_data.get('output_dir', self.output_dir)
            
            # DFM 모델 특정 속성 로드
            self.train_window_size = model_data.get('train_window_size', self.train_window_size)
            self.n_factors = model_data.get('n_factors', self.n_factors)
            self.l1_ratio_range = model_data.get('l1_ratio_range', self.l1_ratio_range)
            self.alpha_range = model_data.get('alpha_range', self.alpha_range)
            self.xgb_params = model_data.get('xgb_params', self.xgb_params)
            self.lgb_params = model_data.get('lgb_params', self.lgb_params)
            self.smoothing_window = model_data.get('smoothing_window')
            self.avg_before_release = model_data.get('avg_before_release', False)
            
            self.model_path = path # 로드 경로 저장
            logger.info(f"모델을 {path}에서 로드했습니다 (호환성 모드 적용 가능).")
            # 데이터 로드는 사용 시 필요하면 수행
            
        except FileNotFoundError:
            logger.error(f"모델 파일({path})을 찾을 수 없습니다.")
            raise
        except KeyError as e:
            logger.error(f"모델 파일({path}) 로드 중 키 오류 발생: {e}. 파일 형식이 호환되지 않을 수 있습니다.")
            # 호환되지 않는 파일이어도 기본적인 로드는 시도
            logger.warning("일부 속성 로드에 실패했을 수 있습니다.")
        except Exception as e:
            logger.error(f"모델 파일({path}) 로드 중 예외 발생: {e}", exc_info=True)
            raise