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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("midas_model")

class MIDASModel:
    def __init__(
        self,
        X_path: str,
        y_path: str,
        target: str = 'CPI_YOY',
        lookback_periods: int = 12,  # Monthly periods to look back
        poly_degree: int = 2,        # Polynomial degree for weight function
        max_lags: Dict[str, int] = None,  # Dict of {variable_prefix: max_lag}
        train_window_size: int = 730,  # Days
        output_dir: str = 'output/midas_results'
    ):
        self.X_path = X_path
        self.y_path = y_path
        self.target = target
        self.lookback_periods = lookback_periods
        self.poly_degree = poly_degree
        self.max_lags = max_lags or {'default': 30}  # Default 30 days lag for all variables
        self.train_window_size = train_window_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.X = None
        self.y = None
        self.X_freq = 'D'  # Default daily frequency for X
        self.y_freq = 'M'  # Default monthly frequency for y
        self.coefficients = {}
        self.selected_features = []
        self.scaler = StandardScaler()
        
    def load_data(self) -> None:
        """Load X and y data and convert to appropriate frequencies"""
        try:
            self.X = pd.read_csv(self.X_path, parse_dates=['date'], index_col='date')
            self.y = pd.read_csv(self.y_path, parse_dates=['date'], index_col='date')
            
            # Detect X frequency
            if len(self.X) > 0:
                date_diffs = pd.Series(self.X.index).diff().dropna()
                most_common_diff = date_diffs.mode()[0]
                if most_common_diff.days == 1:
                    self.X_freq = 'D'
                elif most_common_diff.days == 7:
                    self.X_freq = 'W'
                else:
                    self.X_freq = 'D'  # Default to daily
            
            # Detect y frequency
            if len(self.y) > 0:
                date_diffs = pd.Series(self.y.index).diff().dropna()
                most_common_diff = date_diffs.mode()[0]
                if most_common_diff.days >= 28 and most_common_diff.days <= 31:
                    self.y_freq = 'M'
                    # Convert y to monthly end frequency
                    self.y.index = pd.DatetimeIndex(self.y.index).to_period('M').to_timestamp('M')
                else:
                    self.y_freq = 'D'  # Keep as daily if not monthly
            
            logger.info(f"Data loaded: X({self.X_freq})={self.X.shape}, y({self.y_freq})={self.y.shape}")
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _almon_weights(self, params: np.ndarray, lags: int) -> np.ndarray:
        """
        Calculate Almon polynomial weights for MIDAS
        
        Args:
            params: Parameters of the Almon polynomial
            lags: Number of lags
        
        Returns:
            Array of weights
        """
        weights = np.zeros(lags)
        for i in range(lags):
            norm_i = i / (lags - 1) if lags > 1 else 0  # Normalize lag index to [0,1]
            weight = 0
            for j, param in enumerate(params):
                weight += param * (norm_i ** j)
            weights[i] = np.exp(weight)  # Exponential to ensure positive weights
        
        # Normalize weights to sum to 1
        return weights / weights.sum() if weights.sum() > 0 else weights
    
    def _aggregate_high_freq_data(self, X_high_freq: pd.DataFrame, 
                                 y_dates: pd.DatetimeIndex, 
                                 feature: str,
                                 params: np.ndarray) -> pd.Series:
        """
        Aggregate high-frequency data to match target frequency using Almon weights
        
        Args:
            X_high_freq: High-frequency feature data
            y_dates: Target frequency dates
            feature: Feature name
            params: Almon polynomial parameters
            
        Returns:
            Aggregated series matching y_dates
        """
        # Get max lag for this feature type
        max_lag = None
        for prefix, lag in self.max_lags.items():
            if feature.startswith(prefix):
                max_lag = lag
                break
        if max_lag is None:
            max_lag = self.max_lags.get('default', 30)
        
        # Calculate weights
        weights = self._almon_weights(params, max_lag)
        
        # Create result series
        result = pd.Series(index=y_dates, dtype=float)
        
        # For each target date, calculate weighted average of high-freq data
        for target_date in y_dates:
            # Get end of month for target date if working with monthly data
            if self.y_freq == 'M':
                target_date = pd.Timestamp(target_date).to_period('M').to_timestamp('M')
            
            # Get high-frequency data before target date
            data_before = X_high_freq[X_high_freq.index <= target_date].iloc[-max_lag:]
            
            if len(data_before) > 0:
                # Apply weights (most recent data gets highest weight)
                weighted_values = data_before.values[::-1][:len(weights)] * weights[:len(data_before)]
                result[target_date] = weighted_values.sum()
            else:
                result[target_date] = np.nan
                
        return result
    
    def _objective_func(self, params: np.ndarray, X_train: pd.DataFrame, 
                       y_train: pd.Series, feature: str) -> float:
        """
        Objective function for optimizing MIDAS weights
        
        Args:
            params: Almon polynomial parameters
            X_train: Training features
            y_train: Training target
            feature: Feature name
            
        Returns:
            Mean squared error
        """
        aggregated = self._aggregate_high_freq_data(
            X_train[[feature]], y_train.index, feature, params
        )
        
        valid_indices = ~aggregated.isna() & ~y_train.isna()
        if valid_indices.sum() < 2:
            return np.inf
            
        mse = ((aggregated[valid_indices] - y_train[valid_indices]) ** 2).mean()
        return mse
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> List[str]:
        """
        Select most predictive features based on individual correlation with target
        
        Args:
            X: Features
            y: Target
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        correlations = {}
        
        # For each feature, optimize MIDAS weights and calculate correlation with target
        for feature in X.columns:
            if X[feature].isna().mean() > 0.2:  # Skip features with too many missing values
                continue
                
            try:
                # Initial parameters for Almon polynomial
                initial_params = np.zeros(self.poly_degree + 1)
                initial_params[0] = 0  # Start with equal weights
                
                # Optimize weights
                result = minimize(
                    self._objective_func,
                    initial_params,
                    args=(X[[feature]], y, feature),
                    method='BFGS',
                    options={'maxiter': 100}
                )
                
                if result.success:
                    # Calculate aggregated feature with optimized weights
                    aggregated = self._aggregate_high_freq_data(
                        X[[feature]], y.index, feature, result.x
                    )
                    
                    # Calculate correlation with target
                    valid_indices = ~aggregated.isna() & ~y.isna()
                    if valid_indices.sum() >= 5:  # Need enough points for reliable correlation
                        corr = np.abs(np.corrcoef(
                            aggregated[valid_indices], 
                            y[valid_indices]
                        )[0, 1])
                        
                        if not np.isnan(corr):
                            correlations[feature] = corr
            except Exception as e:
                logger.warning(f"Error processing feature {feature}: {str(e)}")
                continue
        
        # Select top features
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Top features by correlation: {top_features[:n_features]}")
        
        return [f[0] for f in top_features[:n_features]]
    
    def fit(self, n_features: int = 10) -> None:
        """
        Fit MIDAS model
        
        Args:
            n_features: Number of features to use
        """
        self.load_data()
        
        # Define training period
        end_date = self.y.index.max()
        train_start = end_date - pd.Timedelta(days=self.train_window_size)
        
        # Filter data
        mask_X = (self.X.index >= train_start) & (self.X.index <= end_date)
        X_train_full = self.X.loc[mask_X].copy()
        
        # For y, we need to consider its frequency
        if self.y_freq == 'M':
            train_start_month = pd.Timestamp(train_start).to_period('M')
            mask_y = (self.y.index >= train_start_month.to_timestamp('M')) & (self.y.index <= end_date)
        else:
            mask_y = (self.y.index >= train_start) & (self.y.index <= end_date)
            
        y_train_full = self.y[self.target].loc[mask_y].copy()
        
        # Select features
        self.selected_features = self._select_features(X_train_full, y_train_full, n_features)
        
        # Scale features
        X_scaled = pd.DataFrame(index=X_train_full.index)
        for feature in self.selected_features:
            if feature in X_train_full.columns:
                valid_data = X_train_full[feature].dropna()
                if len(valid_data) > 0:
                    scaler = StandardScaler()
                    X_scaled[feature] = pd.Series(
                        index=X_train_full.index,
                        data=np.nan
                    )
                    X_scaled.loc[valid_data.index, feature] = scaler.fit_transform(
                        valid_data.values.reshape(-1, 1)
                    ).flatten()
                    
                    # Store scaler for later use
                    self.coefficients[f"{feature}_scaler"] = scaler
        
        # Train MIDAS weights for each feature
        for feature in self.selected_features:
            if feature in X_scaled.columns:
                try:
                    # Initial parameters for Almon polynomial
                    initial_params = np.zeros(self.poly_degree + 1)
                    initial_params[0] = 0  # Start with equal weights
                    
                    # Optimize weights
                    result = minimize(
                        self._objective_func,
                        initial_params,
                        args=(X_scaled[[feature]], y_train_full, feature),
                        method='BFGS',
                        options={'maxiter': 200}
                    )
                    
                    if result.success:
                        self.coefficients[feature] = result.x
                        logger.info(f"Trained weights for {feature}: {result.x}")
                    else:
                        logger.warning(f"Failed to optimize weights for {feature}")
                        
                except Exception as e:
                    logger.warning(f"Error training weights for {feature}: {str(e)}")
        
        # Train final regression model
        aggregated_features = pd.DataFrame(index=y_train_full.index)
        
        for feature in self.selected_features:
            if feature in self.coefficients:
                aggregated = self._aggregate_high_freq_data(
                    X_scaled[[feature]], y_train_full.index, feature, self.coefficients[feature]
                )
                aggregated_features[feature] = aggregated
        
        # Drop rows with missing values
        valid_indices = ~y_train_full.isna()
        for col in aggregated_features.columns:
            valid_indices = valid_indices & ~aggregated_features[col].isna()
            
        if valid_indices.sum() < len(self.selected_features) + 1:
            raise ValueError("Not enough valid data points for regression")
            
        X_reg = aggregated_features.loc[valid_indices]
        y_reg = y_train_full.loc[valid_indices]
        
        # Fit OLS model
        X_reg_sm = sm.add_constant(X_reg)
        model = sm.OLS(y_reg, X_reg_sm)
        results = model.fit()
        
        # Store regression coefficients
        self.coefficients['regression'] = {
            'const': results.params['const'],
            'features': {feature: results.params[feature] for feature in self.selected_features 
                         if feature in results.params}
        }
        
        logger.info(f"Regression results:\n{results.summary().tables[1]}")
        
    def predict(self, X: Optional[pd.DataFrame] = None, 
               start_date: Optional[pd.Timestamp] = None,
               end_date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Generate nowcasts
        
        Args:
            X: Optional new features data
            start_date: Start date for predictions
            end_date: End date for predictions
            
        Returns:
            Series of predictions
        """
        # Use provided X or stored X
        X_pred = X if X is not None else self.X
        
        # Filter by date range if provided
        if start_date is not None:
            X_pred = X_pred[X_pred.index >= start_date]
        if end_date is not None:
            X_pred = X_pred[X_pred.index <= end_date]
        
        # Scale features
        X_scaled = pd.DataFrame(index=X_pred.index)
        for feature in self.selected_features:
            if feature in X_pred.columns and f"{feature}_scaler" in self.coefficients:
                scaler = self.coefficients[f"{feature}_scaler"]
                valid_mask = ~X_pred[feature].isna()
                X_scaled[feature] = pd.Series(
                    index=X_pred.index,
                    data=np.nan
                )
                if valid_mask.sum() > 0:
                    X_scaled.loc[valid_mask, feature] = scaler.transform(
                        X_pred.loc[valid_mask, feature].values.reshape(-1, 1)
                    ).flatten()
        
        # Generate target dates
        if self.y_freq == 'M':
            if start_date is not None and end_date is not None:
                start_month = pd.Timestamp(start_date).to_period('M')
                end_month = pd.Timestamp(end_date).to_period('M')
                target_dates = pd.date_range(
                    start=start_month.to_timestamp('M'),
                    end=end_month.to_timestamp('M'),
                    freq='M'
                )
            else:
                # Use all months in X_pred
                start_month = pd.Timestamp(X_pred.index.min()).to_period('M')
                end_month = pd.Timestamp(X_pred.index.max()).to_period('M')
                target_dates = pd.date_range(
                    start=start_month.to_timestamp('M'),
                    end=end_month.to_timestamp('M'),
                    freq='M'
                )
        else:
            # For daily frequency, use X_pred index
            target_dates = X_pred.index
            
        # Calculate aggregated features
        aggregated_features = pd.DataFrame(index=target_dates)
        
        for feature in self.selected_features:
            if feature in self.coefficients and feature in X_scaled.columns:
                aggregated = self._aggregate_high_freq_data(
                    X_scaled[[feature]], target_dates, feature, self.coefficients[feature]
                )
                aggregated_features[feature] = aggregated
        
        # Apply regression coefficients
        predictions = pd.Series(index=target_dates, dtype=float)
        predictions[:] = self.coefficients['regression']['const']
        
        for feature, coef in self.coefficients['regression']['features'].items():
            if feature in aggregated_features.columns:
                predictions += coef * aggregated_features[feature]
        
        return predictions
    
    def evaluate(self, start_date: Optional[pd.Timestamp] = None,
                end_date: Optional[pd.Timestamp] = None) -> Dict:
        """
        Evaluate model performance
        
        Args:
            start_date: Start date for evaluation
            end_date: End date for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate predictions
        predictions = self.predict(start_date=start_date, end_date=end_date)
        
        # Filter actual values
        y_actual = self.y[self.target].copy()
        if self.y_freq == 'M':
            y_actual.index = pd.DatetimeIndex(y_actual.index).to_period('M').to_timestamp('M')
            
        # Align predictions and actual values
        merged = pd.DataFrame({'pred': predictions, 'actual': y_actual})
        merged = merged.dropna(subset=['actual'])
        
        if len(merged) < 2:
            return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}
        
        # Calculate metrics
        rmse = np.sqrt(((merged['pred'] - merged['actual']) ** 2).mean())
        mae = (merged['pred'] - merged['actual']).abs().mean()
        
        # Calculate MAPE only for non-zero actual values
        non_zero = merged['actual'] != 0
        if non_zero.sum() > 0:
            mape = ((merged.loc[non_zero, 'pred'] - merged.loc[non_zero, 'actual']).abs() / 
                   merged.loc[non_zero, 'actual'].abs()).mean() * 100
        else:
            mape = np.nan
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
        
    def plot_results(self) -> None:
        """Plot nowcast results and feature importance"""
        # Generate predictions for the entire dataset
        predictions = self.predict()
        
        # Get actual values
        y_actual = self.y[self.target].copy()
        if self.y_freq == 'M':
            y_actual.index = pd.DatetimeIndex(y_actual.index).to_period('M').to_timestamp('M')
            
        # Create results dataframe
        results = pd.DataFrame({
            'Predicted': predictions,
            'Actual': y_actual
        })
        
        # Save to CSV
        results.to_csv(os.path.join(self.output_dir, 'nowcast_results.csv'))
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['Predicted'], label='Nowcast', linewidth=2)
        plt.plot(results.index, results['Actual'], 'r--', label='Actual', alpha=0.7)
        plt.scatter(results.index, results['Actual'], color='red', s=20)
        plt.title('MIDAS Nowcast Results')
        plt.xlabel('Date')
        plt.ylabel(f'{self.target} (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'nowcast_plot.png'), dpi=300)
        plt.close()
        
        # Plot feature importance
        if 'regression' in self.coefficients:
            coefs = self.coefficients['regression']['features']
            importance = pd.Series(coefs).abs().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            importance.plot(kind='bar')
            plt.title('Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Absolute Coefficient')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            # Save to CSV
            importance.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'))