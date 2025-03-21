"""
main.py
모델 실행 파일 - 모델 타입 선택 및 오늘 날짜의 Nowcast 출력
"""
from model import DFMModel
import pandas as pd

def main():
    # 모델 타입 선택: 'elasticnet', 'xgboost', 'lightgbm'
    model_type = 'elasticnet'  # 원하는 모델로 변경 가능
    
    xgb_params = {'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'n_estimators': [100, 200, 300]}
    
    lgb_params = {'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'n_estimators': [100, 200, 300]}

    # 모델 초기화
    model = DFMModel(
        X_path='data/processed/X_processed.csv',
        y_path='data/processed/y_processed.csv',
        target='CPI_YOY',
        train_window_size=365 * 5,  # 10년
        n_factors=2,
        model_type=model_type,
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        model_path=f'src/models/dfm_model_{model_type}.joblib'
    )
    
    # 모델 학습
    model.fit()
    
    # 전체 Nowcast 출력
    model.export_nowcast_csv(f'output/nowcasts_{model_type}.csv')
    model.plot_results(f'output/{model_type}')
    model.export_feature_importance(f'output/{model_type}')
    
    # 오늘 날짜의 Nowcast 출력
    today = pd.Timestamp.today().to_period('D')
    nowcast_df = pd.read_csv(f'output/nowcasts_{model_type}.csv', index_col=0, parse_dates=True)
    nowcast_df.index = pd.DatetimeIndex(nowcast_df.index).to_period('D')
    
    if today in nowcast_df.index:
        today_pred = nowcast_df.loc[today, 'predicted']
        today_actual = nowcast_df.loc[today, 'actual']
        print(f"오늘 ({today.to_timestamp()}):")
        print(f"  - Predicted CPI YoY ({model_type.capitalize()}): {today_pred:.2f}%")
        print(f"  - Actual CPI YoY: {today_actual if not pd.isna(today_actual) else 'N/A'}")
    else:
        print(f"오늘 ({today.to_timestamp()}) 데이터가 없습니다.")

if __name__ == "__main__":
    main()