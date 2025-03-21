from dfm_model_gpt import DFMModel

def main():
    model = DFMModel(
        X_path='data/processed/X_processed.csv',
        y_path='data/processed/y_processed.csv',
        target='CPI_YOY',
        train_window_size=730,
        forecast_horizon=20,
        n_factors=30,
        model_path='src/models/dfm_model_gpt.joblib'
    )
    model.fit()
    model.export_nowcast_csv('output/nowcasts.csv')
    model.plot_results('output')
    model.export_feature_importance('output')

if __name__ == "__main__":
    main()
