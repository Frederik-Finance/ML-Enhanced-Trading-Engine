import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from src.DataManager import DataManager
from src.ModelCreator import ModelCreator
from src.EnsembleVoting import EnhancedStrategyEnsemble


if __name__ == "__main__":
    dm = DataManager(start_date='2005-11-01', end_date='2024-10-31', symbol='SPY')
    feature_df = dm.prepare_enhanced_features()
    dm.train_test_split()
    dm.fit_scaler_to_train_data()
    train_data = dm.fit_pca_to_train_data(visualize=True)
    test_data = dm.transform_test_data_with_pca()
    
    mc = ModelCreator(dm)
    mc.train_models_parallel(train_data)
    
    print("\nTraining Performance:")
    train_metrics_df = pd.DataFrame.from_dict(mc.performance_metrics, orient='index')
    print(train_metrics_df)
    
    print("\nTest Performance:")
    test_metrics = mc.evaluate_models(test_data)
    print(test_metrics)
    
    train_mse = {model: metrics['mse'] for model, metrics in mc.performance_metrics.items()}
    test_mse = {model: metrics['mse'] for model, metrics in test_metrics.iterrows()}
    
    mse_df = pd.DataFrame({
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })
    
    mse_df.reset_index(inplace=True)
    mse_df.rename(columns={'index': 'Model'}, inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mse_df.melt(id_vars='Model'), 
                x='Model', 
                y='value', 
                hue='variable')
    plt.title('Mean Squared Error (MSE) for Train and Test Data')
    plt.ylabel('MSE')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()
    
    ensemble = EnhancedStrategyEnsemble(
        model_creator=mc,
        lookback_window=60,
        update_frequency=15,
        metric='returns'
    )
    
    X_test = test_data.drop('Target', axis=1)
    returns_test = test_data['Target']
    
    ensemble_predictions, weight_df = ensemble.predict_and_analyze(X_test, returns_test)
    
    cumulative_returns = pd.DataFrame(index=test_data.index)
    cumulative_returns['Benchmark'] = (1 + returns_test).cumprod()
    
    model_predictions = mc.predict(test_data)
    for model_name, y_pred in model_predictions.items():
        strategy_returns = pd.Series(returns_test.values * np.sign(y_pred), index=test_data.index)
        cum_returns = (1 + strategy_returns).cumprod()
        cumulative_returns[model_name] = cum_returns
    
    strategy_returns_ensemble = pd.Series(returns_test.values * np.sign(ensemble_predictions), index=test_data.index)
    cumulative_returns['Ensemble'] = (1 + strategy_returns_ensemble).cumprod()
    
    plt.figure(figsize=(12, 8))
    for column in cumulative_returns.columns:
        sns.lineplot(data=cumulative_returns, x=cumulative_returns.index, y=column, label=column)
    
    plt.title('Cumulative Returns of Models vs Ensemble vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    ensemble.plot_analysis(X_test, returns_test)
    
    metrics = ensemble.get_performance_metrics()
    print("\nEnsemble Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
