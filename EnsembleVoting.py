import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib

from DataManager import DataManager
from ModelCreator import ModelCreator


class EnhancedStrategyEnsemble:
    def __init__(
        self,
        model_creator: ModelCreator,
        lookback_window: int = 60,
        update_frequency: int = 20,
        metric: str = 'returns'
    ):
        self.model_creator = model_creator
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        self.metric = metric
        
        self.models = self.model_creator.trained_models
        
        self.strategy_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        self.performance_history = []
        self.weight_history = []
        self.current_best_strategy = None
        self.cumulative_returns = None
        self.strategy_returns = None
        
    def calculate_strategy_performance(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        method: str = 'sharpe'
    ) -> float:
        """Calculate performance metric for a strategy."""
        strategy_returns = np.sign(predictions) * returns
        epsilon = 1e-8

        if method == 'sharpe':
            std = strategy_returns.std() + epsilon
            return np.sqrt(252) * strategy_returns.mean() / std
        elif method == 'returns':
            return (1 + strategy_returns).prod() - 1
        elif method == 'sortino':
            negative_returns = strategy_returns[strategy_returns < 0]
            std_neg = negative_returns.std() + epsilon
            return np.sqrt(252) * strategy_returns.mean() / std_neg
        else:
            raise ValueError(f"Unsupported performance metric: {method}")
        
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    def update_strategy_weights(
        self,
        X: pd.DataFrame,
        returns: pd.Series,
        current_idx: int
    ) -> Dict[str, float]:
        """Update strategy weights based on recent performance."""
        if current_idx < self.lookback_window:
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            self.weight_history.append(weights)
            self.performance_history.append({name: 0.0 for name in self.models.keys()})
            self.current_best_strategy = None
            return weights
                    
        start_idx = max(0, current_idx - self.lookback_window)
        X_history = X.iloc[start_idx:current_idx]
        returns_history = returns.iloc[start_idx:current_idx]
        
        performances = {}
        for name, model in self.models.items():
            predictions = model.predict(X_history)
            perf = self.calculate_strategy_performance(
                returns_history,
                predictions,
                self.metric
            )
            performances[name] = perf
        
        perf_values = np.array(list(performances.values()))
        weights = self.softmax(perf_values)
        weights_dict = dict(zip(performances.keys(), weights))
            
        self.performance_history.append(performances)
        self.weight_history.append(weights_dict)
            
        self.current_best_strategy = max(performances.items(), key=lambda x: x[1])[0]
            
        return weights_dict
                
    def predict_and_analyze(self, X: pd.DataFrame, returns: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:

        predictions = pd.Series(index=X.index, dtype=float)
        strategy_returns = pd.Series(index=X.index, dtype=float)
        daily_weights = []
        
        for i in range(len(X)):
            if i % self.update_frequency == 0:
                self.strategy_weights = self.update_strategy_weights(X, returns, i)
                
            daily_weights.append(self.strategy_weights.copy())
            
            pred = 0
            for name, model in self.models.items():
                model_pred = model.predict(X.iloc[[i]])
                pred += model_pred[0] * self.strategy_weights[name]
                
            predictions.iloc[i] = pred
            strategy_returns.iloc[i] = np.sign(pred) * returns.iloc[i]
            
        self.strategy_returns = strategy_returns
        self.cumulative_returns = (1 + strategy_returns).cumprod()
        
        weight_df = pd.DataFrame(daily_weights, index=X.index)
        
        return predictions, weight_df
                
    def plot_analysis(self, X: pd.DataFrame, returns: pd.Series):
        """Generate comprehensive analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        history_df = pd.DataFrame(self.performance_history)
        history_df.index = X.index[::self.update_frequency][:len(history_df)]
        history_df.plot(ax=ax1, title='Strategy Performance History')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Performance Metric')
        
        weight_df = pd.DataFrame(self.weight_history)
        weight_df.index = X.index[:len(weight_df)]
        weight_df.plot(ax=ax2, title='Strategy Weight Evolution')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Weight')
        
        benchmark_returns = (1 + returns).cumprod()
        ax3.plot(benchmark_returns.index, benchmark_returns, label='Benchmark (Buy & Hold)', linestyle='--')
        ax3.plot(self.cumulative_returns.index, self.cumulative_returns, label='Dynamic Ensemble')
        ax3.set_title('Cumulative Returns Comparison')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Return')
        ax3.legend()
        ax3.grid(True)
        
        rolling_perf_series = self.strategy_returns.rolling(window=self.lookback_window).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        rolling_perf_series.plot(ax=ax4, title=f'Rolling {self.metric.capitalize()} Ratio')
        ax4.set_xlabel('Date')
        ax4.set_ylabel(f'{self.metric.capitalize()} Ratio')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        strategy_returns = self.strategy_returns
        
        metrics = {
            'Total Return': self.cumulative_returns.iloc[-1] - 1,
            'Annualized Return': (self.cumulative_returns.iloc[-1]) ** (252 / len(strategy_returns)) - 1,
            'Sharpe Ratio': np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else np.nan,
            'Max Drawdown': (self.cumulative_returns / self.cumulative_returns.cummax() - 1).min(),
            'Win Rate': np.mean(strategy_returns > 0),
            'Profit Factor': (
                strategy_returns[strategy_returns > 0].sum() / 
                abs(strategy_returns[strategy_returns < 0].sum())
            ) if strategy_returns[strategy_returns < 0].sum() != 0 else np.nan
        }
        
        return metrics

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
