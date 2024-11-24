import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Optional: For enhanced plot aesthetics
from typing import Dict
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
# Ensure that DataManager is in the same directory or adjust the import path accordingly
from DataManager import DataManager


class ModelCreator:
    """
    Handles model training and performance tracking.
    """
    def __init__(self, data_manager: DataManager):
        """
        Initialize ModelCreator with a DataManager instance.
        
        Args:
            data_manager (DataManager): An instance of DataManager with prepared data.
        """
        self.data_manager = data_manager
        self.model_configs = {
            'elastic_net': {
                'model': ElasticNet(
                    alpha=0.0001,
                    l1_ratio=0.5,
                    random_state=42
                ),
                'name': 'Elastic Net'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=1000,  # Increased because we're using very weak learners
                    learning_rate=0.01,  # Keep small learning rate for better generalization
                    max_depth=1,        # True tree stump
                    min_samples_leaf=20,  # Increased to prevent overfitting
                    subsample=0.7,      # Slightly reduced for better generalization
                    random_state=42
                ),
                'name': 'Gradient Boosting'
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=1000,  # Increased to compensate for weaker individual trees
                    max_depth=1,        # True tree stump
                    min_samples_leaf=20,  # Increased to prevent overfitting
                    bootstrap=True,     # Added explicit bootstrap sampling
                    max_features='sqrt',  # Added feature subsampling
                    random_state=42
                ),
                'name': 'Random Forest'
            }
        }
        self.performance_metrics = {}
        self.trained_models = {}

    @staticmethod
    def _calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        # Convert arrays to pandas Series for easier calculation
        y_pred = pd.Series(y_pred, index=pd.RangeIndex(len(y_pred)))
        y_true = pd.Series(y_true, index=pd.RangeIndex(len(y_true)))
        
        # Get signals and returns
        signals = np.sign(y_pred)
        strategy_returns = signals * y_true
        strategy_returns = pd.Series(strategy_returns, index=y_true.index)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'hit_rate': np.mean(np.sign(y_pred) == np.sign(y_true)),
            'sharpe': np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else np.nan
        }
        
        # Calculate cumulative returns and drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        drawdown = cum_returns / cum_returns.cummax() - 1
        
        metrics.update({
            'total_return': cum_returns.iloc[-1] - 1,
            'max_drawdown': drawdown.min()
        })
        
        return metrics

    @staticmethod
    def _train_single_model(model_config: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train a single model and compute its performance metrics"""
        try:
            # Extract model and name
            model = model_config['model']
            model_name = model_config['name']
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_train)
            
            # Calculate metrics
            metrics = ModelCreator._calculate_metrics(y_pred, y_train.values)
            
            return {
                'model_name': model_name,
                'model': model,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error training {model_config['name']}: {str(e)}")
            return None

    def train_models_parallel(self, pca_train_data: pd.DataFrame):
        """
        Train all models sequentially using PCA-transformed training data.
        
        Args:
            pca_train_data (pd.DataFrame): PCA-transformed training data with 'Target' column.
        """
        # Separate features and target
        X_train = pca_train_data.drop('Target', axis=1)
        y_train = pca_train_data['Target']
        
        # Loop over model configurations and train them
        for config in self.model_configs.values():
            model_name = config['name']
            try:
                result = self._train_single_model(config, X_train, y_train)
                if result is not None:
                    self.trained_models[model_name] = result['model']
                    self.performance_metrics[model_name] = result['metrics']
                    print(f"Successfully trained {model_name}")
                else:
                    print(f"Training failed for {model_name}")
            except Exception as e:
                print(f"Failed to train {model_name}: {str(e)}")

    def predict(self, pca_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all trained models.
        
        Args:
            pca_data (pd.DataFrame): PCA-transformed data with 'Target' column.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping model names to their predictions.
        """
        predictions = {}
        X = pca_data.drop('Target', axis=1)
        
        for model_name, model in self.trained_models.items():
            try:
                predictions[model_name] = model.predict(X)
            except Exception as e:
                print(f"Prediction failed for {model_name}: {str(e)}")
                predictions[model_name] = np.full(len(X), np.nan)
                
        return predictions

    def evaluate_models(self, pca_test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all models on test data.
        
        Args:
            pca_test_data (pd.DataFrame): PCA-transformed test data with 'Target' column.
        
        Returns:
            pd.DataFrame: DataFrame containing performance metrics for each model.
        """
        X_test = pca_test_data.drop('Target', axis=1)
        y_test = pca_test_data['Target']
        
        test_metrics = {}
        for model_name, model in self.trained_models.items():
            try:
                # Generate predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_pred, y_test.values)
                test_metrics[model_name] = metrics
            except Exception as e:
                print(f"Evaluation failed for {model_name}: {str(e)}")
                test_metrics[model_name] = {
                    'mse': np.nan,
                    'r2': np.nan,
                    'hit_rate': np.nan,
                    'sharpe': np.nan,
                    'total_return': np.nan,
                    'max_drawdown': np.nan
                }
                
        return pd.DataFrame.from_dict(test_metrics, orient='index')

    def get_rolling_performance(self, pca_test_data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        rolling_metrics = pd.DataFrame(index=pca_test_data.index)
        X_test = pca_test_data.drop('Target', axis=1)
        y_test = pca_test_data['Target']
        
        for model_name, model in self.trained_models.items():
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Calculate strategy returns
                returns = pd.Series(y_test.values * np.sign(y_pred), index=pca_test_data.index)
                
                # Calculate rolling Sharpe Ratio
                rolling_sharpe = (
                    np.sqrt(252) * 
                    returns.rolling(window).mean() / 
                    returns.rolling(window).std()
                )
                
                # Calculate rolling Hit Rate
                rolling_hit_rate = (
                    (np.sign(y_pred) == np.sign(y_test.values))
                    .rolling(window)
                    .mean()
                )
                
                # Calculate rolling total returns
                rolling_total_return = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
                
                # Assign to rolling_metrics DataFrame
                rolling_metrics[f'{model_name}_sharpe'] = rolling_sharpe
                rolling_metrics[f'{model_name}_hit_rate'] = rolling_hit_rate
                rolling_metrics[f'{model_name}_returns'] = rolling_total_return
            except Exception as e:
                print(f"Rolling performance calculation failed for {model_name}: {str(e)}")
                rolling_metrics[f'{model_name}_sharpe'] = np.nan
                rolling_metrics[f'{model_name}_hit_rate'] = np.nan
                rolling_metrics[f'{model_name}_returns'] = np.nan
                
        return rolling_metrics

    def save_models(self, directory: str):
        """
        Save models and their metrics.
        
        Args:
            directory (str): Directory path where models will be saved.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_data = {
                'models': self.trained_models,
                'metrics': self.performance_metrics,
                'configs': self.model_configs
            }
            filename = f"{directory}/models_{timestamp}.joblib"
            joblib.dump(save_data, filename)
            print(f"Saved models to {filename}")
        except Exception as e:
            print(f"Failed to save models: {str(e)}")

    def load_models(self, filepath: str):
        """
        Load saved models and metrics.
        
        Args:
            filepath (str): File path to the saved models.
        """
        try:
            save_data = joblib.load(filepath)
            self.trained_models = save_data['models']
            self.performance_metrics = save_data['metrics']
            self.model_configs = save_data['configs']
            print(f"Successfully loaded models from {filepath}")
        except Exception as e:
            print(f"Failed to load models: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize DataManager and prepare data
    dm = DataManager('2005-11-01', '2024-10-31', 'SPY')
    feature_df = dm.prepare_enhanced_features()
    dm.train_test_split()
    dm.fit_scaler_to_train_data()
    train_data = dm.fit_pca_to_train_data(visualize=True)
    test_data = dm.transform_test_data_with_pca()
    
    # Initialize and train models
    mc = ModelCreator(dm)
    mc.train_models_parallel(train_data)
    
    # Print training performance summary
    print("\nTraining Performance:")
    train_metrics_df = pd.DataFrame.from_dict(mc.performance_metrics, orient='index')
    print(train_metrics_df)
    
    # Evaluate models on test data
    print("\nTest Performance:")
    test_metrics = mc.evaluate_models(test_data)
    print(test_metrics)
    
    # Get rolling performance
    rolling_metrics = mc.get_rolling_performance(test_data, window=60)
    print("\nRolling Performance Summary:")
    print(rolling_metrics.describe())
    
    ##############################
    # Plotting MSEs for Train and Test
    ##############################

    # Extract MSEs
    train_mse = {model: metrics['mse'] for model, metrics in mc.performance_metrics.items()}
    test_mse = {model: metrics['mse'] for model, metrics in test_metrics.iterrows()}

    # Create a DataFrame for MSEs
    mse_df = pd.DataFrame({
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })

    # Reset index to have model names as a column
    mse_df.reset_index(inplace=True)
    mse_df.rename(columns={'index': 'Model'}, inplace=True)

    # Set Seaborn style
    sns.set(style='whitegrid')

    # Plot MSEs using Seaborn for better aesthetics
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
    
    ##############################
    # Plotting Cumulative Returns
    ##############################
    
    # Generate predictions for test data
    predictions = mc.predict(test_data)
    
    # Initialize a DataFrame to store cumulative returns
    cumulative_returns = pd.DataFrame(index=test_data.index)
    
    # Calculate benchmark cumulative returns
    cumulative_returns['Benchmark'] = (1 + test_data['Target']).cumprod()
    
    # Calculate and store cumulative returns for each model
    for model_name, y_pred in predictions.items():
        # Calculate strategy returns
        strategy_returns = pd.Series(test_data['Target'].values * np.sign(y_pred), index=test_data.index)
        
        # Calculate cumulative returns
        cum_returns = (1 + strategy_returns).cumprod()
        
        # Store in the DataFrame
        cumulative_returns[model_name] = cum_returns
    
    # Plot cumulative returns using Seaborn for better aesthetics
    plt.figure(figsize=(12, 8))
    for column in cumulative_returns.columns:
        sns.lineplot(data=cumulative_returns, x=cumulative_returns.index, y=column, label=column)
    
    plt.title('Cumulative Returns of Models vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    ##############################
    # Optional: Save the Plots
    ##############################
    
    # Save MSEs plot
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
    # plt.savefig('mse_plot.png')
    plt.show()
    
    # Save Cumulative Returns plot
    plt.figure(figsize=(12, 8))
    for column in cumulative_returns.columns:
        sns.lineplot(data=cumulative_returns, x=cumulative_returns.index, y=column, label=column)
    
    plt.title('Cumulative Returns of Models vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('cumulative_returns_plot.png')
    plt.show()
