import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import Tuple, Dict
import pandas_ta as ta
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DataManager:
    def __init__(self, start_date: str, end_date: str, symbol: str = 'SPY'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbol = symbol
        self.scaler = StandardScaler()
        self.pca = None
        self.scaled_train_data = None

        self.raw_data = None
        self.raw_feature_df = None 
        self._load_initial_data()
        
    def _load_initial_data(self):
        print(f"Downloading data for {self.symbol}...")
        self.raw_data = yf.download(self.symbol, 
                                  start=self.start_date,
                                  end=self.end_date,
                                  progress=False)
        print("Data download complete.")

    def prepare_enhanced_features(self):
        df = self.raw_data.copy()        
        df['Returns'] = df['Adj Close'].pct_change()
        df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        
        for i in range(1, 11):
            decay = np.exp(-0.1 * i)
            df[f'Return_Lag_{i}'] = df['Returns'].shift(i) * decay
            df[f'Log_Return_Lag_{i}'] = df['Log_Returns'].shift(i) * decay
            
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Adj Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Adj Close'].ewm(span=window, adjust=False).mean()
            df[f'Price_to_SMA_{window}'] = df['Adj Close'] / df[f'SMA_{window}']
            
            df[f'ROC_{window}'] = (df['Adj Close'] - df['Adj Close'].shift(window)) / df['Adj Close'].shift(window)
            
        for window in [5, 10, 20, 50]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
            
            df[f'Parkinson_Volatility_{window}'] = (
                np.log(df['High'] / df['Low'])
                .rolling(window=window)
                .std() / np.sqrt(4 * np.log(2))
            )
            
            high_low = np.log(df['High'] / df['Low'])
            close_open = np.log(df['Close'] / df['Open'])
            df[f'GK_Volatility_{window}'] = (
                (0.5 * high_low ** 2 - (2 * np.log(2) - 1) * close_open ** 2)
                .rolling(window=window)
                .mean()
                .apply(np.sqrt)
            )

        df['Volume_Returns'] = df['Volume'].pct_change()
        for window in [5, 10, 20, 50]:
            df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_SMA_{window}']
            df[f'Volume_Trend_{window}'] = df[f'Volume_SMA_{window}'].pct_change()
            
            df[f'Money_Flow_{window}'] = (
                df['Adj Close'] * df['Volume']
            ).rolling(window=window).mean()

        custom_strategy = ta.Strategy(
            name="Custom",
            description="Enhanced Technical Indicators Strategy",
            ta=[
                {"kind": "rsi"},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "stoch"},
                {"kind": "adx"},
                {"kind": "cci"},
                {"kind": "mfi"},
                {"kind": "kst"},
                {"kind": "trix"},
                {"kind": "vwap"}
            ]
        )
        
        df.ta.strategy(custom_strategy)
        df['Target'] = df['Returns'].shift(-1)
        
        df = df.dropna()
        self.raw_feature_df = df

        return df
    
    def train_test_split(self, train_set_pct=0.8):
        split_idx = int(len(self.raw_feature_df) * train_set_pct)
        self.train_data_df = self.raw_feature_df[:split_idx]
        self.test_data_df  = self.raw_feature_df[split_idx:]

    def fit_scaler_to_train_data(self):
        self.train_data_df = self.train_data_df.fillna(method='ffill')

        X_train = self.train_data_df.drop(columns=['Target'])
        y_train = self.train_data_df['Target']

        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )

        self.scaled_train_data = pd.concat([scaled_features, y_train], axis=1)
        return self.scaled_train_data


    def fit_pca_to_train_data(self, n_components=20, visualize=False):
        if self.scaled_train_data is None:
            self.fit_scaler_to_train_data()
        
        X_train_scaled = self.scaled_train_data.drop(columns=['Target'])
        y_train = self.scaled_train_data['Target']
        
        self.pca = PCA(n_components=n_components)
        principal_components = self.pca.fit_transform(X_train_scaled)

        self.explained_variance = pd.DataFrame(
            data=self.pca.explained_variance_ratio_ * 100,
            columns=['Explained Variance (%)'],
            index=[f"PC_{i}" for i in range(1, n_components+1)]
        )
        self.cumulative_variance = self.explained_variance.cumsum()
        self.cumulative_variance.columns = ['Cumulative Explained Variance (%)']

        if visualize:
            fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

            self.explained_variance['Explained Variance (%)'].plot.barh(
                ax=axes[0],
                title="Explained Variance by Principal Components",
                xlabel="Explained Variance (%)",
                ylabel="Principal Components"
            )

            self.cumulative_variance['Cumulative Explained Variance (%)'].plot(
                ax=axes[1],
                title="Cumulative Explained Variance",
                xlabel="Number of Principal Components",
                ylabel="Cumulative Variance (%)",
                marker='o'
            )

            plt.tight_layout()
            plt.show()

        principal_components_df = pd.DataFrame(
            data=principal_components,
            index=X_train_scaled.index,
            columns=[f'PC_{i}' for i in range(1, self.pca.n_components_ + 1)]
        )
        
        principal_components_df = pd.concat([principal_components_df, y_train], axis=1)
        self.principal_components_df = principal_components_df
        return self.principal_components_df

    def transform_test_data_with_scaler(self):
        self.test_data_df = self.test_data_df.fillna(method='ffill')

        X_test = self.test_data_df.drop(columns=['Target'])
        y_test = self.test_data_df['Target']

        scaled_features = pd.DataFrame(
            self.scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )

        self.scaled_test_data = pd.concat([scaled_features, y_test], axis=1)
        return self.scaled_test_data

    def transform_test_data_with_pca(self):
        if not hasattr(self, 'scaled_test_data'):
            self.transform_test_data_with_scaler()

        X_test_scaled = self.scaled_test_data.drop(columns=['Target'])
        y_test = self.scaled_test_data['Target']
        
        test_principal_components = self.pca.transform(X_test_scaled)
        test_principal_components_df = pd.DataFrame(
            test_principal_components,
            index=X_test_scaled.index,
            columns=[f"PC_{i}" for i in range(1, self.pca.n_components_ + 1)]
        )

        test_principal_components_df = pd.concat(
            [test_principal_components_df, y_test], axis=1
        )
        return test_principal_components_df



    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.fillna(method='ffill')
        
        X_data = data.drop(columns=['Target'])
        y_data = data['Target']

        scaled_features = pd.DataFrame(
            self.scaler.transform(X_data),
            index=X_data.index,
            columns=X_data.columns
        )

        scaled_data = pd.concat([scaled_features, y_data], axis=1)
        return scaled_data


    def apply_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.fillna(method='ffill') 
        
        X_data = data.drop(columns=['Target'])
        y_data = data['Target']
        
        principal_components = self.pca.transform(X_data)
        
        principal_components_df = pd.DataFrame(
            principal_components,
            index=X_data.index,
            columns=[f"PC_{i}" for i in range(1, self.pca.n_components_ + 1)]
        )

        principal_components_df = pd.concat(
            [principal_components_df, y_data], axis=1
        )
        return principal_components_df



dm = DataManager('2005-01-01', '2024-10-31', 'SPY')
feature_df = dm.prepare_enhanced_features()
dm.train_test_split()
dm.fit_scaler_to_train_data()
X_train_pca = dm.fit_pca_to_train_data(visualize=True)
