"""
AlgoTrade AI — MarketPulse Optimizer
Everything-in-one runnable notebook script (Python)

Contents:
1) Setup & requirements
2) Download / load data (yfinance or local Kaggle CSV)
3) Compute technical indicators (10 used in the base paper)
4) Preprocessing & supervised dataset creation
5) Training: XGBoost, RandomForest, LSTM (windowed)
6) Evaluation metrics and plots
7) Simple backtest using model predictions
8) Save models and scalers
9) Minimal Streamlit snippet (at the end) to hook into this pipeline

HOW TO RUN:
- Recommended: run this file as a Jupyter notebook. Each section is separated with comments and can be executed cell-by-cell.
- Ensure you have installed the packages in requirements.txt (listed below).

Requirements (pip):
# pip install pandas numpy scikit-learn xgboost tensorflow==2.12.0 yfinance joblib matplotlib ta streamlit

Place any local dataset CSV under ./data/ (e.g. data/tehran_stock.csv) or use Yahoo tickers via yfinance.
If you have the base paper PDF uploaded at /mnt/data/entropy-22-00840.pdf you can keep it as reference.

"""

# -----------------------------
# 1) SETUP & IMPORTS
# -----------------------------
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

import xgboost as xgb
import yfinance as yf

# Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 2) DATA INGESTION
# -----------------------------

def download_yahoo(ticker, start='2010-01-01', end=None, save_csv=None):
    df = yf.download(ticker, start=start, end=end, progress=False)

    # --- FIX 1: Flatten MultiIndex columns if present ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- FIX 2: Handle missing 'Adj Close' column safely ---
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj Close' in df.columns:
        expected_cols.insert(4, 'Adj Close')  # include it if present

    df = df[expected_cols]
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

    # --- Optional: Save to CSV if requested ---
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv)

    return df



def load_local_csv(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df

# Example: download_yahoo('BTC-USD', start='2018-01-01')

# -----------------------------
# 3) TECHNICAL INDICATORS
# -----------------------------

def SMA(series, n=14):
    return series.rolling(n).mean()


def WMA(series, n=14):
    weights = np.arange(1, n+1)
    return series.rolling(n).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)


def MOM(series, n=14):
    return series - series.shift(n-1)


def stochastic_k(df, n=14):
    low_n = df['Low'].rolling(n).min()
    high_n = df['High'].rolling(n).max()
    return 100*(df['Close'] - low_n)/(high_n - low_n + 1e-9)


def stochastic_d(df, n=3, k_n=14):
    k = stochastic_k(df, n=k_n)
    return k.rolling(n).mean()


def RSI(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean()
    rs = ma_up/(ma_down + 1e-9)
    return 100 - (100/(1+rs))


def MACD(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def larry_williams_r(df, n=14):
    hh = df['High'].rolling(n).max()
    ll = df['Low'].rolling(n).min()
    return 100*(hh - df['Close'])/(hh - ll + 1e-9)


def accumulation_distribution(df):
    mf_mult = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9))
    mf_vol = mf_mult * df['Volume']
    return mf_vol.cumsum()  # cumulative A/D


def CCI(df, n=20):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    sma = TP.rolling(n).mean()
    mad = TP.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (TP - sma) / (0.015 * mad + 1e-9)
    return cci


def add_indicators(df):
    df = df.copy()
    df['SMA14'] = SMA(df['Close'], 14)
    df['WMA14'] = WMA(df['Close'], 14)
    df['MOM14'] = MOM(df['Close'], 14)
    df['STCK'] = stochastic_k(df, 14)
    df['STCD'] = stochastic_d(df, 3, 14)
    df['RSI14'] = RSI(df['Close'], 14)
    macd, signal = MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_SIGNAL'] = signal
    df['LWR14'] = larry_williams_r(df, 14)
    df['ADO'] = accumulation_distribution(df)
    df['CCI20'] = CCI(df, 20)
    df = df.dropna()
    return df

# -----------------------------
# 4) PREPROCESSING: supervised dataset
# -----------------------------

def create_supervised(df, feature_cols, target_col='Close', n_ahead=1):
    """
    Creates supervised dataset where target is next-day return instead of next-day price.
    return_t+1 = (Close_t+1 - Close_t) / Close_t
    """
    X = df[feature_cols].copy()
    
    # Compute future return as target
    y = (df[target_col].shift(-n_ahead) - df[target_col]) / (df[target_col] + 1e-9)
    
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    y_df = pd.DataFrame({'target': y})
    
    dataset = pd.concat([X, y_df], axis=1).dropna()
    return dataset





def normalize_df(df, feature_cols):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler

# -----------------------------
# 5) TRAIN / EVALUATE FUNCTIONS
# -----------------------------

def metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Directional Accuracy: % of times model predicted correct direction (up/down)
    # Much more meaningful than MAPE when targets are near-zero returns
    directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'DA': directional_acc}


def train_xgb(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, metrics(y_test, preds), preds


def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, metrics(y_test, preds), preds

# LSTM helper to build windowed sequences
def create_windowed_sequences(df_features, target_series, window_size=10):
    X, y = [], []
    arr = df_features.values
    t = target_series.values
    for i in range(len(arr) - window_size):
        X.append(arr[i:i+window_size])
        y.append(t[i+window_size])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=64):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, units=64)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    preds = model.predict(X_test).flatten()
    return model, metrics(y_test, preds), preds

# -----------------------------
# 6) BACKTEST (simple): signals from predicted next-day price
# -----------------------------

def simple_backtest(df_with_preds, pred_col='pred', threshold=0.001):
    """
    Backtest with Long, Short, and Hold signals.
    
    signal = +1  → predicted return >  +threshold  (Long / Buy)
    signal = -1  → predicted return <  -threshold  (Short / Sell)
    signal =  0  → within threshold band           (Hold / No trade)

    transaction_cost = 0.1% per trade (realistic for crypto exchanges like Binance)
    """
    TRANSACTION_COST = 0.001  # 0.1% per trade

    df = df_with_preds.copy()

    # --- Signals ---
    df['signal'] = 0
    df.loc[df[pred_col] >  threshold, 'signal'] =  1   # Long
    df.loc[df[pred_col] < -threshold, 'signal'] = -1   # Short

    # --- Actual next-day return ---
    df['asset_ret'] = df['Close'].pct_change().shift(-1)

    # --- Strategy return = signal * actual return ---
    df['strategy_ret'] = df['signal'] * df['asset_ret']

    # --- Deduct transaction cost on every trade entry ---
    # A trade entry is when signal changes from previous row
    df['trade_entry'] = (df['signal'] != df['signal'].shift(1)).astype(int)
    df['strategy_ret'] = df['strategy_ret'] - (df['trade_entry'] * TRANSACTION_COST)

    # --- Cumulative returns ---
    df['cum_strategy'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    df['cum_asset']    = (1 + df['asset_ret'].fillna(0)).cumprod()

    # --- Sharpe Ratio (annualised, crypto = 365) ---
    strat_rets = df['strategy_ret'].dropna()
    sharpe = (strat_rets.mean() / (strat_rets.std() + 1e-9)) * np.sqrt(365)

    # --- Max Drawdown ---
    cum = df['cum_strategy']
    drawdown = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    max_drawdown = drawdown.min() * 100

    # --- Win Rate (only on active trades, signal != 0) ---
    active = df[df['signal'] != 0]['strategy_ret'].dropna()
    win_rate = (active > 0).mean() * 100 if len(active) > 0 else 0.0

    # --- Trade counts ---
    long_trades  = int((df['signal'] ==  1).sum())
    short_trades = int((df['signal'] == -1).sum())

    return df, {
        'sharpe':        round(sharpe, 3),
        'max_drawdown':  round(max_drawdown, 2),
        'win_rate':      round(win_rate, 2),
        'long_trades':   long_trades,
        'short_trades':  short_trades,
        'total_trades':  long_trades + short_trades
    }


# -----------------------------
# 7) END-TO-END DEMO PIPELINE (combines above)
# -----------------------------

def run_pipeline(ticker='BTC-USD', start='2018-01-01', window_size=10, test_size=0.2, val_frac=0.1, model_choice='XGBoost'):
    print(f"Running pipeline for: {ticker}")
    df = download_yahoo(ticker, start=start)
    df_ind = add_indicators(df)

    feature_cols = ['SMA14','WMA14','MOM14','STCK','STCD','RSI14','MACD','MACD_SIGNAL','LWR14','ADO','CCI20']
    data = create_supervised(df_ind, feature_cols=feature_cols, target_col='Close', n_ahead=1)

    # Normalize features
    df_scaled, scaler = normalize_df(data, feature_cols)

    # Split into train/test (time-aware split is better, but for demo we do sequential split so test is the last portion)
    n_test = int(len(df_scaled) * test_size)
    df_train = df_scaled.iloc[:-n_test]
    df_test = df_scaled.iloc[-n_test:]

    # further split train into train/val
    n_val = int(len(df_train) * val_frac)
    df_val = df_train.iloc[-n_val:]
    df_train = df_train.iloc[:-n_val]

    X_train = df_train[feature_cols].values
    y_train = df_train['target'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['target'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['target'].values

    # Train XGBoost
    xgb_model, xgb_metrics, xgb_preds = train_xgb(X_train, y_train, X_test, y_test)
    print('XGBoost metrics:', xgb_metrics)

    # Train Random Forest
    rf_model, rf_metrics, rf_preds = train_rf(X_train, y_train, X_test, y_test)
    print('RandomForest metrics:', rf_metrics)

    # LSTM requires windowed sequences (we create using the scaled full dataset so indices align)
    # Build windowed sequences from the scaled data (use df_scaled)
    X_windowed, y_windowed = create_windowed_sequences(df_scaled[feature_cols], df_scaled['target'], window_size=window_size)
    # split windowed sequences into train/val/test consistent with earlier splits
    n_total = len(X_windowed)
    n_test_w = int(n_total * test_size)
    n_val_w = int((n_total - n_test_w) * val_frac)

    X_tr_w = X_windowed[:-(n_test_w + n_val_w)]
    y_tr_w = y_windowed[:-(n_test_w + n_val_w)]
    X_val_w = X_windowed[-(n_test_w + n_val_w):-n_test_w] if n_val_w>0 else X_windowed[:0]
    y_val_w = y_windowed[-(n_test_w + n_val_w):-n_test_w] if n_val_w>0 else y_windowed[:0]
    X_test_w = X_windowed[-n_test_w:]
    y_test_w = y_windowed[-n_test_w:]

    lstm_model, lstm_metrics, lstm_preds = train_lstm(X_tr_w, y_tr_w, X_val_w, y_val_w, X_test_w, y_test_w, epochs=30)
    print('LSTM metrics:', lstm_metrics)

    # Save models + scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    lstm_model.save('models/lstm_model.h5')
    print('Models saved to ./models/')

    # Route predictions based on model_choice
    if model_choice == 'RandomForest':
        chosen_preds = rf_preds
        chosen_label = 'RandomForest'
        # RF preds align with X_test (same length)
        df_test_orig_index = df_scaled.iloc[-len(X_test):].index
    elif model_choice == 'LSTM':
        chosen_preds = lstm_preds
        chosen_label = 'LSTM'
        # LSTM test set is shorter due to windowing — align to last n rows
        df_test_orig_index = df_scaled.iloc[-len(lstm_preds):].index
    else:
        chosen_preds = xgb_preds
        chosen_label = 'XGBoost'
        df_test_orig_index = df_scaled.iloc[-len(X_test):].index

    df_back = df_ind.loc[df_test_orig_index].copy()
    # Trim to match pred length (LSTM may differ slightly)
    df_back = df_back.iloc[-len(chosen_preds):]
    df_back['pred'] = chosen_preds

    back, backtest_stats = simple_backtest(df_back, pred_col='pred')

    # Plot cumulative returns
    plt.figure(figsize=(10,6))
    plt.plot(back['cum_asset'], label='Hold Asset')
    plt.plot(back['cum_strategy'], label=f'Strategy ({chosen_label})')
    plt.legend()
    plt.title(f'Cumulative returns: {ticker} [{chosen_label}]')
    plt.show()

    return {
        'xgb_metrics': xgb_metrics,
        'rf_metrics': rf_metrics,
        'lstm_metrics': lstm_metrics,
        'backtest_df': back,
        'backtest_stats': backtest_stats,
        'chosen_label': chosen_label
    }

# -----------------------------
# 8) RUN DEMO when executed directly
# -----------------------------
if __name__ == '__main__':
    out = run_pipeline(ticker='BTC-USD', start='2018-01-01')
    print('\nBacktest head:')
    print(out['backtest_df'][['Close','pred','signal','strategy_ret','cum_strategy']].tail())
    print('\nBacktest stats:')
    print(out['backtest_stats'])

