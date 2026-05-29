# AlgoTrade AI — MarketPulse Optimizer

**An AI-powered algorithmic trading system for cryptocurrency markets using ensemble machine learning and deep learning.**

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Abstract](#abstract)
- [Research Foundation](#research-foundation)
- [System Architecture](#system-architecture)
- [Technical Indicators](#technical-indicators)
- [Machine Learning Models](#machine-learning-models)
- [Backtesting & Performance Metrics](#backtesting--performance-metrics)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Key Findings & Discussion](#key-findings--discussion)
- [Future Work](#future-work)
- [Team & Contributions](#team--contributions)
- [References](#references)
- [License](#license)

---

## Abstract

Financial markets are inherently noisy, non-stationary, and nonlinear — characteristics that render traditional econometric models inadequate for short-term forecasting. **AlgoTrade AI — MarketPulse Optimizer** addresses this challenge by constructing a supervised learning framework that predicts next-day cryptocurrency returns using eleven hand-engineered technical indicators as input features. Three model architectures — **XGBoost**, **Random Forest**, and **Long Short-Term Memory (LSTM)** — are trained on historical price data, evaluated on a time-preserving holdout set, and deployed through a live signal generator. The system incorporates transaction-cost-aware backtesting with risk-adjusted performance metrics (Sharpe ratio, maximum drawdown, directional accuracy, and win rate), enabling rigorous comparison against a passive buy-and-hold baseline.

This work demonstrates that gradient-boosted tree ensembles (XGBoost) consistently outperform both bagging-based ensembles and recurrent neural architectures on structured tabular market data, achieving directional accuracy exceeding 55% — a statistically meaningful edge over random guessing in high-frequency financial time series.

---

## Research Foundation

This project draws methodological inspiration from the following literature:

| Paper | Relevance |
|-------|-----------|
| **Kumar et al. (2020)** — *"A Deep Learning Framework for Stock Price Prediction"* | Established the viability of LSTM for financial time series forecasting |
| **Chen & Guestrin (2016)** — *"XGBoost: A Scalable Tree Boosting System"* | Formalised the gradient boosting framework used as our primary model |
| **Fischer & Krauss (2018)** — *"Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions"* | Demonstrated LSTM's ability to capture temporal dependencies in S&P 500 constituents |
| **Wilder (1978)** — *"New Concepts in Technical Trading Systems"* | Original formulation of RSI, the indicator consistently ranked most important by our models |

The problem is framed as a **regression task** predicting the one-day-ahead percentage return rather than raw price — a deliberate design choice that:

1. **Ensures stationarity**: Percentage returns are inherently more stationary than raw prices, satisfying critical assumptions of most ML models
2. **Enables cross-asset signal comparability**: A 1% prediction for BTC-USD and ETH-USD carries the same decision threshold
3. **Aligns with portfolio theory**: Expected returns (not absolute prices) drive capital allocation decisions

---

## System Architecture

```
┌─────────────────┐
│  Yahoo Finance   │  ← Live OHLCV data via yfinance API
│      API         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion  │  ← pandas: flatten MultiIndex, select columns
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature       │  ← Compute 11 technical indicators
│  Engineering     │     (SMA, WMA, MOM, STCK, STCD, RSI, MACD,
│                  │      MACD_SIGNAL, LWR, ADO, CCI)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Supervised     │  ← Target: (Closeₜ₊₁ − Closeₜ) / Closeₜ
│   Dataset        │     Features: normalised indicator values
└────────┬────────┘
         │
         ├────────────────────┬──────────────────────┐
         ▼                    ▼                      ▼
   ┌──────────┐       ┌──────────────┐       ┌──────────┐
   │ XGBoost  │       │ RandomForest │       │   LSTM   │
   │ n_est=200│       │ n_est=200    │       │ 64 units │
   │ depth=6  │       │ depth=10     │       │ win=10d  │
   │ lr=0.05  │       │ n_jobs=−1    │       │ ES pat=8 │
   └────┬─────┘       └──────┬───────┘       └────┬─────┘
        │                    │                     │
        ▼                    ▼                     ▼
   ┌─────────────────────────────────────────────────────┐
   │               Signal Generator                       │
   │   pred > +0.1% → BUY (+1)                            │
   │   pred < −0.1% → SELL (−1)                           │
   │   |pred| ≤ 0.1% → HOLD (0)                           │
   └──────────────────────┬──────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │               Backtest Engine                        │
   │   • Transaction cost: 0.1% per trade entry           │
   │   • Sharpe Ratio (annualised, 365-day year)          │
   │   • Maximum Drawdown                                 │
   │   • Win Rate (active trades only)                    │
   │   • Cumulative return curves vs Buy & Hold baseline  │
   └──────────────────────┬──────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │          Live Dashboard (Flask + Streamlit)          │
   │   • Real-time XGBoost signal (5-min refresh)         │
   │   • Interactive Plotly charts                        │
   │   • Investment calculator                            │
   │   • Feature importance analysis                      │
   │   • Radar chart model comparison                     │
   └─────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Predict returns, not prices | Returns are stationary → models converge faster and generalise better |
| 0.1% signal threshold | Filters noise below realistic transaction costs on major exchanges |
| Time-preserving train/test split | Prevents look-ahead bias; test set represents true "future" data |
| Feature normalisation via MinMaxScaler | Prevents magnitude-dominated features (e.g., ADO) from overpowering bounded indicators (e.g., RSI) |
| Dual interface (Streamlit + Flask) | Streamlit for rapid prototyping; Flask + vanilla JS for production-grade deployment |

---

## Technical Indicators

Eleven technical indicators are computed from raw OHLCV data, each capturing a distinct market phenomenon:

| # | Indicator | Abbreviation | Category | Interpretation |
|---|-----------|-------------|----------|----------------|
| 1 | Simple Moving Average | SMA₁₄ | Trend | Mean closing price over 14 periods |
| 2 | Weighted Moving Average | WMA₁₄ | Trend | Linearly-weighted mean (recent prices weighted higher) |
| 3 | Momentum | MOM₁₄ | Trend | Price difference over 14 periods |
| 4 | Stochastic %K | STCK | Oscillator | Position of close within high-low range |
| 5 | Stochastic %D | STCD | Oscillator | 3-period SMA of %K — signal line |
| 6 | Relative Strength Index | RSI₁₄ | Oscillator | Normalised magnitude of recent gains vs losses |
| 7 | MACD | MACD | Trend-Momentum | Difference of 12-period and 26-period EMAs |
| 8 | MACD Signal Line | MACD_SIGNAL | Trend-Momentum | 9-period EMA of MACD |
| 9 | Williams %R | LWR₁₄ | Oscillator | Inverse of Stochastic — close relative to high-low range |
| 10 | Accumulation/Distribution | ADO | Volume | Cumulative money flow (price movement × volume) |
| 11 | Commodity Channel Index | CCI₂₀ | Oscillator | Deviation of typical price from its SMA |

These indicators were selected based on their prevalence in academic technical analysis literature and their complementarity — covering trend-following, mean-reversion, momentum, and volume-based signals simultaneously.

---

## Machine Learning Models

### XGBoost (`n_estimators=200, learning_rate=0.05, max_depth=6`)

> *"An expert voting panel where each expert's weakness is addressed by the next."*

XGBoost is a gradient-boosted tree ensemble that sequentially adds weak learners (shallow decision trees), each correcting the residual errors of its predecessors. The algorithm minimises a regularised objective:

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

where $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ penalises tree complexity.

**Why it excels on tabular market data:**
- **Nonlinear interactions**: Trees automatically capture interactions between indicators without manual feature crosses
- **Robustness to outliers**: Loss function is less sensitive to extreme market movements than MSE-based models
- **Built-in regularisation**: L1/L2 penalties prevent overfitting to noise in financial data

### Random Forest (`n_estimators=200, max_depth=10, n_jobs=−1`)

> *"Many independent analysts vote independently; the majority wins."*

Random Forest builds an ensemble of de-correlated decision trees through bootstrap aggregation (bagging) and random feature subset selection. Each tree is trained on a bootstrap sample of the data with only a random subset of features considered at each split.

**Strengths:**
- Lower variance than single decision trees
- Naturally handles feature importance estimation via out-of-bag error
- Parallelisable (`n_jobs=−1` uses all CPU cores)

### LSTM (`units=64, sequence_length=10, EarlyStopping patience=8`)

> *"An analyst with memory — each prediction reads the last 10 days as a single sequence."*

The LSTM architecture addresses the vanishing gradient problem in recurrent neural networks through gating mechanisms:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}$$

Input data is structured as `[samples, sequence_length, features]` — i.e., each training example is a 10-day × 11-indicator tensor.

**Training configuration:**
- Optimiser: Adam
- Loss: Mean Squared Error
- Batch size: 32
- Max epochs: 30 (with EarlyStopping on validation loss, patience=8)

---

## Backtesting & Performance Metrics

### Signal Generation Rule

| Condition | Signal | Position |
|-----------|--------|----------|
| $\hat{r}_{t+1} > +0.001$ | `+1` (BUY) | Long |
| $\hat{r}_{t+1} < -0.001$ | `−1` (SELL) | Short |
| $|\hat{r}_{t+1}| \leq 0.001$ | `0` (HOLD) | No position |

The ±0.1% threshold is calibrated to typical crypto exchange transaction costs — predictions within this band would yield negative expected profit after fees.

### Performance Metrics

| Metric | Formula / Definition | Interpretation |
|--------|---------------------|----------------|
| **Directional Accuracy (DA)** | $\frac{1}{n}\sum \mathbb{1}[\text{sgn}(\hat{y}_i) = \text{sgn}(y_i)] \times 100$ | % of days where predicted direction matches actual. >50% = better than coin flip |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Average magnitude of prediction error |
| **Sharpe Ratio** | $\frac{\bar{r}_{strategy}}{\sigma_{strategy}} \times \sqrt{365}$ | Return per unit of risk, annualised. >1.0 = good |
| **Maximum Drawdown** | $\min_t \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s} \times 100$ | Worst peak-to-trough decline. Lower is safer |
| **Win Rate** | % of active trades (signal ≠ 0) with positive return | Consistency of profitable signals |

### Backtest Assumptions

- **Transaction cost**: 0.1% per trade entry (representative of Binance/Bitstamp spot trading fees)
- **Execution price**: Closing price of signal day (conservative — no intraday slippage advantage)
- **Data frequency**: Daily OHLCV
- **Short selling**: Allowed (signal = −1 assumes a short position, gaining from declining prices)

---

## Project Structure

```
PROJECT CODE/
│
├── app.py                    # Streamlit application (interactive dashboard)
├── flask_app.py              # Flask REST API + web application backend
├── main.py                   # Core ML pipeline (data, indicators, training, backtesting)
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── templates/
│   └── index.html            # Flask frontend (vanilla HTML/CSS/JS + Plotly.js)
│
├── static/
│   ├── css/
│   │   └── style.css         # Dark theme CSS (Streamlit-faithful design)
│   └── js/
│       └── main.js           # Client-side logic (API calls, chart rendering)
│
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model (joblib)
│   ├── rf_model.pkl          # Trained Random Forest model (joblib)
│   ├── lstm_model.h5         # Trained LSTM model (Keras HDF5)
│   └── scaler.pkl            # Fitted MinMaxScaler
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd "PROJECT CODE"

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import xgboost, tensorflow, flask, streamlit; print('All dependencies OK')"
```

### Required Packages (`requirements.txt`)

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow==2.12.0
yfinance>=0.2.30
joblib>=1.3.0
matplotlib>=3.7.0
streamlit>=1.35.0
flask>=3.0.0
plotly>=5.18.0
```

---

## Usage

### Option 1: Streamlit Dashboard (Recommended for Exploration)

```bash
streamlit run app.py
```

Opens an interactive web dashboard at `http://localhost:8501` featuring:

- **Live signal banner**: Real-time XGBoost prediction for any Yahoo Finance ticker (5-minute auto-refresh)
- **Configurable backtest**: Select cryptocurrency, date range, and model architecture
- **Interactive visualisations**: Plotly charts for cumulative returns, feature importance, and model comparison radar
- **Investment calculator**: Simulate exact profit/loss for any investment amount
- **CSV export**: Download full backtest results

### Option 2: Flask Web Application (Production)

```bash
python flask_app.py
```

Launches a Flask server at `http://127.0.0.1:5000` with a REST API and responsive web interface. The backend supports asynchronous pipeline execution via UUID-tracked jobs.

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/api/live-signal` | Get real-time trading signal |
| `POST` | `/api/run-pipeline` | Trigger full training + backtest pipeline |
| `GET` | `/api/job/<job_id>` | Poll pipeline job status |
| `GET` | `/api/download/<job_id>` | Download backtest CSV |

### Option 3: Direct Python Pipeline

```python
from main import run_pipeline

output = run_pipeline(
    ticker='ETH-USD',       # any Yahoo Finance ticker
    start='2018-01-01',     # training data start date
    window_size=10,          # LSTM sequence length
    model_choice='XGBoost'   # 'XGBoost' | 'RandomForest' | 'LSTM'
)

print("Backtest stats:", output['backtest_stats'])
print("XGBoost metrics:", output['xgb_metrics'])
```

---

## Experimental Results

Performance is evaluated on a time-preserving test split (most recent 20% of available data). Below are representative results for BTC-USD trained from 2018-01-01.

### Model Comparison

| Metric | XGBoost | RandomForest | LSTM |
|--------|---------|-------------|------|
| **Direction Accuracy** | ~56–60% | ~54–58% | ~52–56% |
| **RMSE** | ~0.025 | ~0.026 | ~0.028 |
| **Training Speed** | Fast (seconds) | Fast (seconds) | Slow (minutes) |
| **Interpretability** | High (feature importance) | High (feature importance) | Low (black box) |

### Backtest Performance (XGBoost Strategy vs Buy & Hold)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 0.8–1.4 (varies by market regime and training period) |
| **Maximum Drawdown** | Typically 15–35% |
| **Win Rate (active trades)** | 50–58% |
| **Signal frequency** | BUY ~ | HOLD ~ | SELL ~ (market-dependent)

> **Note**: Cryptocurrency markets exhibit regime-dependent behaviour. Performance varies across bull, bear, and sideways market phases. The 0.1% noise threshold filters approximately 40–60% of days as HOLD, reducing overtrading.

---

## Key Findings & Discussion

### 1. Tree-Based Models Outperform LSTM on Tabular Indicator Data

Contrary to the prevailing narrative that deep learning dominates financial forecasting, our experiments show that XGBoost and Random Forest consistently achieve higher directional accuracy than LSTM when using the same 11-engineered features. This aligns with recent literature indicating that gradient boosting on well-engineered features often matches or exceeds deep learning on structured tabular problems — especially with daily-frequency data where the signal-to-noise ratio is low.

### 2. The Signal Threshold Is Critical

Without the ±0.1% threshold, the strategy generates trades on nearly every day, incurring transaction costs that erode any predictive edge. The threshold acts as a **statistical significance filter** — only acting on predictions strong enough to survive real-world friction costs.

### 3. Feature Importance Validates Indicator Selection

XGBoost's built-in feature importance consistently ranks RSI, MACD, and Stochastic %K among the top indicators — validating that the indicator set is not arbitrary but empirically justified. The Accumulation/Distribution line (ADO) typically carries the least weight, suggesting volume-based signals are less informative at daily frequency.

### 4. Returns Prediction Beats Price Prediction

Framing the problem as return prediction rather than price prediction eliminates the non-stationarity problem that plagues most financial ML projects. Models trained to predict raw prices simply learn yesterday's price; models trained to predict returns must learn the actual signal.

---

## Future Work

The following extensions are identified as high-impact directions for continued research:

1. **Multi-Asset Portfolio Optimisation**: Extend from single-asset signals to a Markowitz- or Black-Litterman-based portfolio allocation using covariance between predicted returns
2. **Alternative Data Sources**: Incorporate on-chain metrics (wallet activity, exchange flows, hash rate), sentiment from social media (NLP on Twitter/Reddit), and macroeconomic indicators
3. **Reinforcement Learning Agent**: Replace the static threshold-based signal generator with a Deep Q-Network (DQN) or PPO agent that learns optimal position sizing and trade timing
4. **Probabilistic Forecasting**: Move from point predictions to full predictive distributions using Quantile Regression or Bayesian Neural Networks — enabling Value at Risk (VaR) and Expected Shortfall estimation
5. **High-Frequency Data**: Test model performance on 1-hour or 15-minute candles with additional microstructure features (bid-ask spread, order book depth)
6. **Adversarial Robustness**: Evaluate strategy resilience against adversarial market conditions using stress-testing and Monte Carlo simulation

---

## Competencies Demonstrated

This project was developed as a team of four undergraduate students. The work demonstrates applied competence across the following areas — all of which are directly relevant to graduate-level study in Artificial Intelligence and Machine Learning.

### Core ML & Data Science

- **Supervised learning pipeline design**: End-to-end implementation from raw data ingestion through feature engineering, model training, evaluation, and deployment
- **Ensemble methods**: Gradient boosting (XGBoost) and bagging (Random Forest) with hyperparameter optimisation and comparative analysis
- **Deep learning for time series**: LSTM architecture design, windowed sequence generation, EarlyStopping regularisation, and interpretation of recurrent model behaviour on financial data
- **Feature engineering from domain knowledge**: Translation of financial market microstructure into 11 quantitative indicators spanning trend, momentum, oscillator, and volume categories
- **Statistical evaluation**: Directional accuracy, RMSE, Sharpe ratio, maximum drawdown, and win rate — with critical awareness of which metrics are misleading in financial contexts (e.g., why RMSE alone is uninformative for return prediction)
- **Time-series-aware validation**: Time-preserving train/validation/test splits to prevent look-ahead bias, a critical technique rarely applied in undergraduate projects

### Software Engineering

- **Dual-interface architecture**: Streamlit for rapid experimentation; Flask + vanilla JavaScript/Plotly.js + CSS for production web deployment
- **Asynchronous pipeline execution**: UUID-tracked background jobs with polling-based status updates
- **REST API design**: Clean endpoint structure with JSON serialisation, error handling, and CSV download support
- **Model persistence and reproducibility**: Serialisation of trained models (joblib, HDF5) and fitted scalers for consistent inference

### Financial Domain Knowledge

- **Technical analysis theory**: Understanding of indicator families and their economic interpretations
- **Transaction-cost-aware backtesting**: Realistic simulation incorporating exchange fees and signal-entry cost deduction
- **Risk-adjusted performance measurement**: Sharpe ratio interpretation, drawdown analysis, regime-awareness in strategy evaluation

---

## Relevance to German AI/ML Research

This project intersects with several active research areas in German universities:

| Research Area | German Institutions Active in This Space | Project Relevance |
|---------------|------------------------------------------|-------------------|
| **Time Series Forecasting** | TU Munich, University of Tübingen, Hasso Plattner Institute | LSTM + XGBoost comparison on non-stationary financial data |
| **Financial Machine Learning** | Goethe University Frankfurt, Karlsruhe Institute of Technology | Transaction-cost-aware backtesting, signal threshold optimisation |
| **Explainable AI (XAI)** | Fraunhofer IAIS, TU Berlin, University of Bonn | Feature importance analysis validates which indicators drive predictions |
| **Reinforcement Learning for Trading** | University of Freiburg, TU Darmstadt | Identified as primary future work direction (DQN/PPO for position sizing) |
| **Applied Deep Learning** | Max Planck Institute for Intelligent Systems (Tübingen) | LSTM architecture design with EarlyStopping for noise-prone financial data |

---

## References

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD.
2. Fischer, T., & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669.
3. Kumar, D., Sarangi, P. K., & Verma, R. (2020). *A systematic review of stock market prediction using machine learning*. Artificial Intelligence Review.
4. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
5. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
6. Sharpe, W. F. (1994). *The Sharpe Ratio*. Journal of Portfolio Management, 21(1), 49–58.

---

<p align="center"><i>"In trading, data is the new alpha."</i></p>
