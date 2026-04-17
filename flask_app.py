"""
AlgoTrade AI — Flask Web Application
Replaces Streamlit with a proper Flask + HTML/CSS/JS website.
"""

import os
import sys
import uuid
import threading
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be before pyplot
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from io import BytesIO
from flask import Flask, request, jsonify, render_template, Response

warnings.filterwarnings('ignore')

# Ensure main.py is importable (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from main import (
    run_pipeline, download_yahoo, add_indicators,
    create_supervised, normalize_df
)

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)

FEATURE_COLS = [
    'SMA14', 'WMA14', 'MOM14', 'STCK', 'STCD',
    'RSI14', 'MACD', 'MACD_SIGNAL', 'LWR14', 'ADO', 'CCI20'
]
FEATURE_LABELS = {
    'SMA14':        'Simple Moving Avg (14d)',
    'WMA14':        'Weighted Moving Avg (14d)',
    'MOM14':        'Momentum (14d)',
    'STCK':         'Stochastic %K',
    'STCD':         'Stochastic %D',
    'RSI14':        'RSI — Relative Strength',
    'MACD':         'MACD',
    'MACD_SIGNAL':  'MACD Signal Line',
    'LWR14':        'Williams %R (14d)',
    'ADO':          'Accumulation/Distribution',
    'CCI20':        'Commodity Channel Index (20d)',
}

# In-memory job store
jobs: dict = {}


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _safe(v):
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _serialize_metrics(m: dict) -> dict:
    return {k: round(float(v), 6) for k, v in m.items()}


def _serialize_backtest(df: pd.DataFrame) -> dict:
    cols = ['Close', 'pred', 'signal', 'strategy_ret', 'cum_strategy', 'cum_asset']
    df2 = df[cols].copy()
    df2 = df2.replace([np.inf, -np.inf], 0).fillna(0)
    idx = df2.index.strftime('%Y-%m-%d') if hasattr(df2.index, 'strftime') else [str(i) for i in df2.index]
    return {
        'dates':        idx.tolist(),
        'close':        [round(float(v), 4) for v in df2['Close']],
        'pred':         [round(float(v), 6) for v in df2['pred']],
        'signal':       [int(v) for v in df2['signal']],
        'strategy_ret': [round(float(v), 6) for v in df2['strategy_ret']],
        'cum_strategy': [round(float(v), 4) for v in df2['cum_strategy']],
        'cum_asset':    [round(float(v), 4) for v in df2['cum_asset']],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/live-signal', methods=['POST'])
def live_signal_api():
    data   = request.get_json() or {}
    ticker = data.get('ticker', 'BTC-USD')
    try:
        since = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')
        df    = download_yahoo(ticker, start=since)
        if df is None or len(df) < 30:
            return jsonify({'error': 'Insufficient data for ticker'}), 400

        df      = add_indicators(df)
        dataset = create_supervised(df, FEATURE_COLS)
        if len(dataset) < 20:
            return jsonify({'error': 'Not enough processed rows'}), 400

        X = dataset[FEATURE_COLS].values
        y = dataset['target'].values

        model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1,
            max_depth=4, verbosity=0
        )
        model.fit(X[:-1], y[:-1])
        pred = float(model.predict(X[-1].reshape(1, -1))[0])

        close_today = float(df['Close'].iloc[-1])
        close_prev  = float(df['Close'].iloc[-2])
        day_change  = (close_today - close_prev) / close_prev * 100

        if pred > 0.001:
            action, color = 'BUY',  '#00e676'
        elif pred < -0.001:
            action, color = 'SELL', '#ff1744'
        else:
            action, color = 'HOLD', '#ffea00'

        signal_strength = (
            'Strong'   if abs(pred) > 0.005 else
            'Moderate' if abs(pred) > 0.001 else
            'Weak'
        )

        return jsonify({
            'action':          action,
            'color':           color,
            'pred_pct':        round(pred * 100, 4),
            'close':           round(close_today, 2),
            'day_change':      round(day_change, 2),
            'as_of':           df.index[-1].strftime('%d %b %Y'),
            'signal_strength': signal_strength,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline_api():
    data         = request.get_json() or {}
    ticker       = data.get('ticker', 'BTC-USD')
    start_date   = data.get('start_date', '2018-01-01')
    window_size  = int(data.get('window_size', 10))
    model_choice = data.get('model_choice', 'XGBoost')

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'running'}

    def worker():
        try:
            os.chdir(BASE_DIR)  # models saved relative to BASE_DIR
            output = run_pipeline(
                ticker=ticker,
                start=start_date,
                window_size=window_size,
                model_choice=model_choice
            )
            plt.close('all')

            # Feature importances from saved XGBoost model
            feature_importance = {}
            xgb_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
            if os.path.exists(xgb_path):
                try:
                    saved_xgb = joblib.load(xgb_path)
                    imps = saved_xgb.feature_importances_
                    feature_importance = {
                        FEATURE_LABELS[FEATURE_COLS[i]]: round(float(imps[i]), 5)
                        for i in range(len(FEATURE_COLS))
                    }
                except Exception:
                    pass

            back_df = output['backtest_df']
            result = {
                'ticker':        ticker,
                'chosen_label':  output['chosen_label'],
                'start_date':    start_date,
                'xgb_metrics':   _serialize_metrics(output['xgb_metrics']),
                'rf_metrics':    _serialize_metrics(output['rf_metrics']),
                'lstm_metrics':  _serialize_metrics(output['lstm_metrics']),
                'backtest_stats': {k: _safe(v) for k, v in output['backtest_stats'].items()},
                'backtest':      _serialize_backtest(back_df),
                'feature_importance': feature_importance,
                'final_strategy': round(float(back_df['cum_strategy'].iloc[-1]), 4),
                'final_buyhold':  round(float(back_df['cum_asset'].iloc[-1]), 4),
            }
            jobs[job_id] = {'status': 'done', 'result': result}
        except Exception as e:
            import traceback
            jobs[job_id] = {
                'status': 'error',
                'error':  str(e),
                'trace':  traceback.format_exc()
            }

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>', methods=['GET'])
def get_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@app.route('/api/download/<job_id>', methods=['GET'])
def download_csv(job_id):
    job = jobs.get(job_id)
    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Job not ready'}), 404

    result   = job['result']
    backtest = result['backtest']
    df = pd.DataFrame({
        'Date':                backtest['dates'],
        'Close':               backtest['close'],
        'Predicted Return':    backtest['pred'],
        'Signal':              backtest['signal'],
        'Strategy Return':     backtest['strategy_ret'],
        'Cumulative Strategy': backtest['cum_strategy'],
        'Cumulative Asset':    backtest['cum_asset'],
    })
    buf = BytesIO(df.to_csv(index=False).encode('utf-8'))
    buf.seek(0)
    fname = f"{result['ticker']}_backtest_{result['chosen_label']}.csv"
    return Response(
        buf,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={fname}'}
    )


if __name__ == '__main__':
    print("AlgoTrade AI — starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
