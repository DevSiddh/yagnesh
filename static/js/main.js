'use strict';

// ── Globals ──────────────────────────────────────────────────────────────────
let currentJobId   = null;
let pollTimer      = null;
let pipelineResult = null;

// ── Plotly dark layout (matching app.py's template="plotly_dark") ─────────────
const DARK = {
  template:      'plotly_dark',
  paper_bgcolor: '#0e1117',
  plot_bgcolor:  '#0e1117',
  font:          { color: '#fafafa', family: 'Source Sans Pro, Segoe UI, system-ui, sans-serif', size: 13 },
};

const $ = id => document.getElementById(id);

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
    $('tab-' + tab).style.display = 'block';
    if (tab === 'architecture') renderArchitecture();
  });
});

// ── LSTM window label ─────────────────────────────────────────────────────────
$('cfg-window').addEventListener('input', e => {
  $('window-val').textContent  = e.target.value;
  $('window-val2').textContent = e.target.value;
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(v, d = 2) {
  return Number(v).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
}
function rateDA(v)    { return v >= 60 ? ['🟢','Great'] : v >= 53 ? ['🟡','Okay'] : ['🔴','Needs work']; }
function rateSharpe(v){ return v >= 1.0 ? ['🟢','Great'] : v >= 0.5 ? ['🟡','Okay'] : ['🔴','Needs work']; }
function rateDD(v)    { const a=Math.abs(v); return a<=20?['🟢','Great']:a<=40?['🟡','Okay']:['🔴','Needs work']; }
function rateWR(v)    { return v >= 55 ? ['🟢','Great'] : v >= 50 ? ['🟡','Okay'] : ['🔴','Needs work']; }

// ── Live Signal ───────────────────────────────────────────────────────────────
function fetchLiveSignal() {
  const ticker = $('cfg-ticker').value.trim() || 'BTC-USD';
  $('live-title').textContent = `⚡ Live Signal — What should you do with ${ticker} RIGHT NOW?`;
  $('live-spinner').style.display = 'flex';
  $('live-result').style.display  = 'none';
  $('live-error').style.display   = 'none';

  fetch('/api/live-signal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker }),
  })
    .then(r => r.json())
    .then(d => {
      $('live-spinner').style.display = 'none';
      if (d.error) {
        $('live-error').style.display   = 'block';
        $('live-error').textContent     = 'Could not fetch live signal. Check ticker symbol or internet connection.';
        return;
      }

      const card = $('live-card');
      card.style.borderColor = d.color;
      card.style.background  = d.color + '22';

      $('live-emoji').textContent       = d.action === 'BUY' ? '🟢' : d.action === 'SELL' ? '🔴' : '⬜';
      $('live-action').textContent      = d.action;
      $('live-action').style.color      = d.color;
      $('live-asof').textContent        = 'AI Recommendation as of ' + d.as_of;

      $('m-price').textContent          = '$' + fmt(d.close, 2);
      $('m-change').textContent         = (d.day_change >= 0 ? '+' : '') + fmt(d.day_change, 2) + '% today';
      $('m-change').className           = 'metric-delta ' + (d.day_change >= 0 ? 'pos' : 'neg');

      $('m-direction').textContent      = d.pred_pct >= 0 ? '📈 Rise' : '📉 Fall';
      $('m-pred').textContent           = (d.pred_pct >= 0 ? '+' : '') + fmt(d.pred_pct, 4) + '% expected return';
      $('m-pred').className             = 'metric-delta ' + (d.pred_pct >= 0 ? 'pos' : 'neg');

      $('m-strength').textContent       = d.signal_strength;
      $('m-strength-sub').textContent   = Math.abs(d.pred_pct) > 0.1 ? 'above noise threshold' : 'within noise band';

      $('live-steps').innerHTML = `
        <li>Downloads <strong>last 120 days</strong> of ${ticker} prices from Yahoo Finance (right now)</li>
        <li>Computes all <strong>11 technical indicators</strong> from that fresh data</li>
        <li>Trains a <strong>quick XGBoost model</strong> on those 120 days (in memory, takes seconds)</li>
        <li>Predicts the <strong>expected return for tomorrow</strong></li>
        <li>Applies the rule: predicted return <strong>&gt; +0.1%</strong> → BUY | <strong>&lt; −0.1%</strong> → SELL | in between → HOLD</li>
      `;
      $('live-expander-title').textContent = `📖 How is this live signal generated?`;

      $('live-result').style.display = 'block';
    })
    .catch(() => {
      $('live-spinner').style.display = 'none';
      $('live-error').style.display   = 'block';
      $('live-error').textContent     = 'Could not fetch live signal. Check ticker symbol or internet connection.';
    });
}

fetchLiveSignal();

let liveDebounce = null;
$('cfg-ticker').addEventListener('input', () => {
  clearTimeout(liveDebounce);
  liveDebounce = setTimeout(fetchLiveSignal, 900);
});

// ── Run Pipeline ──────────────────────────────────────────────────────────────
$('run-btn').addEventListener('click', () => {
  const ticker       = $('cfg-ticker').value.trim() || 'BTC-USD';
  const start_date   = $('cfg-start').value || '2018-01-01';
  const model_choice = $('cfg-model').value;
  const window_size  = parseInt($('cfg-window').value);

  $('run-btn').disabled            = true;
  $('pre-run').style.display       = 'none';
  $('run-progress').style.display  = 'block';
  $('results').style.display       = 'none';
  $('progress-sub').textContent    = `${ticker} · ${model_choice} · from ${start_date}`;

  fetch('/api/run-pipeline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker, start_date, window_size, model_choice }),
  })
    .then(r => r.json())
    .then(data => {
      currentJobId = data.job_id;
      pollTimer    = setInterval(pollJob, 3000);
    })
    .catch(err => pipelineError('Failed to start: ' + err.message));
});

function pollJob() {
  fetch('/api/job/' + currentJobId)
    .then(r => r.json())
    .then(data => {
      if (data.status === 'done') {
        clearInterval(pollTimer);
        showResults(data.result);
      } else if (data.status === 'error') {
        clearInterval(pollTimer);
        pipelineError(data.error);
      }
    })
    .catch(err => { clearInterval(pollTimer); pipelineError(err.message); });
}

function pipelineError(msg) {
  $('run-btn').disabled           = false;
  $('run-progress').style.display = 'none';
  $('pre-run').style.display      = 'block';
  alert('Pipeline error:\n' + msg);
}

// ── Show Results ──────────────────────────────────────────────────────────────
function showResults(r) {
  pipelineResult = r;
  $('run-btn').disabled           = false;
  $('run-progress').style.display = 'none';
  $('results').style.display      = 'block';

  const { ticker, chosen_label: chosen, xgb_metrics: xm, rf_metrics: rfm,
          lstm_metrics: lm, backtest_stats: st, backtest: bt,
          feature_importance: fi, final_strategy: fs, final_buyhold: fb } = r;

  const outperf = fs > fb;
  const diff    = Math.abs(fs - fb).toFixed(3);
  const chosenM = chosen === 'LSTM' ? lm : chosen === 'RandomForest' ? rfm : xm;

  // ── Success banner ──────────────────────────────────────────────────────
  $('success-banner').innerHTML = `✅ Full analysis complete for <strong>${ticker}</strong> — strategy driven by <strong>${chosen}</strong>`;
  $('success-banner').style.display = 'block';

  // ── Plain English Summary ───────────────────────────────────────────────
  const arrow = outperf ? '📈' : '📉';
  const word  = outperf ? 'outperformed ✅' : 'underperformed ❌';
  const [eDA, vDA]   = rateDA(chosenM.DA);
  const [eSH, vSH]   = rateSharpe(st.sharpe);

  $('summary-left').innerHTML = `
    <strong>If you invested ₹1,000 at the start of this period:</strong><br><br>
    🤖 Following our AI signals → <strong>₹${fmt(fs * 1000, 0)}</strong><br>
    🧍 Just holding ${ticker} → <strong>₹${fmt(fb * 1000, 0)}</strong><br><br>
    ${arrow} Our strategy <strong>${word}</strong> Buy &amp; Hold by <strong>${diff}×</strong>
  `;
  $('summary-right').innerHTML = `
    <strong>Quick health check:</strong><br><br>
    ${eDA} AI Direction Accuracy: <strong>${chosenM.DA.toFixed(1)}%</strong> — ${vDA}<br>
    ${eSH} Risk-Adjusted Return (Sharpe): <strong>${st.sharpe}</strong> — ${vSH}<br>
    🎯 Win Rate on trades: <strong>${st.win_rate}%</strong><br>
    📉 Worst loss period: <strong>${st.max_drawdown}%</strong>
  `;

  // ── Model Cards ─────────────────────────────────────────────────────────
  const cards = $('model-cards');
  cards.innerHTML = '';
  [['XGBoost', xm], ['RandomForest', rfm], ['LSTM', lm]].forEach(([name, m]) => {
    const [em, vm] = rateDA(m.DA);
    const badge    = name === chosen ? '<span class="chosen-badge">← selected</span>' : '';
    const div = document.createElement('div');
    div.className = 'model-card' + (name === chosen ? ' chosen' : '');
    div.innerHTML = `
      <h4>${name}${badge}</h4>
      <div class="da-big">${m.DA.toFixed(1)}%</div>
      <div class="verdict">${em} <strong>${vm}</strong></div>
      <div style="font-size:12px;color:#a3a8b8;margin-top:4px">Prediction Error: <code>${m.RMSE.toFixed(6)}</code></div>
      <details class="expander" style="margin-top:0.75rem">
        <summary>📖 Explain</summary>
        <div class="expander-body">
          <ul>
            <li><strong>${m.DA.toFixed(1)}% Direction Accuracy</strong> → model was right <strong>${Math.round(m.DA)} out of 100 days</strong></li>
            <li><strong>RMSE ${m.RMSE.toFixed(6)}</strong> → average prediction error (daily returns are tiny decimals — this is normal)</li>
            <li>50% = coin flip. <strong>Above ~55% = the AI is genuinely learning market patterns.</strong></li>
          </ul>
        </div>
      </details>
    `;
    cards.appendChild(div);
  });

  // ── Radar Chart ─────────────────────────────────────────────────────────
  const cats = ['Direction Accuracy', 'Win Rate (proxy)', 'Sharpe Score', 'Low Error (inv RMSE)'];
  const maxRMSE = Math.max(xm.RMSE, rfm.RMSE, lm.RMSE) + 1e-9;
  function radarVals(m) {
    return [
      m.DA,
      Math.min(m.DA * 0.9, 100),
      Math.min(Math.max(st.sharpe / 3.0 * 100, 0), 100),
      (1 - m.RMSE / maxRMSE) * 100,
    ];
  }
  Plotly.newPlot('chart-radar',
    [['XGBoost', xm, '#00b4d8'], ['RandomForest', rfm, '#90e0ef'], ['LSTM', lm, '#48cae4']].map(([name, m, col]) => {
      const v = radarVals(m); const vc = [...v, v[0]]; const cc = [...cats, cats[0]];
      return { type:'scatterpolar', r:vc, theta:cc, fill:'toself', name, line:{color:col}, opacity:0.7 };
    }),
    { ...DARK, polar:{radialaxis:{visible:true,range:[0,100],color:'#a3a8b8'}},
      showlegend:true, height:420, margin:{t:40,b:40,l:40,r:40} },
    { responsive:true }
  );

  // ── Feature Importance ───────────────────────────────────────────────────
  if (fi && Object.keys(fi).length > 0) {
    const sorted  = Object.entries(fi).sort((a,b) => a[1]-b[1]);
    const labels  = sorted.map(x => x[0]);
    const vals    = sorted.map(x => x[1]);
    const topName = sorted[sorted.length-1][0];
    const topVal  = sorted[sorted.length-1][1];
    Plotly.newPlot('chart-importance',
      [{ type:'bar', orientation:'h', x:vals, y:labels,
         marker:{color:vals, colorscale:'Teal', showscale:false},
         text:vals.map(v=>v.toFixed(3)), textposition:'outside' }],
      { ...DARK, title:'XGBoost Feature Importance — Which indicator matters most?',
        xaxis:{title:'Importance Score', color:'#a3a8b8'},
        yaxis:{color:'#a3a8b8'},
        height:420, margin:{l:220,r:80,t:60,b:40} },
      { responsive:true }
    );
    $('top-feature-msg').className   = 'box-success';
    $('top-feature-msg').innerHTML   = `🏆 <strong>Most important indicator: ${topName}</strong> (score: ${topVal.toFixed(3)}) — the AI found this pattern most predictive of next-day price direction.`;
    $('top-feature-msg').style.display = 'block';
  }

  // ── Backtest Stats ───────────────────────────────────────────────────────
  $('backtest-caption').textContent = `Simulating ${ticker} trading using ${chosen} signals · includes 0.1% transaction cost per trade`;
  const [eSH2,vSH2] = rateSharpe(st.sharpe);
  const [eDD,vDD]   = rateDD(st.max_drawdown);
  const [eWR,vWR]   = rateWR(st.win_rate);
  $('stat-grid').innerHTML = `
    <div class="stat-card"><div class="stat-label">Sharpe Ratio</div><div class="stat-value">${eSH2} ${st.sharpe}</div></div>
    <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value">${eDD} ${st.max_drawdown}%</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">${eWR} ${st.win_rate}%</div></div>
    <div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">${st.total_trades}</div></div>
    <div class="stat-card"><div class="stat-label">Long (Buy)</div><div class="stat-value" style="color:#00e676">${st.long_trades}</div></div>
    <div class="stat-card"><div class="stat-label">Short (Sell)</div><div class="stat-value" style="color:#ff1744">${st.short_trades}</div></div>
  `;
  $('backtest-table').innerHTML = `
    <tr><th>Metric</th><th>Your Value</th><th>Plain English</th></tr>
    <tr><td><strong>Sharpe Ratio</strong></td><td>${st.sharpe}</td><td>Return vs stress ratio. <strong>&gt;1 is good</strong>. ${eSH2} ${vSH2}</td></tr>
    <tr><td><strong>Max Drawdown</strong></td><td>${st.max_drawdown}%</td><td>Worst losing streak before recovery. ${eDD} ${vDD}</td></tr>
    <tr><td><strong>Win Rate</strong></td><td>${st.win_rate}%</td><td>AI was right <strong>${st.win_rate}% of the time</strong> on active trades. ${eWR} ${vWR}</td></tr>
    <tr><td><strong>Total Trades</strong></td><td>${st.total_trades}</td><td>AI only trades when confident (ignores ±0.1% noise)</td></tr>
    <tr><td><strong>Long / Short</strong></td><td>${st.long_trades} / ${st.short_trades}</td><td>Long = bet on rise · Short = bet on fall</td></tr>
  `;

  // ── Cumulative Returns ───────────────────────────────────────────────────
  $('returns-legend').innerHTML = `
    🩶 <strong>Dashed gray</strong> = "I bought ${ticker} and did nothing" (Buy &amp; Hold baseline) &nbsp;·&nbsp;
    🟢 <strong>Green</strong> = "I followed the AI's signals every day" (Our Strategy)<br>
    Value of <strong>1.5</strong> = portfolio grew to <strong>1.5× the starting amount</strong> (+50% total).
    Green <strong>above</strong> gray → AI strategy is winning.
  `;
  Plotly.newPlot('chart-returns', [
    { x:bt.dates, y:bt.cum_asset,    name:'Buy & Hold (passive)', line:{color:'gray',width:2,dash:'dash'} },
    { x:bt.dates, y:bt.cum_strategy, name:`AI Strategy (${chosen})`, line:{color:'#00e676',width:2.5}, fill:'tonexty', fillcolor:'rgba(0,230,118,0.06)' },
  ], {
    ...DARK,
    title:`Cumulative Return: ${ticker} — AI Strategy vs Buy & Hold [${chosen}]`,
    xaxis:{ title:'Date', color:'#a3a8b8', showgrid:false },
    yaxis:{ title:'Portfolio Value (×, starting at 1.0)', color:'#a3a8b8' },
    hovermode:'x unified', height:500,
    margin:{ l:65, r:20, t:60, b:60 },
    shapes:[{ type:'line', x0:bt.dates[0], x1:bt.dates[bt.dates.length-1], y0:1, y1:1, line:{color:'white',width:1,dash:'dot'}, opacity:0.3 }],
    annotations:[{ x:bt.dates[bt.dates.length-1], y:1, xref:'x', yref:'y', text:'Break-even', showarrow:false, font:{color:'rgba(255,255,255,0.35)',size:11}, xanchor:'right' }],
  }, { responsive:true });

  // ── Investment Calculator ────────────────────────────────────────────────
  updateCalc();

  // ── Signals Table ────────────────────────────────────────────────────────
  const n  = Math.min(10, bt.dates.length);
  const s0 = bt.dates.length - n;
  const sigLabel = { '1':'🟢 BUY (Long)', '-1':'🔴 SELL (Short)', '0':'⬜ HOLD' };
  const rows = [];
  for (let i = s0; i < bt.dates.length; i++) {
    const sig = String(bt.signal[i]);
    const sc  = sig === '1' ? 'sig-buy' : sig === '-1' ? 'sig-sell' : 'sig-hold';
    const pc  = bt.pred[i] >= 0 ? 'sig-buy' : 'sig-sell';
    const rc  = bt.strategy_ret[i] >= 0 ? 'sig-buy' : 'sig-sell';
    rows.push(`<tr>
      <td>${bt.dates[i]}</td>
      <td>$${fmt(bt.close[i],2)}</td>
      <td class="${pc}">${bt.pred[i]>=0?'+':''}${bt.pred[i].toFixed(4)}%</td>
      <td class="${sc}">${sigLabel[sig]||sig}</td>
      <td class="${rc}">${bt.strategy_ret[i]>=0?'+':''}${(bt.strategy_ret[i]*100).toFixed(4)}%</td>
      <td>${bt.cum_strategy[i].toFixed(4)}×</td>
    </tr>`);
  }
  $('signals-table').innerHTML = `
    <thead><tr>
      <th>Date</th><th>Closing Price ($)</th><th>Predicted Return</th>
      <th>Action</th><th>Return Earned</th><th>Portfolio Value (×)</th>
    </tr></thead>
    <tbody>${rows.join('')}</tbody>
  `;

  // ── Download ─────────────────────────────────────────────────────────────
  $('download-btn').onclick = () => window.location.href = '/api/download/' + currentJobId;

  $('results').scrollIntoView({ behavior:'smooth', block:'start' });
}

// ── Investment Calculator (live) ──────────────────────────────────────────────
function updateCalc() {
  if (!pipelineResult) return;
  const { final_strategy: fs, final_buyhold: fb, ticker } = pipelineResult;
  const amount  = parseFloat($('calc-amount').value) || 10000;
  const sym     = $('calc-currency').value;
  const outperf = fs > fb;
  const aiR  = amount * fs, bhR = amount * fb;
  const aiP  = aiR - amount, bhP = bhR - amount;
  const aiPct = (fs-1)*100, bhPct = (fb-1)*100;

  $('calc-ai').innerHTML = `
    <div class="metric-label">🤖 AI Strategy result</div>
    <div class="metric-value">${sym}${fmt(aiR,0)}</div>
    <div class="metric-delta ${aiP>=0?'pos':'neg'}">${sym}${aiP>=0?'+':''}${fmt(aiP,0)} (${aiPct>=0?'+':''}${aiPct.toFixed(1)}%)</div>
    <div class="metric-caption">Starting from ${sym}${fmt(amount,0)}</div>
  `;
  $('calc-bh').innerHTML = `
    <div class="metric-label">🧍 Buy &amp; Hold result</div>
    <div class="metric-value">${sym}${fmt(bhR,0)}</div>
    <div class="metric-delta ${bhP>=0?'pos':'neg'}">${sym}${bhP>=0?'+':''}${fmt(bhP,0)} (${bhPct>=0?'+':''}${bhPct.toFixed(1)}%)</div>
    <div class="metric-caption">Starting from ${sym}${fmt(amount,0)}</div>
  `;
  const diff = Math.abs(aiR - bhR);
  const out  = $('calc-outcome');
  if (outperf) {
    out.className   = 'box-success';
    out.innerHTML   = `🏆 AI Strategy earned <strong>${sym}${fmt(diff,0)} more</strong> than just holding ${ticker}!`;
  } else {
    out.className   = 'box-warning';
    out.innerHTML   = `📊 Buy &amp; Hold earned <strong>${sym}${fmt(diff,0)} more</strong> — strategy was more conservative (lower risk too).`;
  }
}
$('calc-amount').addEventListener('input',   updateCalc);
$('calc-currency').addEventListener('change', updateCalc);

// ── Architecture Tab ──────────────────────────────────────────────────────────
let archDone = false;
function renderArchitecture() {
  if (archDone) return;
  archDone = true;

  // Exact same nodes as app.py
  const nodes = [
    { x:1,   y:5,    label:'Yahoo Finance\nAPI',        sub:'Live crypto prices\nBTC, ETH, SOL…',           col:'#00b4d8' },
    { x:3,   y:5,    label:'Data\nIngestion',            sub:'yfinance download\nFlatten MultiIndex',         col:'#0096c7' },
    { x:5,   y:5,    label:'Feature\nEngineering',       sub:'11 Technical\nIndicators computed',             col:'#0077b6' },
    { x:7,   y:5,    label:'Supervised\nDataset',        sub:'X = indicators\ny = next-day return',           col:'#023e8a' },
    { x:9,   y:5,    label:'MinMax\nNormalisation',      sub:'Scale features\nto [0, 1] range',               col:'#03045e' },
    { x:5,   y:2.5,  label:'Train / Val / Test\nSplit',  sub:'80% train · 10% val\n20% test (latest dates)', col:'#1b4332' },
    { x:2,   y:0.5,  label:'XGBoost\nModel',             sub:'n_est=200\nlr=0.05, depth=6',                  col:'#00e676' },
    { x:5,   y:0.5,  label:'RandomForest\nModel',        sub:'n_est=200\ndepth=10, n_jobs=−1',               col:'#00e676' },
    { x:8,   y:0.5,  label:'LSTM\nModel',                sub:'64 units, window=10\nEarlyStopping',            col:'#00e676' },
    { x:5,   y:-2,   label:'Signal\nGenerator',          sub:'pred>+0.1% → BUY\npred<−0.1% → SELL\nelse → HOLD', col:'#ff6d00' },
    { x:5,   y:-4.2, label:'Backtest\nEngine',           sub:'Sharpe · Drawdown\nWin Rate · vs B&H',          col:'#e63946' },
    { x:5,   y:-6.2, label:'Flask\nDashboard',           sub:'Live signal · Charts\nMetrics · Calculator',    col:'#9b5de5' },
  ];

  const shapes = [], annots = [];
  nodes.forEach(({ x, y, label, sub, col }) => {
    const r=parseInt(col.slice(1,3),16), g=parseInt(col.slice(3,5),16), b=parseInt(col.slice(5,7),16);
    shapes.push({ type:'rect', x0:x-1.1,y0:y-0.75,x1:x+1.1,y1:y+0.75,
      fillcolor:`rgba(${r},${g},${b},0.2)`, line:{color:col,width:2}, layer:'below' });
    annots.push({ x,y:y+0.25, text:`<b>${label.replace('\n','<br>')}</b>`, showarrow:false, font:{size:11,color:col}, align:'center' });
    annots.push({ x,y:y-0.3,  text:sub.replace(/\n/g,'<br>'), showarrow:false, font:{size:8.5,color:'#aaaaaa'}, align:'center' });
  });

  [[2.1,5,2.9,5],[4.1,5,4.9,5],[6.1,5,6.9,5],[8.1,5,8.9,5],
   [9,4.25,5,3.25],[3.9,2.5,2.5,1.25],[5,1.75,5,1.25],[6.1,2.5,7.5,1.25],
   [2,-0.25,4,-1.25],[5,-0.25,5,-1.25],[8,-0.25,6,-1.25],
   [5,-2.75,5,-3.45],[5,-4.95,5,-5.45]
  ].forEach(([x0,y0,x1,y1]) =>
    annots.push({ x:x1,y:y1,ax:x0,ay:y0, xref:'x',yref:'y',axref:'x',ayref:'y',
      showarrow:true,arrowhead:2,arrowsize:1.2,arrowwidth:1.5,arrowcolor:'#00e676' })
  );

  Plotly.newPlot('chart-arch', [], {
    ...DARK, shapes, annotations:annots, height:700, showlegend:false,
    xaxis:{visible:false,range:[-0.5,11]}, yaxis:{visible:false,range:[-7.2,6.5]},
    margin:{l:10,r:10,t:10,b:10},
  }, { responsive:true });

  // Process flow — exact same steps as app.py
  const steps = [
    { n:'1', col:'#00b4d8', title:'Fetch Raw Data',
      desc:'Downloads OHLCV data (Open, High, Low, Close, Volume) from Yahoo Finance.\nFlattens MultiIndex columns. Filters to required fields only.' },
    { n:'2', col:'#0096c7', title:'Compute 11 Indicators',
      desc:'Calculates SMA, WMA, Momentum, Stochastic %K/%D, RSI, MACD, Williams %R, A/D, CCI.\nEach indicator captures a different market pattern (trend, momentum, volume, oscillation).' },
    { n:'3', col:'#0077b6', title:'Build Supervised Dataset',
      desc:'Target variable = next-day % return: (Close_t+1 − Close_t) / Close_t\nPredicting returns (not price) makes the model scale-free and stationary.' },
    { n:'4', col:'#023e8a', title:'Normalise Features',
      desc:'MinMaxScaler scales all 11 features to [0, 1].\nPrevents large-valued indicators (like A/D) from dominating small ones (like RSI).' },
    { n:'5', col:'#1b4332', title:'Time-Aware Split',
      desc:'Train: first 72% · Validation: next 8% · Test: last 20% (most recent dates).\nNever shuffle — preserves time order. Test set simulates real future prediction.' },
    { n:'6', col:'#00e676', title:'Train 3 Models',
      desc:'XGBoost & RandomForest: fit on (X_train, y_train), predict on X_test.\nLSTM: builds windowed sequences of shape [samples, window, features] then trains with EarlyStopping.' },
    { n:'7', col:'#ff6d00', title:'Generate Signals',
      desc:'For every test-set day: if predicted return > +0.1% → BUY (+1)\nIf < −0.1% → SELL (−1) · If within ±0.1% band → HOLD (0).\nThe 0.1% threshold filters noise smaller than transaction cost.' },
    { n:'8', col:'#e63946', title:'Backtest',
      desc:'strategy_return = signal × actual_next_day_return − (0.1% if trade_entry)\nCumulative product of (1 + daily_return) gives portfolio growth curve.\nSharpe, Drawdown, Win Rate computed from this curve.' },
    { n:'9', col:'#9b5de5', title:'Display Results',
      desc:'Flask shows: live signal · model metrics · feature importance · radar chart\nCumulative returns graph · investment calculator · signals table · CSV download.' },
  ];

  const pf = $('process-flow');
  pf.innerHTML = '';
  steps.forEach(({ n, col, title, desc }) => {
    const [r,g,b] = [parseInt(col.slice(1,3),16), parseInt(col.slice(3,5),16), parseInt(col.slice(5,7),16)];
    const d = document.createElement('div');
    d.className = 'flow-step';
    d.style.cssText = `border-left-color:${col};background:rgba(${r},${g},${b},0.05)`;
    d.innerHTML = `
      <div class="flow-num" style="background:${col}">${n}</div>
      <div>
        <div class="flow-title" style="color:${col}">${title}</div>
        <div class="flow-desc">${desc}</div>
      </div>`;
    pf.appendChild(d);
  });
}
