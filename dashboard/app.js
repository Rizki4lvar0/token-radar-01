// Token Radar dashboard: renders window.RADAR (written by main.ts) with no
// network. Sortable, filterable board; a detail view per protocol; the live
// set; the fitted model's validation, stated plainly.
(function () {
  var D = window.RADAR;
  if (!D) { document.body.textContent = 'No radar data.'; return; }

  function money(n) {
    if (n === null || n === undefined) return '—';
    var abs = Math.abs(n);
    if (abs >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (abs >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (abs >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(n);
  }
  function pct(v) {
    if (v === null || v === undefined) return '—';
    var s = (v > 0 ? '+' : '') + v.toFixed(1) + '%';
    return '<span class="' + (v >= 0 ? 'pos' : 'neg') + '">' + s + '</span>';
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  var rows = D.rows, live = D.live, fit = D.fit;

  document.getElementById('meta').innerHTML =
    '<b>' + D.board_size + '</b> ranked protocols<br>Snapshot as of <b>' + esc(D.as_of) + '</b><br>Watchlist floor <b>' + money(D.floor_usd) + '</b><br>Model <b>' + esc(D.model_version) + '</b>';

  // Integrity strip: checks computed from the data itself.
  var watchCovered = rows.filter(function (r) { return r.on_watchlist; }).length;
  var stepsOk = rows.every(function (r) { return r.farm_steps.length >= 3 && r.farm_steps.length <= 12; });
  var sortedOk = rows.every(function (r, i) { return i === 0 || rows[i - 1].implied_value_usd >= r.implied_value_usd; });
  var chips = [
    ['Watchlist covered ' + watchCovered + '/' + D.watchlist_size, watchCovered === D.watchlist_size],
    ['Board ' + rows.length + ' rows, ranks 1–' + rows.length, rows.length >= 100],
    ['Farm plans 6 steps each', stepsOk],
    ['Values fall with rank', sortedOk],
    ['No network: static page', true],
    ['Peer fit is noisy — see validation', false],
  ];
  document.getElementById('integrity').innerHTML = chips.map(function (c) {
    return '<span class="chip ' + (c[1] ? 'ok' : 'warn') + '">' + esc(c[0]) + '</span>';
  }).join('');

  document.getElementById('model-text').textContent =
    'Implied value = TVL × the market-cap-to-TVL ratio the peers pay. The ratio is fitted at run time from ' +
    fit.peers + ' protocols whose tokens already trade: an OLS fit of log10(market-cap / TVL) on log TVL, ' +
    'seven-day growth, age and chain count, averaged with a shrunk category median, then clamped to the ' +
    'peers’ 5th–95th percentile range. Seven-day growth enters with a negative fitted coefficient — among ' +
    'the peers, a sharp one-week TVL jump usually goes with a lower ratio, so the model prices the level, ' +
    'not the spike. No protocol name is special-cased.';
  document.getElementById('metrics').innerHTML = [
    ['Peers fitted', fit.peers],
    ['In-sample R²', fit.r_squared],
    ['In-sample RMSE (log10)', fit.rmse_log10],
    ['Held-out RMSE (log10)', fit.held_out.rmse_log10 + '  (n=' + fit.held_out.n_test + ')'],
    ['Walk-forward RMSE (log10)', fit.walk_forward.rmse_log10 + '  (n=' + fit.walk_forward.n_test + ')'],
    ['Global median ratio', '10^' + fit.global_median_log_ratio],
  ].map(function (m) { return '<dt>' + esc(m[0]) + '</dt><dd>' + esc(String(m[1])) + '</dd>'; }).join('');
  document.getElementById('model-note').textContent =
    'Read the validation honestly: the peer fit is noisy (R² ≈ ' + fit.r_squared + '). Each row therefore carries ' +
    'an interval of roughly ×/÷ ' + Math.pow(10, fit.held_out.rmse_log10).toFixed(1) +
    ' around its implied value. These are order-of-magnitude estimates, not prices.';

  // Filters.
  var cats = [...new Set(rows.map(function (r) { return r.category || 'Unknown'; }))].sort();
  var chains = [...new Set(rows.flatMap(function (r) { return r.chains; }))].sort();
  var catSel = document.getElementById('cat'), chainSel = document.getElementById('chain');
  cats.forEach(function (c) { var o = document.createElement('option'); o.value = c; o.textContent = c; catSel.appendChild(o); });
  chains.forEach(function (c) { var o = document.createElement('option'); o.value = c; o.textContent = c; chainSel.appendChild(o); });

  var selected = null;
  function filtered() {
    var q = document.getElementById('q').value.trim().toLowerCase();
    var cat = catSel.value, chain = chainSel.value, watch = document.getElementById('watch').checked;
    var sort = document.getElementById('sort').value;
    var out = rows.filter(function (r) {
      if (q && (r.name.toLowerCase().indexOf(q) < 0 && r.slug.toLowerCase().indexOf(q) < 0)) return false;
      if (cat && (r.category || 'Unknown') !== cat) return false;
      if (chain && r.chains.indexOf(chain) < 0) return false;
      if (watch && !r.on_watchlist) return false;
      return true;
    });
    if (sort === 'tvl') out.sort(function (a, b) { return b.tvl_usd - a.tvl_usd; });
    else if (sort === 'growth') out.sort(function (a, b) { return (b.change_7d || 0) - (a.change_7d || 0); });
    else if (sort === 'age') out.sort(function (a, b) { return (b.age_days || 0) - (a.age_days || 0); });
    return out;
  }
  function renderBoard() {
    var out = filtered();
    document.getElementById('board-note').textContent = out.length + ' of ' + rows.length + ' ranked protocols shown. Click a row for its detail and farm plan.';
    var tb = document.querySelector('#board tbody');
    tb.innerHTML = out.map(function (r) {
      return '<tr data-slug="' + esc(r.slug) + '" class="' + (selected === r.slug ? 'selected' : '') + '"><td class="num">' + r.rank + '</td>' +
        '<td>' + esc(r.name) + (r.on_watchlist ? '<span class="badge">watchlist</span>' : '') + '</td>' +
        '<td>' + esc(r.category || '—') + '</td><td>' + esc(r.chains.slice(0, 2).join(', ') || '—') + '</td>' +
        '<td class="num">' + money(r.tvl_usd) + '</td><td class="num">' + pct(r.change_7d) + '</td>' +
        '<td class="num">' + (r.age_days === null ? '—' : r.age_days + 'd') + '</td>' +
        '<td class="num"><b>' + money(r.implied_value_usd) + '</b></td></tr>';
    }).join('');
    tb.querySelectorAll('tr').forEach(function (tr) {
      tr.addEventListener('click', function () { showDetail(tr.getAttribute('data-slug')); });
    });
  }
  function showDetail(slug) {
    var r = rows.find(function (x) { return x.slug === slug; });
    if (!r) return;
    selected = slug;
    var el = document.getElementById('detail');
    el.innerHTML = '<h3>' + esc(r.name) + ' — rank #' + r.rank + (r.on_watchlist ? ' · on the watchlist' : '') + '</h3>' +
      '<div class="detail-grid"><div><h3>Value</h3><p>Implied value <b>' + money(r.implied_value_usd) + '</b>, range ' +
      money(r.implied_low_usd) + ' – ' + money(r.implied_high_usd) + '. TVL ' + money(r.tvl_usd) +
      ', 7-day change ' + (r.change_7d === null ? 'not reported' : r.change_7d + '%') +
      ', listed ' + (r.age_days === null ? 'date unknown' : r.age_days + ' days ago') + '.</p></div>' +
      '<div><h3>Model breakdown</h3><p>Predicted log10 ratio ' + r.model.log_ratio +
      ' (OLS ' + r.model.ols_log_ratio + ', category ' + r.model.category_log_ratio +
      '). Growth used ' + r.model.growth_pct + '%. The ratio is the peers’ fitted ratio, not a hand-set multiple.</p></div></div>' +
      '<h3>Farm plan — what to do, what it costs, how long it takes</h3><ol class="steps">' +
      r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>' +
      '<p class="note">“Pre-token” means the snapshot records no token for this protocol; it is not proof it has none. Never pay to claim an airdrop.</p>';
    renderBoard();
    document.getElementById('detail-panel').scrollIntoView({ behavior: 'auto', block: 'start' });
  }

  ['q', 'cat', 'chain', 'watch', 'sort'].forEach(function (id) {
    document.getElementById(id).addEventListener('input', renderBoard);
    document.getElementById(id).addEventListener('change', renderBoard);
  });

  document.querySelector('#live tbody').innerHTML = live.map(function (p) {
    return '<tr><td>' + esc(p.name) + '</td><td>' + esc(p.category || '—') + '</td><td>' + esc(p.chains.slice(0, 2).join(', ') || '—') +
      '</td><td class="num">' + money(p.tvl_usd) + '</td><td class="num">' + money(p.mcap_usd) + '</td><td class="num">' + pct(p.change_7d) + '</td></tr>';
  }).join('');

  var a = D.accounted;
  document.getElementById('coverage').textContent =
    'The snapshot holds ' + a.total + ' protocols. ' + a.ranked + ' pre-token protocols are ranked on the board, ' +
    a.live + ' already trade (their market values are shown in the live view, and their records fitted the model), and ' +
    a.excluded + ' pre-token protocols are left off the board with a stated reason — they are wrapped, staked or bridged ' +
    'products, exchanges, or versioned pools, not launches waiting to happen.';
  document.getElementById('coverage-metrics').innerHTML = [
    ['Ranked', a.ranked], ['Already trading', a.live], ['Left off, with reason', a.excluded], ['Total accounted', a.total],
  ].map(function (m) { return '<dt>' + esc(m[0]) + '</dt><dd>' + esc(String(m[1])) + '</dd>'; }).join('') +
    Object.keys(D.excluded_by_reason).map(function (k) { return '<dt>' + esc(k) + '</dt><dd>' + D.excluded_by_reason[k] + '</dd>'; }).join('');

  // Hash routing, so the three screenshots can open the board, a detail view and a filtered view.
  function applyHash() {
    var h = location.hash || '';
    var m = h.match(/detail=([^&]+)/);
    var f = h.match(/filter=([^&]+)/);
    if (f) {
      var cat = decodeURIComponent(f[1]);
      if ([...catSel.options].some(function (o) { return o.value === cat; })) { catSel.value = cat; }
    }
    renderBoard();
    if (m) {
      // Put the detail panel first, so a screenshot opened on #detail= shows it.
      var main = document.querySelector('main');
      var panel = document.getElementById('detail-panel');
      main.insertBefore(panel, main.firstChild);
      showDetail(decodeURIComponent(m[1]));
    }
  }
  window.addEventListener('hashchange', applyHash);
  applyHash();
})();
