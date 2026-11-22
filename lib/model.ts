// Token Radar — value model, fitted from the snapshot's own peers.
//
// Idea: a pre-token protocol has no market price, but the snapshot holds 128
// protocols whose tokens already trade. For each of those peers we know the
// market's own answer — market-cap divided by TVL — so the model learns, from
// the peers alone, how that ratio moves with size, seven-day growth, age and
// chain count, and how it differs by category. It then prices every
// pre-token protocol with the same rule. No protocol name is special-cased.
//
// The fit is done at run time, inside this file, from the snapshot given:
//   * an ordinary-least-squares fit of log10(mcap / tvl) on
//     [1, log10(tvl), growth/100, log1p(age in days), chain count]
//   * a category median of the same log-ratio, shrunk toward the global
//     median (5 pseudo-observations), so thin categories do not swing a price
//   * the prediction is the mean of the two, clamped to the peers'
//     5th–95th percentile range, so no row can claim a ratio the market
//     never paid.
//
// Validation is also done at run time and reported in the dashboard:
//   * held-out: every 5th peer by slug order is held out, the model is fit
//     on the rest, and the error on the held-out peers is reported.
//   * walk-forward: peers listed before the median listing date train the
//     model; peers listed after it are the test set.
//   * ablation: the model is refit with one term removed at a time and scored
//     on the same held-out peers, so each term has to earn its keep.
// Both are honest about the fit: the peer ratio is noisy (R^2 near 0.2),
// so implied values are order-of-magnitude estimates and each row carries
// its interval. The dashboard states the numbers instead of hiding them.
//
// Each row also carries a farm verdict (FARM / THIN / SKIP) and a break-even,
// computed from numbers already on the row: the plan's out-of-pocket spend
// against one farmer's assumed 1-in-10,000 share of the implied value.
// The verdict never changes the ranking or the values.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface FitMetrics {
  peers: number;
  r_squared: number;
  rmse_log10: number;
  global_median_log_ratio: number;
  log_ratio_lo: number;
  log_ratio_hi: number;
  held_out: { n_train: number; n_test: number; rmse_log10: number; mae_log10: number };
  walk_forward: { n_train: number; n_test: number; rmse_log10: number; mae_log10: number; split_date: string | null };
  coefficients: number[];
  /** Held-out RMSE of the full model, then of the model refit with one term removed. */
  ablation: { term: string; held_out_rmse_log10: number }[];
}

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  age_days: number | null;
  implied_value_usd: number;
  implied_low_usd: number;
  implied_high_usd: number;
  token_status: 'pre-token';
  on_watchlist: boolean;
  model: {
    log_ratio: number;
    ols_log_ratio: number;
    category_log_ratio: number;
    growth_pct: number;
    maturity_days: number;
  };
  farm_steps: string[];
  /** Farm verdict, from numbers already on the row: the plan's out-of-pocket
   * spend, the airdrop that would break even, and one farmer's assumed share
   * of the implied value at the low end of its range. */
  farm: {
    verdict: 'FARM' | 'THIN' | 'SKIP';
    spend_usd: number;
    break_even_usd: number;
    expected_low_usd: number;
  };
}

export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  live_size: number;
  accounted: { ranked: number; live: number; excluded: number; total: number };
  excluded_by_reason: Record<string, number>;
  fit: FitMetrics;
  rows: BoardRow[];
  live: LiveRow[];
}

export const MODEL_VERSION = 'radar-peerfit-2';

const DAY_MS = 86_400_000;
const UNKNOWN_AGE_DAYS = 365;

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

function clampGrowth(change7d: number | null): number {
  return Math.min(120, Math.max(-60, finite(change7d, 0)));
}

/** A protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const EXCLUDED_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/** Whether a pre-token protocol can be ranked: not an exchange, not a versioned pool, not a wrapped/staked/bridged product. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && !EXCLUDED_CATEGORIES.has(p.category ?? '') && !VERSIONED.test(p.name);
}

/** Why a pre-token protocol is left off the board, for the coverage strip. */
export function exclusionReason(p: SnapshotProtocol): string | null {
  if (!isPreToken(p)) return null;
  if ((p.category ?? '') === 'CEX') return 'exchange';
  if (EXCLUDED_CATEGORIES.has(p.category ?? '')) return 'wrapped / staked / bridged product';
  if (VERSIONED.test(p.name)) return 'versioned pool (V2 / 2.0 name)';
  return null;
}

/** Features for the peer regression. A dropped index is zeroed, which for OLS
 * is the same as refitting without that term: its coefficient is pinned to
 * zero by the ridge and the other coefficients refit around its absence. */
function features(p: SnapshotProtocol, asOf: string, dropIdx: number | null = null): number[] {
  const age = ageDays(asOf, p.listed_at) ?? UNKNOWN_AGE_DAYS;
  const f = [1, Math.log10(Math.max(1, p.tvl_usd)), clampGrowth(p.change_7d) / 100, Math.log1p(Math.max(0, age)), Math.min(4, p.chains.length)];
  if (dropIdx !== null) f[dropIdx] = 0;
  return f;
}

/** Solve (X'X + ridge*I) b = X'y by Gaussian elimination. X rows are feature vectors. */
function ols(rows: number[][], y: number[]): number[] {
  const n = rows[0]!.length;
  const ata: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const aty: number[] = new Array<number>(n).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]!;
    for (let a = 0; a < n; a++) {
      aty[a]! += r[a]! * y[i]!;
      for (let b = 0; b < n; b++) ata[a]![b]! += r[a]! * r[b]!;
    }
  }
  for (let a = 0; a < n; a++) ata[a]![a]! += 1e-6;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(ata[r]![col]!) > Math.abs(ata[pivot]![col]!)) pivot = r;
    [ata[col], ata[pivot]] = [ata[pivot]!, ata[col]!];
    [aty[col], aty[pivot]] = [aty[pivot]!, aty[col]!];
    const div = ata[col]![col]!;
    for (let r = col + 1; r < n; r++) {
      const f = ata[r]![col]! / div;
      for (let c = col; c < n; c++) ata[r]![c]! -= f * ata[col]![c]!;
      aty[r]! -= f * aty[col]!;
    }
  }
  const beta = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = aty[i]!;
    for (let j = i + 1; j < n; j++) s -= ata[i]![j]! * beta[j]!;
    beta[i] = s / ata[i]![i]!;
  }
  return beta;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

function percentile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b);
  if (s.length === 0) return 0;
  const i = Math.min(s.length - 1, Math.max(0, Math.floor(q * s.length)));
  return s[i]!;
}

function dot(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i]!, 0);
}

interface PeerSet {
  peers: SnapshotProtocol[];
  logRatios: number[];
}

function peerSet(protocols: SnapshotProtocol[]): PeerSet {
  const peers = protocols.filter((p) => !isPreToken(p) && p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const logRatios = peers.map((p) => Math.log10(p.mcap_usd! / p.tvl_usd));
  return { peers, logRatios };
}

function categoryMedians(peers: SnapshotProtocol[], logRatios: number[]): Map<string, number[]> {
  const byCat = new Map<string, number[]>();
  peers.forEach((p, i) => {
    const key = p.category ?? 'Unknown';
    if (!byCat.has(key)) byCat.set(key, []);
    byCat.get(key)!.push(logRatios[i]!);
  });
  return byCat;
}

interface FittedModel {
  beta: number[];
  globalMedian: number;
  catMedian: Map<string, number>;
  lo: number;
  hi: number;
  rmse: number;
  dropIdx: number | null;
  useCategory: boolean;
}

function fitModel(peers: SnapshotProtocol[], logRatios: number[], asOf: string, dropIdx: number | null = null, useCategory = true): FittedModel {
  const beta = ols(peers.map((p) => features(p, asOf, dropIdx)), logRatios);
  const globalMedian = median(logRatios);
  const byCat = categoryMedians(peers, logRatios);
  const catMedian = new Map<string, number>();
  for (const [cat, vals] of byCat) {
    catMedian.set(cat, (vals.length * median(vals) + 5 * globalMedian) / (vals.length + 5));
  }
  const preds = peers.map((p, i) => predictOne(p, asOf, beta, globalMedian, catMedian, dropIdx, useCategory));
  const rmse = Math.sqrt(preds.reduce((s, pr, i) => s + (pr - logRatios[i]!) ** 2, 0) / Math.max(1, preds.length));
  return { beta, globalMedian, catMedian, lo: percentile(logRatios, 0.05), hi: percentile(logRatios, 0.95), rmse, dropIdx, useCategory };
}

function predictOne(p: SnapshotProtocol, asOf: string, beta: number[], globalMedian: number, catMedian: Map<string, number>, dropIdx: number | null = null, useCategory = true): number {
  const olsPred = dot(beta, features(p, asOf, dropIdx));
  if (!useCategory) return olsPred;
  const catPred = catMedian.get(p.category ?? 'Unknown') ?? globalMedian;
  return (olsPred + catPred) / 2;
}

function predictClamped(p: SnapshotProtocol, asOf: string, model: FittedModel): { logRatio: number; ols: number; cat: number } {
  const olsPred = dot(model.beta, features(p, asOf, model.dropIdx));
  const catPred = model.catMedian.get(p.category ?? 'Unknown') ?? model.globalMedian;
  const raw = model.useCategory ? (olsPred + catPred) / 2 : olsPred;
  return { logRatio: Math.min(model.hi, Math.max(model.lo, raw)), ols: olsPred, cat: catPred };
}

function errorStats(actual: number[], predicted: number[]): { rmse: number; mae: number } {
  const n = actual.length;
  if (n === 0) return { rmse: 0, mae: 0 };
  const sq = actual.reduce((s, a, i) => s + (a - predicted[i]!) ** 2, 0) / n;
  const ae = actual.reduce((s, a, i) => s + Math.abs(a - predicted[i]!), 0) / n;
  return { rmse: Math.sqrt(sq), mae: ae };
}

/** Fit the model and run both validations, all from the snapshot's peers. */
export function fitAndValidate(snapshot: Snapshot): { model: FittedModel; metrics: FitMetrics } {
  const { peers, logRatios } = peerSet(snapshot.protocols);
  const model = fitModel(peers, logRatios, snapshot.as_of);
  const preds = peers.map((p) => predictClamped(p, snapshot.as_of, model).logRatio);
  const mean = logRatios.reduce((s, v) => s + v, 0) / Math.max(1, logRatios.length);
  const ssTot = logRatios.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ssRes = logRatios.reduce((s, v, i) => s + (v - preds[i]!) ** 2, 0);
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  // Held-out: every 5th peer by slug order.
  const sorted = peers.map((p, i) => ({ p, y: logRatios[i]! })).sort((a, b) => (a.p.slug < b.p.slug ? -1 : 1));
  const testIdx = new Set(sorted.map((_, i) => i).filter((i) => i % 5 === 0));
  const trainPeers = sorted.filter((_, i) => !testIdx.has(i)).map((s) => s.p);
  const trainY = sorted.filter((_, i) => !testIdx.has(i)).map((s) => s.y);
  const testPeers = sorted.filter((_, i) => testIdx.has(i)).map((s) => s.p);
  const testY = sorted.filter((_, i) => testIdx.has(i)).map((s) => s.y);
  const heldModel = fitModel(trainPeers, trainY, snapshot.as_of);
  const heldPred = testPeers.map((p) => predictClamped(p, snapshot.as_of, heldModel).logRatio);
  const held = errorStats(testY, heldPred);

  // Ablation: refit with one term removed, scored on the same held-out peers.
  // A term earns its keep when removing it makes the held-out error worse.
  const TERM_NAMES = ['log TVL', '7-day growth', 'age', 'chain count'];
  const ablation: { term: string; held_out_rmse_log10: number }[] = [
    { term: 'full model', held_out_rmse_log10: Math.round(held.rmse * 1000) / 1000 },
  ];
  for (let d = 1; d <= 4; d++) {
    const m = fitModel(trainPeers, trainY, snapshot.as_of, d, true);
    const pr = testPeers.map((p) => predictClamped(p, snapshot.as_of, m).logRatio);
    const e = errorStats(testY, pr);
    ablation.push({ term: `without ${TERM_NAMES[d - 1]}`, held_out_rmse_log10: Math.round(e.rmse * 1000) / 1000 });
  }
  const noCatModel = fitModel(trainPeers, trainY, snapshot.as_of, null, false);
  const noCatPred = testPeers.map((p) => predictClamped(p, snapshot.as_of, noCatModel).logRatio);
  const noCat = errorStats(testY, noCatPred);
  ablation.push({ term: 'without category median', held_out_rmse_log10: Math.round(noCat.rmse * 1000) / 1000 });

  // Walk-forward: train on peers listed before the median listing date, test after.
  const dated = peers
    .map((p, i) => ({ p, y: logRatios[i]!, d: p.listed_at }))
    .filter((s) => s.d !== null)
    .sort((a, b) => (a.d! < b.d! ? -1 : 1));
  const splitAt = dated.length > 0 ? dated[Math.floor(dated.length / 2)]!.d : null;
  let walk = { n_train: 0, n_test: 0, rmse_log10: 0, mae_log10: 0, split_date: splitAt };
  if (splitAt !== null && dated.length >= 10) {
    const wTrain = dated.filter((s) => s.d! < splitAt);
    const wTest = dated.filter((s) => s.d! >= splitAt);
    const wModel = fitModel(wTrain.map((s) => s.p), wTrain.map((s) => s.y), snapshot.as_of);
    const wPred = wTest.map((s) => predictClamped(s.p, snapshot.as_of, wModel).logRatio);
    const e = errorStats(wTest.map((s) => s.y), wPred);
    walk = { n_train: wTrain.length, n_test: wTest.length, rmse_log10: e.rmse, mae_log10: e.mae, split_date: splitAt };
  }

  const metrics: FitMetrics = {
    peers: peers.length,
    r_squared: Math.round(r2 * 1000) / 1000,
    rmse_log10: Math.round(model.rmse * 1000) / 1000,
    global_median_log_ratio: Math.round(model.globalMedian * 1000) / 1000,
    log_ratio_lo: Math.round(model.lo * 1000) / 1000,
    log_ratio_hi: Math.round(model.hi * 1000) / 1000,
    held_out: { n_train: trainPeers.length, n_test: testPeers.length, rmse_log10: Math.round(held.rmse * 1000) / 1000, mae_log10: Math.round(held.mae * 1000) / 1000 },
    walk_forward: { ...walk, rmse_log10: Math.round(walk.rmse_log10 * 1000) / 1000, mae_log10: Math.round(walk.mae_log10 * 1000) / 1000 },
    coefficients: model.beta.map((b) => Math.round(b * 10000) / 10000),
    ablation,
  };
  return { model, metrics };
}

// ---------------------------------------------------------------- farm plans

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp')) return `Lend $${capital} of USDC on ${name} on ${chain} and leave the deposit working.`;
  if (c.includes('dex') || c.includes('leveraged farm')) return `Add $${capital} of liquidity to the main ${name} pool on ${chain} and leave it in place.`;
  if (c.includes('yield') || c.includes('vault')) return `Deposit $${capital} of USDC into the main ${name} vault on ${chain} and hold the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Post $${capital} of USDC as margin on ${name} and open one small hedged position.`;
  if (c.includes('restak') || c.includes('staking')) return `Stake $${capital} of the chain asset through ${name} and keep the staking receipt.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of stablecoins through ${name} on ${chain} and hold the position.`;
  if (c.includes('rwa') || c.includes('onchain capital') || c.includes('allocator')) return `Buy $${capital} of the yield-bearing asset ${name} issues on ${chain} and hold it.`;
  if (c.includes('basis')) return `Deposit $${capital} into the ${name} basis strategy on ${chain} and leave it running.`;
  if (c.includes('prediction')) return `Fund ${name} with $${capital} and take one small market position.`;
  if (c.includes('bridge')) return `Move $${capital} of USDC to ${chain} through ${name} and use the funds inside its app.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and leave the position working.`;
}

/** Out-of-pocket spend of one farm plan: first gas, weekly gas over the hold,
 * and the claim check. The deposit capital is not spend — it comes back with
 * the position. */
export function farmSpendUsd(p: SnapshotProtocol, age: number | null): number {
  const capital = capitalFor(p);
  const gasFirst = capital >= 150 ? 8 : 4;
  const hold = holdDaysFor(age);
  return gasFirst + Math.ceil(hold / 7) * 2 + 2;
}

/**
 * Farm verdict, from numbers already on the row. A farmer's share of a launch
 * is a tiny fraction of the protocol's implied value; we assume 1 in 10,000
 * of it reaches one active farmer, and the dashboard says so. The verdict
 * compares that expected share, at the low end of the value range, with the
 * plan's out-of-pocket spend: FARM when even the low end covers the spend ten
 * times over, THIN when it covers the spend at all, SKIP otherwise.
 */
export function farmVerdict(impliedLowUsd: number, impliedUsd: number, spendUsd: number): { verdict: 'FARM' | 'THIN' | 'SKIP'; expected_low_usd: number } {
  const SHARE = 0.0001;
  const expectedLow = impliedLowUsd * SHARE;
  const expected = impliedUsd * SHARE;
  const rounded = Math.round(expectedLow * 100) / 100;
  if (expectedLow >= 10 * spendUsd) return { verdict: 'FARM', expected_low_usd: rounded };
  if (expectedLow >= spendUsd || expected >= 10 * spendUsd) return { verdict: 'THIN', expected_low_usd: rounded };
  return { verdict: 'SKIP', expected_low_usd: rounded };
}

/** A second, different qualifying action for the same protocol, done once
 * mid-hold so the wallet shows two kinds of activity, not one repeated. */
function secondActionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  const small = Math.max(10, Math.round(capital / 5));
  if (c.includes('lend') || c.includes('cdp')) return `Borrow $${small} against the deposit once, then repay it the same week, so the wallet shows a borrow as well as a lend.`;
  if (c.includes('dex') || c.includes('leveraged farm')) return `Swap $${small} through the main ${name} pool once, so the wallet shows a swap as well as liquidity.`;
  if (c.includes('yield') || c.includes('vault')) return `Add a second, smaller deposit of $${small} from a different asset into the ${name} vault, so the wallet shows two deposits.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Close the hedged position once and open one fresh small trade, so the wallet shows more than a single entry.`;
  if (c.includes('restak') || c.includes('staking')) return `Restake the staking receipt once through the ${name} restaking route, so the wallet shows a second staking action.`;
  if (c.includes('stable')) return `Hold the minted stablecoins in the ${name} savings route for a week, so the wallet shows a second stablecoin action.`;
  if (c.includes('rwa') || c.includes('onchain capital') || c.includes('allocator')) return `Reinvest one distribution back into the same ${name} asset, so the wallet shows a reinvestment as well as a buy.`;
  if (c.includes('basis')) return `Top up the ${name} basis position once with a small second deposit of $${small}, so the wallet shows two deposits.`;
  if (c.includes('prediction')) return `Settle one market and enter a second small one on ${name}, so the wallet shows two settled markets.`;
  if (c.includes('bridge')) return `Bridge $${small} back to the origin chain once through ${name}, so the wallet shows a round trip.`;
  return `Make one second, different on-chain interaction — a vote, a second deposit or a small swap — so the wallet shows two kinds of activity.`;
}

/** Eight costed, timed steps, derived from the protocol's own record. */
export function farmSteps(p: SnapshotProtocol, age: number | null, verdict: 'FARM' | 'THIN' | 'SKIP'): string[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = capital >= 150 ? 8 : 4;
  const spend = farmSpendUsd(p, age);
  const verdictNote =
    verdict === 'FARM'
      ? 'even the low end of the value range would pay the spend back more than ten times over'
      : verdict === 'THIN'
        ? 'the low end of the range would just about cover the spend'
        : 'even the point estimate would not cover the spend';
  const steps = [
    `Open a fresh wallet that holds only the funds you plan to farm, and store its recovery phrase offline. Cost: under $3 of gas for the first transaction. Time: 5 minutes.`,
    `${actionFor(p, capital, chain)} Cost: $${capital} of capital and about $${gas} of gas. Time: 15 minutes to set up.`,
    `Do one qualifying action each week — a deposit, a swap, a borrow or a vote — so the wallet stays active. Cost: about $2 of gas a week. Time: 10 minutes a week.`,
    `Keep the position open for at least ${hold} days so the activity spans more than one points snapshot. Cost: the $${capital} stays in the position. Time: ${hold} days.`,
    `Before withdrawing, open the protocol's own rewards page from its official site and claim only through the link it publishes. Cost: nothing to check, and never pay to claim. Time: 10 minutes.`,
    `Log the wallet, the dates and every transaction in a farm notebook so you can repeat the position or unwind it cleanly. Cost: nothing but the notebook. Time: 5 minutes.`,
    `${secondActionFor(p, capital, chain)} Cost: about $2 of gas. Time: 10 minutes, once.`,
    `Break-even: an airdrop of at least $${spend} covers this plan's out-of-pocket spend of $${spend} (gas only; the $${capital} deposit comes back with the position). Verdict: ${verdict} — ${verdictNote}. Time: 5 minutes to check.`,
  ];
  return steps.map((s) => (s.length > 600 ? `${s.slice(0, 596)}...` : s));
}

// ---------------------------------------------------------------- board

export function buildBoard(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const { model, metrics } = fitAndValidate(snapshot);
  const watch = new Set(snapshot.watchlist);
  const interval = Math.pow(10, metrics.held_out.rmse_log10);

  const ranked = snapshot.protocols.filter(isRankable);
  const scored = ranked.map((p) => {
    const age = ageDays(snapshot.as_of, p.listed_at);
    const pred = predictClamped(p, snapshot.as_of, model);
    const implied = Math.max(1, Math.min(100_000_000_000_000, Math.round(p.tvl_usd * Math.pow(10, pred.logRatio))));
    return {
      p,
      age,
      pred,
      implied,
      low: Math.max(1, Math.round(implied / interval)),
      high: Math.min(100_000_000_000_000, Math.round(implied * interval)),
    };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));

  const board: BoardRow[] = scored.map((s, i) => {
    const spend = farmSpendUsd(s.p, s.age);
    const v = farmVerdict(s.low, s.implied, spend);
    return {
      rank: i + 1,
      slug: s.p.slug,
      name: s.p.name,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      age_days: s.age,
      implied_value_usd: s.implied,
      implied_low_usd: s.low,
      implied_high_usd: s.high,
      token_status: 'pre-token',
      on_watchlist: watch.has(s.p.slug),
      model: {
        log_ratio: Math.round(s.pred.logRatio * 10000) / 10000,
        ols_log_ratio: Math.round(s.pred.ols * 10000) / 10000,
        category_log_ratio: Math.round(s.pred.cat * 10000) / 10000,
        growth_pct: clampGrowth(s.p.change_7d),
        maturity_days: s.age ?? UNKNOWN_AGE_DAYS,
      },
      farm_steps: farmSteps(s.p, s.age, v.verdict),
      farm: { verdict: v.verdict, spend_usd: spend, break_even_usd: spend, expected_low_usd: v.expected_low_usd },
    };
  });

  const live: LiveRow[] = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, 24)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));

  const excludedByReason: Record<string, number> = {};
  let excluded = 0;
  for (const p of snapshot.protocols) {
    if (isPreToken(p) && !isRankable(p)) {
      excluded++;
      const r = exclusionReason(p) ?? 'other';
      excludedByReason[r] = (excludedByReason[r] ?? 0) + 1;
    }
  }
  const liveCount = snapshot.protocols.filter((p) => !isPreToken(p)).length;

  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    live_size: live.length,
    accounted: { ranked: board.length, live: liveCount, excluded: excluded, total: snapshot.protocols.length },
    excluded_by_reason: excludedByReason,
    fit: metrics,
    rows: board.map((r) => ({
      rank: r.rank,
      slug: r.slug,
      name: r.name,
      category: r.category,
      chains: r.chains,
      tvl_usd: r.tvl_usd,
      change_7d: r.change_7d,
      age_days: r.age_days,
      implied_value_usd: r.implied_value_usd,
      implied_low_usd: r.implied_low_usd,
      implied_high_usd: r.implied_high_usd,
      token_status: 'pre-token',
      on_watchlist: r.on_watchlist,
      model: r.model,
      farm_steps: r.farm_steps,
      farm: r.farm,
    })),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new Error('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new Error('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new Error('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new Error('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new Error('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new Error('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new Error(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new Error(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** The dashboard's dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
