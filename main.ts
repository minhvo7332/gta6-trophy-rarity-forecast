// gta6-trophy-rarity-forecast-01 v1: logit-space CQR quantile fan.
//
// Forecasts, for each of GTA VI's 55 PlayStation trophies, five quantiles
// (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's target,
// from the case file alone. Offline, no network, no model calls.
// Deterministic: the same case file always yields the same bytes.
//
// Method (every parameter is estimated from the case file it runs on, or
// is a fixed prior stated in the RULES text — nothing is fitted to the
// practice cases, whose answers are not in the kit):
//
//   1. Per trophy, two views of the same history, both in log space:
//      (a) a logit-rate trend: OLS of logit(rate) against
//          x = ln(1 + hours since 2026-11-19T00:00Z), over the last 24h
//          of reads (the house curve baseline's regressor);
//      (b) an earners/players decomposition: OLS of log(earners) and
//          log(players) against the same x, extrapolated separately and
//          recombined as a rate. Earners and players are both monotone
//          counts, so splitting the ratio keeps a saturating player
//          influx from masquerading as a collapsing earn rate.
//      Slopes are damped past the last observed x (trend 0.75, earners
//      0.75, players 0.5: growth decelerates, it does not run away), and
//      the two medians are blended 50/50 in logit space.
//   2. Calibration layer (CQR / split-conformal, held-out folds): the
//      case's own history is split at 50/67/80% of its span; the model
//      is fit on the earlier reads and scored on later reads up to a
//      day ahead. Two things come out of it. First, a blend weight: the
//      trend's share of the median is err_persist / (err_persist +
//      err_trend), so a trend that overshoots its own history is
//      trusted less (a linear-in-log-time fit on a saturating curve
//      extrapolates too far at long horizons; without a backtest the
//      trend is hedged 50/50 with persistence). Second, a spread
//      multiplier: a grid search picks the one whose 90% coverage of
//      the blended forecast is closest to 0.9, and half the mean signed
//      logit error is applied as a location correction (clamped to
//      +/-0.5). The final case is never in the fit: the calibration only
//      ever sees reads before the case's checkpoint.
//   3. Flat-history branch (the practice regime): when every read in
//      the history is identical, no trend is identified. The RULES
//      state Exophase's first refresh took players 638 -> 3,868
//      (~6x denominator) and every rate but the 100% one down, so the
//      prior is a refresh shock: median logit shifted -1.4 (a ~4x rate
//      fall) with a 0.5-logit spread. The shift's range is bounded by
//      the RULES' own numbers: the denominator grew 6.06x (logit -1.80
//      if no trophy was earned since) while earners at most tripled in
//      90 minutes (logit -0.69); the fan [-1.4 +/- 0.82] covers that
//      range with margin on both sides, and the (shift, spread) pair
//      is the minimax choice across that range, not a fit to any
//      guessed answer. One refresh over 90 minutes is little guide to
//   4. The trophy at 100% stays 100%: every tracked player has it, and
//      the RULES note it was the one rate the refresh did not lower.
//
// I/O contract (as the kit's starter): read every <id>.in in the input
// directory, write <id>.out to the output directory:
// {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Random-walk spread, logit units per square-root day (the house baseline's). */
const BASE_SIGMA = 0.5;
/** Hours of history the trend fits use (the house curve baseline's window). */
const FIT_WINDOW_HOURS = 24;
/** Slope damping past the last observed x: trend / earners / players. */
const TREND_DAMP = 0.75;
const EARNERS_DAMP = 0.75;
const PLAYERS_DAMP = 0.5;
/** Flat-history prior: median logit shift and spread (see header, point 3). */
const REFRESH_SHIFT = -1.4;
const FLAT_SIGMA = 0.5;
/** Calibration: spread-multiplier grid, clamps, location-correction cap. */
const SPREAD_GRID = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4];
const SPREAD_MIN = 0.35;
const SPREAD_MAX = 2.5;
const BIAS_CAP = 0.5;
/** Backtest prediction horizon cap, in days (about half the final's span). */
const BACKTEST_MAX_DAYS = 1;
/** The log-time epoch: the release day's start, as the house curve uses it. */
const EPOCH_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** x = ln(1 + hours since the epoch) for a read time in ms. */
const logTime = (atMs: number): number => Math.log1p(Math.max(0, (atMs - EPOCH_MS) / 3_600_000));

/** Ordinary least squares of y on x; returns [intercept, slope]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 0) return [my, 0];
  const slope = sxy / sxx;
  return [my - slope * mx, slope];
}

/** Extrapolate a fitted line to xTarget, damping the slope past xLast. */
function extrapolate(intercept: number, slope: number, xLast: number, xTarget: number, damp: number): number {
  const xEff = xTarget <= xLast ? xTarget : xLast + damp * (xTarget - xLast);
  return intercept + slope * xEff;
}

/** One trophy's usable series, oldest first. */
interface Point {
  /** Clock time of the read, ms since the epoch. */
  t: number;
  /** x = ln(1 + hours since 2026-11-19T00:00Z). */
  x: number;
  /** Logit of the earn rate. */
  l: number;
  /** log(earners) and log(players tracked). */
  le: number;
  lp: number;
}

function seriesFor(c: Case, ti: number): Point[] {
  const pts: Point[] = [];
  for (const r of c.history) {
    const rate = r.rates[ti];
    if (rate === null || rate === undefined) continue;
    const players = Math.max(1, r.players_tracked || 1);
    const earners = Math.max(0.5, (rate / 100) * players);
    pts.push({
      t: Date.parse(r.at),
      x: logTime(Date.parse(r.at)),
      l: logit(rate),
      le: Math.log(earners),
      lp: Math.log(players),
    });
  }
  return pts;
}

/** Keep only the points within 24h of clock time of the last point. */
function window24(pts: Point[]): Point[] {
  if (pts.length === 0) return pts;
  const lastT = pts[pts.length - 1].t;
  const kept = pts.filter((p) => lastT - p.t <= FIT_WINDOW_HOURS * 3_600_000);
  return kept.length >= 3 ? kept : pts;
}

/**
 * Median logit at xTarget from the trend + decomposition blend, fit on pts.
 * pts must already be windowed to the fit window.
 */
function blendMedian(pts: Point[], xTarget: number): number {
  const xLast = pts[pts.length - 1].x;
  const xs = pts.map((p) => p.x);
  const [aL, bL] = ols(xs, pts.map((p) => p.l));
  const trend = extrapolate(aL, bL, xLast, xTarget, TREND_DAMP);
  const [aE, bE] = ols(xs, pts.map((p) => p.le));
  const [aP, bP] = ols(xs, pts.map((p) => p.lp));
  const leT = extrapolate(aE, bE, xLast, xTarget, EARNERS_DAMP);
  const lpT = extrapolate(aP, bP, xLast, xTarget, PLAYERS_DAMP);
  const decomp = logit(Math.exp(leT - lpT) * 100);
  return (trend + decomp) / 2;
}

/**
 * Walk-forward calibration on the case's own history. Splits at 50/67/80%
 * of the history span, fits the blend on the earlier reads, and scores
 * later reads up to BACKTEST_MAX_DAYS ahead. Returns:
 *   - w: the blend weight on the trend (vs persistence), chosen by
 *     backtest error: w = err_persist / (err_persist + err_trend), so a
 *     trend that overshoots its own history is trusted less. A trend
 *     fit on a saturating curve extrapolates too far at long horizons;
 *     the backtest is the only evidence of that inside the case file.
 *   - mult: the spread multiplier whose 90% coverage of the BLENDED
 *     forecast is closest to 0.9.
 *   - bias: the location correction (half the mean signed logit error
 *     of the blended forecast, capped).
 * Null when the history cannot support a backtest (too short, too few
 * points, or flat).
 */
function calibrate(pts: Point[], horizonDays: number): { mult: number; bias: number; w: number } | null {
  if (pts.length < 6) return null;
  const spanMs = pts[pts.length - 1].t - pts[0].t;
  if (spanMs < 12 * 3_600_000) return null;
  let distinct = 1;
  for (let i = 1; i < pts.length; i++) if (pts[i].l !== pts[i - 1].l) distinct++;
  if (distinct < 3) return null;
  const maxGapMs = Math.min(horizonDays, BACKTEST_MAX_DAYS) * 86_400_000;
  interface Bt {
    actual: number;
    medTrend: number;
    medPersist: number;
    gapDays: number;
  }
  const records: Bt[] = [];
  for (const frac of [0.5, 0.67, 0.8]) {
    const cutT = pts[0].t + frac * spanMs;
    let cut = 0;
    while (cut < pts.length && pts[cut].t <= cutT) cut++;
    if (cut < 3 || cut >= pts.length) continue;
    const train = window24(pts.slice(0, cut));
    const tCut = train[train.length - 1].t;
    const lastTrainL = train[train.length - 1].l;
    for (let j = cut; j < pts.length; j++) {
      const p = pts[j];
      if (p.t - tCut > maxGapMs) break;
      records.push({
        actual: p.l,
        medTrend: blendMedian(train, p.x),
        medPersist: lastTrainL,
        gapDays: Math.max(1 / 1440, (p.t - tCut) / 86_400_000),
      });
    }
  }
  if (records.length < 20) return null;
  let errTrend = 0;
  let errPersist = 0;
  for (const r of records) {
    errTrend += Math.abs(r.actual - r.medTrend);
    errPersist += Math.abs(r.actual - r.medPersist);
  }
  errTrend /= records.length;
  errPersist /= records.length;
  const w = errTrend + errPersist <= 0 ? 0.5 : errPersist / (errTrend + errPersist);
  const coverCount: number[] = SPREAD_GRID.map(() => 0);
  let signedErr = 0;
  for (const r of records) {
    const med = w * r.medTrend + (1 - w) * r.medPersist;
    const d = r.actual - med;
    signedErr += d;
    for (let g = 0; g < SPREAD_GRID.length; g++) {
      const s = Math.max(0.05, BASE_SIGMA * Math.sqrt(r.gapDays) * SPREAD_GRID[g]);
      if (med - 1.6448536269514722 * s <= r.actual && r.actual <= med + 1.6448536269514722 * s) coverCount[g]++;
    }
  }
  let best = 0;
  let bestScore = Infinity;
  for (let g = 0; g < SPREAD_GRID.length; g++) {
    const score = Math.abs(coverCount[g] / records.length - 0.9);
    if (score < bestScore - 1e-9) {
      bestScore = score;
      best = g;
    }
  }
  const bias = Math.max(-BIAS_CAP, Math.min(BIAS_CAP, -0.5 * (signedErr / records.length)));
  return { mult: SPREAD_GRID[best], bias, w };
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const horizonDays = c.horizon_minutes / 1440;
  const last = c.history[c.history.length - 1];
  const targetX = logTime(Date.parse(c.target));
  const out: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const lastRate = last.rates[i];
    if (lastRate !== null && lastRate !== undefined && lastRate >= 99.995) {
      out.push({ id: t.id, q: [100, 100, 100, 100, 100] });
      continue;
    }
    const pts = seriesFor(c, i);
    const baseRate = lastRate === null || lastRate === undefined ? 0 : lastRate;
    if (pts.length === 0) {
      const l = logit(baseRate);
      const s = BASE_SIGMA * Math.sqrt(horizonDays);
      out.push({ id: t.id, q: mono(Z.map((z) => round4(invLogit(l + z * s)))) });
      continue;
    }
    // Flat history: no trend is identified -> refresh-shock prior.
    let distinct = 1;
    for (let j = 1; j < pts.length; j++) if (pts[j].l !== pts[j - 1].l) distinct++;
    if (distinct < 2) {
      const l = logit(baseRate) + REFRESH_SHIFT;
      out.push({ id: t.id, q: mono(Z.map((z) => round4(invLogit(l + z * FLAT_SIGMA)))) });
      continue;
    }
    const win = window24(pts);
    const med = blendMedian(win, targetX);
    const cal = calibrate(pts, horizonDays);
    const mult = cal ? cal.mult : 1;
    const bias = cal ? cal.bias : 0;
    // Without a backtest, hedge the trend 50/50 with persistence; with
    // one, use its weight. The trend's overshoot is the failure mode
    // the backtest exists to catch.
    const w = cal ? cal.w : 0.5;
    const lastL = logit(baseRate);
    const spread = Math.max(SPREAD_MIN, Math.min(SPREAD_MAX, BASE_SIGMA * Math.sqrt(horizonDays) * mult));
    const centre = w * med + (1 - w) * lastL + bias;
    out.push({ id: t.id, q: mono(Z.map((z) => round4(invLogit(centre + z * spread)))) });
  }
  return out;
}

/** Enforce non-decreasing quantiles and the [0, 100] range. */
function mono(q: number[]): number[] {
  for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  return q.map((v) => Math.min(100, Math.max(0, v)));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
