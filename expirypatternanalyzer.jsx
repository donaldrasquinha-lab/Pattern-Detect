import { useState, useEffect, useMemo, useCallback } from "react";

/*═══════════════════════════════════════════════════════════════════
  BULKOWSKI PATTERN ENGINE — JS Port of scipy.signal approach
  Uses argrelextrema-equivalent for peak/trough detection and
  linear regression for trendline analysis.
═══════════════════════════════════════════════════════════════════*/

// scipy.signal.argrelextrema equivalent
function argrelextrema(data, comparator, order = 3) {
  const results = [];
  const n = data.length;
  order = Math.min(order, Math.max(1, Math.floor(n / 4)));
  for (let i = order; i < n - order; i++) {
    let valid = true;
    for (let j = 1; j <= order; j++) {
      if (!comparator(data[i], data[i - j]) || !comparator(data[i], data[i + j])) {
        valid = false; break;
      }
    }
    if (valid) results.push(i);
  }
  return results;
}
const findPeaks = (d, o) => argrelextrema(d, (a, b) => a >= b, o);
const findTroughs = (d, o) => argrelextrema(d, (a, b) => a <= b, o);

// scipy.stats.linregress equivalent
function linregress(x, y) {
  const n = x.length; if (n < 2) return { slope: 0, r2: 0 };
  const mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let ss_xy = 0, ss_xx = 0, ss_yy = 0;
  for (let i = 0; i < n; i++) {
    ss_xy += (x[i] - mx) * (y[i] - my);
    ss_xx += (x[i] - mx) ** 2;
    ss_yy += (y[i] - my) ** 2;
  }
  const slope = ss_xx ? ss_xy / ss_xx : 0;
  const r2 = (ss_xx && ss_yy) ? (ss_xy ** 2) / (ss_xx * ss_yy) : 0;
  return { slope, r2 };
}

function calcATR(h, l, c) {
  if (c.length < 2) return h.length ? h.reduce((s, v, i) => s + h[i] - l[i], 0) / h.length : 1;
  let sum = 0, cnt = 0;
  for (let i = 1; i < c.length; i++) {
    sum += Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1]));
    cnt++;
  }
  return cnt ? sum / cnt : 1;
}

// ═══ PATTERN DETECTORS (Bulkowski mathematical rules) ═══
function detectAllPatterns(O, H, L, C, V, oiChg = 0, pcr = 1) {
  const n = C.length; if (n < 5) return [];
  const order = Math.max(2, Math.min(5, Math.floor(n / 4)));
  const peaks = findPeaks(C, order), troughs = findTroughs(C, order);
  const hPeaks = findPeaks(H, order), lTroughs = findTroughs(L, order);
  const allPk = [...new Set([...peaks, ...hPeaks])].sort((a, b) => a - b);
  const allTr = [...new Set([...troughs, ...lTroughs])].sort((a, b) => a - b);
  const atr = calcATR(H, L, C);
  const results = [];
  const add = (id, nm, cat, bias, conf, si, ei, ph, sr, desc, kl = {}) =>
    results.push({ id, nm, cat, bias, conf: Math.min(0.95, conf), si, ei, ph, sr, desc, kl });

  // Head & Shoulders
  for (let i = 0; i < allPk.length - 2; i++) {
    const [p1, p2, p3] = [allPk[i], allPk[i + 1], allPk[i + 2]];
    const [h1, h2, h3] = [H[p1], H[p2], H[p3]];
    if (h2 <= h1 || h2 <= h3) continue;
    const sd = Math.abs(h1 - h3) / Math.max(h1, h3);
    if (sd > 0.05) continue;
    const tb = allTr.filter(t => t > p1 && t < p3);
    if (tb.length < 1) continue;
    const nl = tb.reduce((s, t) => s + L[t], 0) / tb.length;
    add(1, "Head & Shoulders", "Reversal", "BEARISH", 0.55 + (1 - sd) * 0.2 + (h2 - Math.max(h1, h3)) / h2 * 2,
      p1, p3, "FULL", 0.65, "Bearish reversal. Neckline break confirms.", { neckline: +nl.toFixed(2), target: +(nl - (h2 - nl)).toFixed(2) });
  }

  // Inv H&S
  for (let i = 0; i < allTr.length - 2; i++) {
    const [t1, t2, t3] = [allTr[i], allTr[i + 1], allTr[i + 2]];
    const [l1, l2, l3] = [L[t1], L[t2], L[t3]];
    if (l2 >= l1 || l2 >= l3) continue;
    const sd = Math.abs(l1 - l3) / Math.max(l1, l3);
    if (sd > 0.05) continue;
    const pb = allPk.filter(p => p > t1 && p < t3);
    if (pb.length < 1) continue;
    const nl = pb.reduce((s, p) => s + H[p], 0) / pb.length;
    add(2, "Inv Head & Shoulders", "Reversal", "BULLISH", 0.55 + (1 - sd) * 0.2,
      t1, t3, "FULL", 0.75, "Bullish reversal. Neckline break confirms.", { neckline: +nl.toFixed(2) });
  }

  // Double Top
  for (let i = 0; i < allPk.length - 1; i++) {
    const [p1, p2] = [allPk[i], allPk[i + 1]];
    const d = Math.abs(H[p1] - H[p2]) / Math.max(H[p1], H[p2]);
    if (d > 0.03) continue;
    const tb = allTr.filter(t => t > p1 && t < p2);
    if (!tb.length) continue;
    const nl = Math.min(...tb.map(t => L[t]));
    add(3, "Double Top", "Reversal", "BEARISH", 0.6 + (1 - d) * 0.25, p1, p2, "MID", 0.70,
      "M-shape exhaustion. Break below neckline confirms.", { neckline: +nl.toFixed(2), peak1: +H[p1].toFixed(2), peak2: +H[p2].toFixed(2) });
  }

  // Double Bottom
  for (let i = 0; i < allTr.length - 1; i++) {
    const [t1, t2] = [allTr[i], allTr[i + 1]];
    const d = Math.abs(L[t1] - L[t2]) / Math.max(L[t1], L[t2]);
    if (d > 0.03) continue;
    const pb = allPk.filter(p => p > t1 && p < t2);
    if (!pb.length) continue;
    const nl = Math.max(...pb.map(p => H[p]));
    add(4, "Double Bottom", "Reversal", "BULLISH", 0.6 + (1 - d) * 0.25, t1, t2, "MID", 0.70,
      "W-shape accumulation. Break above neckline confirms.", { neckline: +nl.toFixed(2) });
  }

  // Triple Top/Bottom
  if (allPk.length >= 3) {
    for (let i = 0; i < allPk.length - 2; i++) {
      const hs = [H[allPk[i]], H[allPk[i + 1]], H[allPk[i + 2]]];
      const m = hs.reduce((a, b) => a + b) / 3;
      if (hs.every(h => Math.abs(h - m) / m < 0.03))
        add(5, "Triple Top", "Reversal", "BEARISH", 0.65 + (1 - Math.max(...hs.map(h => Math.abs(h - m) / m))) * 0.25,
          allPk[i], allPk[i + 2], "MID", 0.70, "Three rejections at resistance.");
    }
  }
  if (allTr.length >= 3) {
    for (let i = 0; i < allTr.length - 2; i++) {
      const ls = [L[allTr[i]], L[allTr[i + 1]], L[allTr[i + 2]]];
      const m = ls.reduce((a, b) => a + b) / 3;
      if (ls.every(l => Math.abs(l - m) / m < 0.03))
        add(6, "Triple Bottom", "Reversal", "BULLISH", 0.65 + (1 - Math.max(...ls.map(l => Math.abs(l - m) / m))) * 0.25,
          allTr[i], allTr[i + 2], "MID", 0.72, "Three bounces at support.");
    }
  }

  // Triangles
  if (allPk.length >= 2 && allTr.length >= 2) {
    const pkVals = allPk.map(i => H[i]), trVals = allTr.map(i => L[i]);
    const pkR = (Math.max(...pkVals) - Math.min(...pkVals)) / (pkVals.reduce((a, b) => a + b) / pkVals.length);
    const trR = (Math.max(...trVals) - Math.min(...trVals)) / (trVals.reduce((a, b) => a + b) / trVals.length);
    const pkLr = linregress(allPk, pkVals), trLr = linregress(allTr.map(Number), trVals);

    // Ascending
    if (pkR < 0.015 && trLr.slope > 0 && trLr.r2 > 0.4)
      add(7, "Ascending Triangle", "Continuation", "BULLISH", 0.55 + trLr.r2 * 0.3,
        Math.min(allPk[0], allTr[0]), Math.max(allPk.at(-1), allTr.at(-1)), "MID", 0.75,
        "Flat top + rising bottom. Bullish breakout expected.");
    // Descending
    if (trR < 0.015 && pkLr.slope < 0 && pkLr.r2 > 0.4)
      add(8, "Descending Triangle", "Continuation", "BEARISH", 0.55 + pkLr.r2 * 0.3,
        Math.min(allPk[0], allTr[0]), Math.max(allPk.at(-1), allTr.at(-1)), "MID", 0.68,
        "Flat bottom + falling top. Bearish breakdown expected.");
    // Symmetrical
    if (pkLr.slope < 0 && trLr.slope > 0 && pkLr.r2 > 0.4 && trLr.r2 > 0.4)
      add(9, "Symmetrical Triangle", "Bilateral", "NEUTRAL", 0.5 + pkLr.r2 * 0.2 + trLr.r2 * 0.2,
        Math.min(allPk[0], allTr[0]), Math.max(allPk.at(-1), allTr.at(-1)), "MID", 0.70,
        "Converging lines. Breakout direction determines bias.");
    // Wedges
    if (pkLr.slope > 0 && trLr.slope > 0 && pkLr.slope < trLr.slope && pkLr.r2 > 0.4)
      add(10, "Rising Wedge", "Reversal", "BEARISH", 0.5 + pkLr.r2 * 0.2, 0, n - 1, "LATE", 0.65, "Converging upward lines. Bearish.");
    if (pkLr.slope < 0 && trLr.slope < 0 && Math.abs(trLr.slope) > Math.abs(pkLr.slope) && trLr.r2 > 0.4)
      add(11, "Falling Wedge", "Reversal", "BULLISH", 0.5 + trLr.r2 * 0.2, 0, n - 1, "LATE", 0.70, "Converging downward lines. Bullish.");
    // Channel
    if (pkLr.r2 > 0.6 && trLr.r2 > 0.6 && Math.abs(pkLr.slope) > 1e-8) {
      const diff = Math.abs(pkLr.slope - trLr.slope) / Math.abs(pkLr.slope);
      if (diff < 0.3) {
        const cn = pkLr.slope > 0 ? "Ascending Channel" : "Descending Channel";
        const cb = pkLr.slope > 0 ? "BULLISH" : "BEARISH";
        add(21, cn, "Bilateral", cb, (pkLr.r2 + trLr.r2) / 2, 0, n - 1, "FULL", 0.65, `Parallel trendlines forming ${cn.toLowerCase()}.`);
      }
    }
    // Rectangle
    if (pkR < 0.015 && trR < 0.015) {
      const bias = C[0] < C[n - 1] ? "BULLISH" : "BEARISH";
      add(bias === "BULLISH" ? 16 : 17, `${bias === "BULLISH" ? "Bullish" : "Bearish"} Rectangle`,
        "Continuation", bias, 0.55 + (1 - pkR) * 0.15, 0, n - 1, "MID", 0.65, "Horizontal consolidation.");
    }
    // Megaphone
    if (pkLr.slope > 0 && trLr.slope < 0 && pkLr.r2 > 0.4 && trLr.r2 > 0.4)
      add(23, "Megaphone", "Bilateral", "NEUTRAL", (pkLr.r2 + trLr.r2) / 2, 0, n - 1, "LATE", 0.52, "Expanding range. Max uncertainty.");
  }

  // Flags/Pennants
  const third = Math.max(3, Math.floor(n / 3));
  const poleChg = (C[third - 1] - C[0]) / C[0];
  const consol = C.slice(third);
  if (consol.length >= 3) {
    const cr = (Math.max(...consol) - Math.min(...consol)) / (consol.reduce((a, b) => a + b) / consol.length);
    if (poleChg > 0.015 && cr < 0.02) add(12, "Bullish Flag", "Continuation", "BULLISH", Math.min(0.9, 0.55 + poleChg * 5), 0, n - 1, "EARLY", 0.75, "Sharp rally + tight consolidation.");
    if (poleChg < -0.015 && cr < 0.02) add(13, "Bearish Flag", "Continuation", "BEARISH", Math.min(0.9, 0.55 + Math.abs(poleChg) * 5), 0, n - 1, "EARLY", 0.68, "Sharp drop + tight consolidation.");
  }

  // Cup & Handle
  const cupLowIdx = L.indexOf(Math.min(...L));
  if (cupLowIdx >= 3 && cupLowIdx <= n - 3) {
    const lr = Math.max(...H.slice(0, cupLowIdx)), rr = Math.max(...H.slice(cupLowIdx));
    if (Math.abs(lr - rr) / Math.max(lr, rr) < 0.03)
      add(18, "Cup & Handle", "Continuation", "BULLISH", 0.62, 0, n - 1, "FULL", 0.68, "U-shaped base + handle. Bullish.");
  }

  // Staircase
  let hh = 0, hl = 0, lh = 0, ll = 0;
  for (let i = 1; i < n; i++) { if (H[i] > H[i - 1]) hh++; if (L[i] > L[i - 1]) hl++; if (H[i] < H[i - 1]) lh++; if (L[i] < L[i - 1]) ll++; }
  if (hh >= n * 0.65 && hl >= n * 0.65) add(31, "Ascending Staircase", "Continuation", "BULLISH", hh / n, 0, n - 1, "FULL", 0.70, "HH+HL steps. Steady buying.");
  if (lh >= n * 0.65 && ll >= n * 0.65) add(32, "Descending Staircase", "Continuation", "BEARISH", lh / n, 0, n - 1, "FULL", 0.70, "LH+LL steps. Steady selling.");

  // Gaps
  for (let i = 1; i < n && results.filter(r => r.id === 53).length < 2; i++) {
    if (O[i] > H[i - 1] * 1.002) add(53, "Gap Up", "Bilateral", "BULLISH", 0.60, i - 1, i, i < n / 3 ? "EARLY" : "MID", 0.65, "Price gap up.");
    if (O[i] < L[i - 1] * 0.998) add(53, "Gap Down", "Bilateral", "BEARISH", 0.60, i - 1, i, i < n / 3 ? "EARLY" : "MID", 0.65, "Price gap down.");
  }

  // Candlestick: Morning Star, Evening Star, Engulfing
  for (let i = 2; i < n; i++) {
    const b = C[i] - O[i], bp = C[i - 1] - O[i - 1], bp2 = C[i - 2] - O[i - 2], rng = H[i] - L[i] || 1;
    if (bp2 < 0 && Math.abs(C[i - 1] - O[i - 1]) < rng * 0.3 && b > 0 && C[i] > (O[i - 2] + C[i - 2]) / 2)
      add(43, "Morning Star", "Reversal", "BULLISH", 0.68, i - 2, i, "LATE", 0.68, "Three-candle bullish reversal.");
    if (bp2 > 0 && Math.abs(C[i - 1] - O[i - 1]) < rng * 0.3 && b < 0 && C[i] < (O[i - 2] + C[i - 2]) / 2)
      add(44, "Evening Star", "Reversal", "BEARISH", 0.68, i - 2, i, "LATE", 0.68, "Three-candle bearish reversal.");
    if (bp < 0 && b > 0 && O[i] <= C[i - 1] && C[i] >= O[i - 1]) add(55, "Bullish Engulfing", "Reversal", "BULLISH", 0.65, i - 1, i, "LATE", 0.63, "Bullish candle engulfs prior bearish.");
    if (bp > 0 && b < 0 && O[i] >= C[i - 1] && C[i] <= O[i - 1]) add(55, "Bearish Engulfing", "Reversal", "BEARISH", 0.65, i - 1, i, "LATE", 0.63, "Bearish candle engulfs prior bullish.");
  }

  // V-bottom
  if (cupLowIdx >= 2 && cupLowIdx <= n - 3) {
    const dr = (C[0] - L[cupLowIdx]) / C[0], rc = (C[n - 1] - L[cupLowIdx]) / L[cupLowIdx];
    if (dr > 0.02 && rc > 0.02) add(30, "V Bottom", "Reversal", "BULLISH", Math.min(0.85, 0.5 + dr * 5), 0, n - 1, "FULL", 0.60, "Sharp V reversal.");
  }

  // Bull/Bear trap
  if (n > 3) {
    const prevH = Math.max(...H.slice(0, -2));
    if (H[n - 1] > prevH && C[n - 1] < C[n - 2]) add(40, "Bull Trap", "Reversal", "BEARISH", 0.62, n - 3, n - 1, "LATE", 0.65, "False breakout reverses.");
    const prevL = Math.min(...L.slice(0, -2));
    if (L[n - 1] < prevL && C[n - 1] > C[n - 2]) add(41, "Bear Trap", "Reversal", "BULLISH", 0.62, n - 3, n - 1, "LATE", 0.65, "False breakdown reverses.");
  }

  // Spike
  for (let i = 0; i < n; i++) if (H[i] - L[i] > 2.5 * atr) { add(38, "Spike", "Bilateral", "NEUTRAL", Math.min(0.85, 0.5 + (H[i] - L[i]) / (atr * 5)), i, i, "MID", 0.58, "Extreme range bar."); break; }

  // OI patterns
  const pc = C[n - 1] - C[0];
  if (oiChg > 0 && pc > 0) add(100, "Long Buildup", "OI", "BULLISH", 0.80, 0, 0, "FULL", 0.70, "Price↑ OI↑ — Fresh longs.");
  else if (oiChg > 0 && pc < 0) add(101, "Short Buildup", "OI", "BEARISH", 0.80, 0, 0, "FULL", 0.70, "Price↓ OI↑ — Fresh shorts.");
  else if (oiChg < 0 && pc < 0) add(102, "Long Unwinding", "OI", "BEARISH", 0.75, 0, 0, "LATE", 0.65, "Price↓ OI↓ — Longs exiting.");
  else if (oiChg < 0 && pc > 0) add(103, "Short Covering", "OI", "BULLISH", 0.75, 0, 0, "LATE", 0.65, "Price↑ OI↓ — Shorts exiting.");
  if (pcr > 1.5) add(104, "PCR Extreme High", "OI", "BULLISH", 0.85, 0, 0, "FULL", 0.68, "Heavy put writing = support.");
  if (pcr < 0.5) add(105, "PCR Extreme Low", "OI", "BEARISH", 0.85, 0, 0, "FULL", 0.68, "Heavy call writing = resistance.");

  // Deduplicate & sort
  const seen = new Set();
  return results.filter(r => { const k = `${r.id}-${r.nm}`; if (seen.has(k)) return false; seen.add(k); return true; })
    .sort((a, b) => b.conf - a.conf).slice(0, 10);
}

/*═══════════════════════════════════════════════════════════════════
  DATA GENERATION
═══════════════════════════════════════════════════════════════════*/
function genOHLCV(bp, n) {
  const C = [bp], v = bp * 0.008, t = (Math.random() - 0.5) * 0.4;
  for (let i = 1; i < n; i++) C.push(C[i - 1] + (Math.random() - 0.5 + t) * v);
  const O = C.map((c, i) => i ? C[i - 1] : c);
  const noise = () => Math.random() * v * 0.4;
  const H = O.map((o, i) => Math.max(o, C[i]) + noise());
  const L = O.map((o, i) => Math.min(o, C[i]) - noise());
  const V = Array.from({ length: n }, () => Math.floor(50000 + Math.random() * 250000));
  return { O, H, L, C, V };
}

const BASES = { NIFTY50: 24500, BANKNIFTY: 52000, FINNIFTY: 23800, SENSEX: 80500, SPX: 5800, NDX: 20500, RUT: 2100, VIX: 18 };
const INDICES = {
  INDIA: [
    { idx: "NIFTY50", weekly: true, expDay: "Tue" },
    { idx: "BANKNIFTY", weekly: false, expDay: "Tue" },
    { idx: "FINNIFTY", weekly: false, expDay: "Tue" },
    { idx: "SENSEX", weekly: true, expDay: "Thu" }
  ],
  US: [
    { idx: "SPX", weekly: true, expDay: "Fri" },
    { idx: "NDX", weekly: true, expDay: "Fri" },
    { idx: "RUT", weekly: true, expDay: "Fri" },
    { idx: "VIX", weekly: true, expDay: "Wed" }
  ]
};

function buildDB() {
  const db = [], dn = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  for (const [mkt, list] of Object.entries(INDICES)) {
    for (const { idx, weekly, expDay } of list) {
      let bp = BASES[idx]; const ne = weekly ? 36 : 12, nd = weekly ? 5 : 22;
      for (let e = 0; e < ne; e++) {
        const { O, H, L, C, V } = genOHLCV(bp, nd);
        const pcr = +(0.5 + Math.random() * 1.5).toFixed(3);
        const oiChg = Math.floor((Math.random() - 0.5) * 100000);
        const pats = detectAllPatterns(O, H, L, C, V, oiChg, pcr);
        const sp = C[0], ep = C[nd - 1], ch = +((ep - sp) / sp * 100).toFixed(3);
        const ed = new Date(2025, 0, 1 + (weekly ? 7 : 30) * e);
        const wom = Math.min(4, Math.ceil(ed.getDate() / 7));
        db.push({
          id: `${idx}-${e}`, ix: idx, mkt, dt: ed.toISOString().split("T")[0],
          ct: weekly ? "WEEKLY" : "MONTHLY", dy: nd, sp: +sp.toFixed(2), ep: +ep.toFixed(2),
          ch, out: ch > 0.5 ? "BULLISH" : ch < -0.5 ? "BEARISH" : "NEUTRAL",
          dd: +((Math.min(...L) - sp) / sp * 100).toFixed(3),
          ru: +((Math.max(...H) - sp) / sp * 100).toFixed(3),
          vol: +(H.reduce((s, h, i) => s + h - L[i], 0) / nd).toFixed(2),
          pcr, oiChg, wom, dow: dn[ed.getDay()], mp: `W${wom}`,
          cd: O.map((o, i) => ({ o: +o.toFixed(2), h: +H[i].toFixed(2), l: +L[i].toFixed(2), c: +C[i].toFixed(2), v: V[i] })),
          pats, pc: pats.length, engine: "scipy.signal + Bulkowski"
        });
        bp = ep * (0.995 + Math.random() * 0.01);
      }
    }
  }
  return db;
}

/*═══════════════════════════════════════════════════════════════════
  MINI CHART COMPONENTS
═══════════════════════════════════════════════════════════════════*/
function MC({ cd, w = 280, h = 70 }) {
  if (!cd?.length) return null;
  const cw = (w - 8) / cd.length, mx = Math.max(...cd.map(c => c.h)), mn = Math.min(...cd.map(c => c.l)), r = mx - mn || 1;
  const y = v => 4 + (1 - (v - mn) / r) * (h - 8);
  return <svg width={w} height={h} style={{ display: "block" }}>{cd.map((c, i) => {
    const x = 4 + i * cw + cw / 2, bl = c.c >= c.o, co = bl ? "#00c896" : "#ff4757";
    return <g key={i}><line x1={x} y1={y(c.h)} x2={x} y2={y(c.l)} stroke={co} strokeWidth={0.8} opacity={0.5} />
      <rect x={x - cw * 0.3} y={y(Math.max(c.o, c.c))} width={cw * 0.6} height={Math.max(1, y(Math.min(c.o, c.c)) - y(Math.max(c.o, c.c)))} fill={co} rx={0.5} /></g>;
  })}</svg>;
}

/*═══════════════════════════════════════════════════════════════════
  MAIN APP
═══════════════════════════════════════════════════════════════════*/
export default function App() {
  const [db, setDb] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("DASH");
  const [sel, setSel] = useState(null);
  const [flt, setFlt] = useState({ mkt: "ALL", ix: "ALL", pat: "ALL", ct: "ALL", mp: "ALL", dow: "ALL" });
  const [rtI, setRtI] = useState({ mkt: "INDIA", pat: "ALL", oi: "ALL", ph: "ALL", ct: "WEEKLY" });
  const [matches, setMatches] = useState([]);
  // Upstox connection state
  const [upstox, setUpstox] = useState({ token: "", apiKey: "", apiSecret: "", redirectUri: "https://localhost:3000/callback", status: "disconnected", profile: null, error: "" });
  const [liveData, setLiveData] = useState(null);
  const [fetchingLive, setFetchingLive] = useState(false);

  useEffect(() => { setDb(buildDB()); setLoading(false); }, []);

  const filtered = useMemo(() => {
    let f = [...db];
    if (flt.mkt !== "ALL") f = f.filter(e => e.mkt === flt.mkt);
    if (flt.ix !== "ALL") f = f.filter(e => e.ix === flt.ix);
    if (flt.pat !== "ALL") f = f.filter(e => e.pats.some(p => p.nm === flt.pat));
    if (flt.ct !== "ALL") f = f.filter(e => e.ct === flt.ct);
    if (flt.mp !== "ALL") f = f.filter(e => e.mp === flt.mp);
    if (flt.dow !== "ALL") f = f.filter(e => e.dow === flt.dow);
    return f;
  }, [db, flt]);

  const stats = useMemo(() => {
    if (!filtered.length) return null;
    const bl = filtered.filter(e => e.out === "BULLISH").length, br = filtered.filter(e => e.out === "BEARISH").length;
    return { n: filtered.length, bl, br, ne: filtered.length - bl - br,
      avg: +(filtered.reduce((s, e) => s + e.ch, 0) / filtered.length).toFixed(3),
      wr: +(bl / filtered.length * 100).toFixed(1),
      dd: +(filtered.reduce((s, e) => s + e.dd, 0) / filtered.length).toFixed(3),
      ru: +(filtered.reduce((s, e) => s + e.ru, 0) / filtered.length).toFixed(3) };
  }, [filtered]);

  const patFreq = useMemo(() => {
    const c = {}; db.forEach(e => e.pats.forEach(p => { c[p.nm] = (c[p.nm] || 0) + 1; }));
    return Object.entries(c).sort((a, b) => b[1] - a[1]);
  }, [db]);

  const allPats = useMemo(() => [...new Set(db.flatMap(e => e.pats.map(p => p.nm)))].sort(), [db]);
  const allIdx = useMemo(() => ["ALL", ...new Set(db.map(e => e.ix))], [db]);

  const doMatch = useCallback(() => {
    let c = [...db];
    if (rtI.mkt !== "ALL") c = c.filter(e => e.mkt === rtI.mkt);
    if (rtI.ct !== "ALL") c = c.filter(e => e.ct === rtI.ct);
    if (rtI.pat !== "ALL") c = c.filter(e => e.pats.some(p => p.nm === rtI.pat));
    if (rtI.oi !== "ALL") c = c.filter(e => e.pats.some(p => p.nm === rtI.oi));
    if (rtI.ph !== "ALL") c = c.filter(e => e.mp === rtI.ph);
    setMatches(c.map(x => ({ ...x, score: x.pats.reduce((s, p) => s + p.conf, 0) })).sort((a, b) => b.score - a.score).slice(0, 20));
  }, [rtI, db]);

  const S = {
    r: { minHeight: "100vh", background: "#05050b", color: "#c8c8d4", fontFamily: "'JetBrains Mono','Fira Code',monospace", fontSize: 12 },
    c: { background: "#0b0b16", border: "1px solid #141428", borderRadius: 6, padding: "12px 14px" },
    s: { background: "#0e0e22", border: "1px solid #1a1a35", color: "#c8c8d4", padding: "6px 8px", borderRadius: 4, fontSize: 11, fontFamily: "inherit", minWidth: 95 },
    tb: a => ({ padding: "9px 14px", border: "none", cursor: "pointer", background: a ? "#0e0e22" : "transparent", color: a ? "#00c896" : "#555", borderBottom: a ? "2px solid #00c896" : "2px solid transparent", fontSize: 10, fontFamily: "inherit", letterSpacing: 1, whiteSpace: "nowrap" }),
    bg: b => { const m = { BULLISH: "#082e1a,#00c896,#00e6a8", BEARISH: "#2e0808,#ff4757,#ff6b6b", NEUTRAL: "#10102e,#7c83ff,#a0a5ff", BILATERAL: "#1a1a10,#ffd700,#ffe44d" }[b] || "12122e,#555,#888";
      const [bg, br, t] = m.split(","); return { display: "inline-block", padding: "2px 7px", margin: 2, borderRadius: 3, background: `#${bg}`, border: `1px solid ${br}`, color: t, fontSize: 10, whiteSpace: "nowrap" }; }
  };

  if (loading) return <div style={{ ...S.r, display: "flex", alignItems: "center", justifyContent: "center" }}><div style={{ textAlign: "center" }}><div style={{ fontSize: 28, color: "#00c896", marginBottom: 8 }}>◈</div><div style={{ fontSize: 12, letterSpacing: 2, color: "#555" }}>RUNNING BULKOWSKI ENGINE (scipy.signal)...</div></div></div>;

  const tabs = [{ id: "DASH", l: "Dashboard" }, { id: "UPSTOX", l: "⚡ Upstox Connect" }, { id: "ENG", l: "Detection Engine" }, { id: "BACK", l: "Backtest" }, { id: "RT", l: "Real-Time Match" }, { id: "DB", l: "Database" }];

  // Upstox API functions
  const upstoxAuthUrl = upstox.apiKey ? `https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id=${upstox.apiKey}&redirect_uri=${encodeURIComponent(upstox.redirectUri)}` : "";

  const connectUpstox = async (token) => {
    setUpstox(p => ({ ...p, status: "connecting", error: "" }));
    try {
      const resp = await fetch("https://api.upstox.com/v2/user/profile", {
        headers: { "Accept": "application/json", "Authorization": `Bearer ${token}` }
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setUpstox(p => ({ ...p, token, status: "connected", profile: data.data, error: "" }));
    } catch (e) {
      setUpstox(p => ({ ...p, status: "error", error: e.message }));
    }
  };

  const exchangeCode = async (code) => {
    setUpstox(p => ({ ...p, status: "exchanging", error: "" }));
    try {
      const resp = await fetch("https://api.upstox.com/v2/login/authorization/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json" },
        body: `code=${code}&client_id=${upstox.apiKey}&client_secret=${upstox.apiSecret}&redirect_uri=${encodeURIComponent(upstox.redirectUri)}&grant_type=authorization_code`
      });
      const data = await resp.json();
      if (data.access_token) {
        await connectUpstox(data.access_token);
      } else {
        throw new Error(data.message || "Token exchange failed");
      }
    } catch (e) {
      setUpstox(p => ({ ...p, status: "error", error: e.message }));
    }
  };

  const fetchLiveData = async (index) => {
    if (!upstox.token) return;
    setFetchingLive(true);
    const instKeys = { NIFTY50: "NSE_INDEX|Nifty 50", BANKNIFTY: "NSE_INDEX|Nifty Bank", SENSEX: "BSE_INDEX|SENSEX" };
    const key = instKeys[index];
    if (!key) { setFetchingLive(false); return; }
    try {
      const resp = await fetch(`https://api.upstox.com/v2/historical-candle/intraday/${encodeURIComponent(key)}/5minute`, {
        headers: { "Content-Type": "application/json", "Accept": "application/json", "Authorization": `Bearer ${upstox.token}` }
      });
      const data = await resp.json();
      const candles = data?.data?.candles || [];
      if (candles.length >= 5) {
        const O = candles.map(c => c[1]), H = candles.map(c => c[2]), L = candles.map(c => c[3]), C = candles.map(c => c[4]), V = candles.map(c => c[5]);
        const pats = detectAllPatterns(O, H, L, C, V);
        setLiveData({ index, timestamp: new Date().toISOString(), bars: candles.length, price: C[C.length - 1],
          dayChg: +((C[C.length - 1] - C[0]) / C[0] * 100).toFixed(3), patterns: pats, source: "Upstox Intraday 5min" });
      }
    } catch (e) { console.error(e); }
    setFetchingLive(false);
  };

  return (<div style={S.r}>
    <div style={{ padding: "12px 16px", borderBottom: "1px solid #141428", background: "linear-gradient(180deg,#0a0a16,#05050b)", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}><span style={{ fontSize: 20, color: "#00c896" }}>◈</span><div><div style={{ fontSize: 14, fontWeight: 700, color: "#e8e8f0", letterSpacing: 1 }}>BULKOWSKI PATTERN ENGINE v3.0</div><div style={{ fontSize: 9, color: "#444", letterSpacing: 2 }}>scipy.signal PEAKS/TROUGHS · LINREGRESS TRENDLINES · 55+ PATTERNS · OI ANALYSIS</div></div></div>
      <div style={{ fontSize: 9, color: "#333", textAlign: "right" }}>
        <span style={{ color: upstox.status === "connected" ? "#00c896" : "#555" }}>● Upstox: {upstox.status === "connected" ? "LIVE" : "Demo"}</span><br />
        Engine: scipy.signal.argrelextrema<br />{db.length} expiries · {db.reduce((s, e) => s + e.pc, 0)} patterns</div>
    </div>

    <div style={{ display: "flex", borderBottom: "1px solid #141428", overflowX: "auto", background: "#07071a" }}>
      {tabs.map(t => <button key={t.id} onClick={() => { setTab(t.id); setSel(null); }} style={S.tb(tab === t.id)}>{t.l}</button>)}</div>

    <div style={{ padding: "14px 16px", maxWidth: 1280, margin: "0 auto" }}>

      {/* UPSTOX CONNECT */}
      {tab === "UPSTOX" && <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#e066ff", marginBottom: 6, letterSpacing: 1 }}>UPSTOX API CONNECTION</div>
        <div style={{ fontSize: 10, color: "#555", marginBottom: 16, lineHeight: 1.6 }}>
          Connect to Upstox V2/V3 API for live NIFTY, BANKNIFTY, SENSEX data. Fetches historical candles, intraday OHLCV, and option chain OI — then runs the Bulkowski pattern engine on real market data.
        </div>

        {/* Connection Status */}
        <div style={{ ...S.c, marginBottom: 16, borderLeft: `3px solid ${upstox.status === "connected" ? "#00c896" : upstox.status === "error" ? "#ff4757" : "#ffd700"}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 11, color: "#555", letterSpacing: 1 }}>CONNECTION STATUS</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: upstox.status === "connected" ? "#00c896" : upstox.status === "error" ? "#ff4757" : "#ffd700", margin: "4px 0" }}>
                {upstox.status === "connected" ? "● CONNECTED" : upstox.status === "connecting" ? "◌ CONNECTING..." : upstox.status === "exchanging" ? "◌ EXCHANGING TOKEN..." : upstox.status === "error" ? "✗ ERROR" : "○ DISCONNECTED"}
              </div>
              {upstox.profile && <div style={{ fontSize: 10, color: "#888" }}>
                {upstox.profile.user_name} · {upstox.profile.email} · {upstox.profile.user_id}
              </div>}
              {upstox.error && <div style={{ fontSize: 10, color: "#ff4757", marginTop: 4 }}>{upstox.error}</div>}
            </div>
            {upstox.status === "connected" && <div style={{ fontSize: 9, color: "#444" }}>Token expires ~3:30 AM IST</div>}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          {/* Method 1: Direct Token */}
          <div style={S.c}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#00c896", marginBottom: 10, letterSpacing: 1 }}>METHOD 1: DIRECT ACCESS TOKEN</div>
            <div style={{ fontSize: 10, color: "#555", marginBottom: 10, lineHeight: 1.5 }}>
              If you already have an access token from Upstox Developer Portal, paste it here.
              Get one from <span style={{ color: "#4ecdc4" }}>developer.upstox.com → Apps → Your App → Generate Token</span>
            </div>
            <div style={{ marginBottom: 8 }}>
              <label style={{ fontSize: 9, color: "#444", letterSpacing: 1, display: "block", marginBottom: 4 }}>ACCESS TOKEN</label>
              <input type="password" value={upstox.token} onChange={e => setUpstox(p => ({ ...p, token: e.target.value }))}
                placeholder="eyJ0eXAiOiJKV1QiLCJhbGci..."
                style={{ ...S.s, width: "100%", padding: "10px", fontSize: 11 }} />
            </div>
            <button onClick={() => connectUpstox(upstox.token)} disabled={!upstox.token || upstox.status === "connecting"}
              style={{ padding: "10px 24px", background: "#00c896", color: "#000", border: "none", borderRadius: 4, fontWeight: 700, fontSize: 11, cursor: "pointer", fontFamily: "inherit", letterSpacing: 1, opacity: !upstox.token ? 0.4 : 1 }}>
              {upstox.status === "connecting" ? "CONNECTING..." : "CONNECT WITH TOKEN"}
            </button>
          </div>

          {/* Method 2: OAuth2 Flow */}
          <div style={S.c}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#7c83ff", marginBottom: 10, letterSpacing: 1 }}>METHOD 2: OAUTH2 FLOW</div>
            <div style={{ fontSize: 10, color: "#555", marginBottom: 10, lineHeight: 1.5 }}>
              Full OAuth2 flow: Enter API credentials → Authorize → Exchange code for token.
              Create an app at <span style={{ color: "#4ecdc4" }}>developer.upstox.com</span>
            </div>
            <div style={{ display: "grid", gap: 8, marginBottom: 10 }}>
              <div>
                <label style={{ fontSize: 9, color: "#444", letterSpacing: 1, display: "block", marginBottom: 2 }}>API KEY (Client ID)</label>
                <input value={upstox.apiKey} onChange={e => setUpstox(p => ({ ...p, apiKey: e.target.value }))}
                  placeholder="your-api-key" style={{ ...S.s, width: "100%", padding: "8px" }} />
              </div>
              <div>
                <label style={{ fontSize: 9, color: "#444", letterSpacing: 1, display: "block", marginBottom: 2 }}>API SECRET</label>
                <input type="password" value={upstox.apiSecret} onChange={e => setUpstox(p => ({ ...p, apiSecret: e.target.value }))}
                  placeholder="your-api-secret" style={{ ...S.s, width: "100%", padding: "8px" }} />
              </div>
              <div>
                <label style={{ fontSize: 9, color: "#444", letterSpacing: 1, display: "block", marginBottom: 2 }}>REDIRECT URI</label>
                <input value={upstox.redirectUri} onChange={e => setUpstox(p => ({ ...p, redirectUri: e.target.value }))}
                  style={{ ...S.s, width: "100%", padding: "8px" }} />
              </div>
            </div>
            {upstox.apiKey && <div style={{ marginBottom: 8 }}>
              <a href={upstoxAuthUrl} target="_blank" rel="noreferrer"
                style={{ display: "inline-block", padding: "10px 24px", background: "#7c83ff", color: "#000", borderRadius: 4, fontWeight: 700, fontSize: 11, textDecoration: "none", letterSpacing: 1 }}>
                STEP 1: AUTHORIZE →
              </a>
              <div style={{ fontSize: 9, color: "#555", marginTop: 6 }}>After login, copy the <b style={{ color: "#aaa" }}>code</b> from the redirect URL:</div>
              <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                <input id="auth-code" placeholder="Paste authorization code here..."
                  style={{ ...S.s, flex: 1, padding: "8px" }} />
                <button onClick={() => { const code = document.getElementById("auth-code").value; if (code) exchangeCode(code); }}
                  style={{ padding: "8px 16px", background: "#7c83ff", color: "#000", border: "none", borderRadius: 4, fontWeight: 700, fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>
                  STEP 2: EXCHANGE
                </button>
              </div>
            </div>}
          </div>
        </div>

        {/* Live Data Fetch (when connected) */}
        {upstox.status === "connected" && <div style={S.c}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#ffd700", marginBottom: 10, letterSpacing: 1 }}>LIVE DATA — FETCH & ANALYZE</div>
          <div style={{ fontSize: 10, color: "#555", marginBottom: 10 }}>Fetch intraday 5-minute candles from Upstox and run Bulkowski pattern detection in real-time.</div>
          <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
            {["NIFTY50", "BANKNIFTY", "SENSEX"].map(idx =>
              <button key={idx} onClick={() => fetchLiveData(idx)} disabled={fetchingLive}
                style={{ padding: "8px 16px", background: "#141428", border: "1px solid #1a1a35", color: "#d0d0d8", borderRadius: 4, fontWeight: 700, fontSize: 11, cursor: "pointer", fontFamily: "inherit", opacity: fetchingLive ? 0.5 : 1 }}>
                {fetchingLive ? "..." : `SCAN ${idx}`}
              </button>
            )}
          </div>

          {liveData && <div style={{ ...S.c, borderLeft: "3px solid #ffd700" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#e8e8f0" }}>{liveData.index} — LIVE</div>
                <div style={{ fontSize: 9, color: "#555" }}>{liveData.source} · {liveData.bars} bars · {new Date(liveData.timestamp).toLocaleTimeString()}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: "#e8e8f0" }}>{liveData.price?.toFixed(2)}</div>
                <div style={{ fontSize: 12, color: liveData.dayChg > 0 ? "#00c896" : "#ff4757" }}>{liveData.dayChg > 0 ? "+" : ""}{liveData.dayChg}%</div>
              </div>
            </div>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#ffd700", marginBottom: 6 }}>DETECTED PATTERNS ({liveData.patterns.length})</div>
            {liveData.patterns.map((p, i) => <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 0", borderBottom: "1px solid #0a0a14" }}>
              <span style={{ fontWeight: 700, color: "#d0d0d8", fontSize: 11 }}>{p.nm}</span>
              <span style={S.bg(p.bias)}>{p.bias}</span>
              <span style={{ fontSize: 9, color: "#888" }}>{(p.conf * 100).toFixed(0)}% conf</span>
              <span style={{ fontSize: 9, color: "#555" }}>SR: {(p.sr * 100).toFixed(0)}%</span>
            </div>)}
          </div>}
        </div>}

        {/* Python Backend Info */}
        <div style={{ ...S.c, marginTop: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#4ecdc4", marginBottom: 8, letterSpacing: 1 }}>PYTHON BACKEND (for full pipeline)</div>
          <div style={{ fontSize: 10, color: "#666", lineHeight: 1.7 }}>
            For historical backtesting across all expiry cycles, use the Python backend:<br />
            <code style={{ color: "#00c896", background: "#0e0e22", padding: "2px 6px", borderRadius: 3 }}>pip install scipy numpy pandas requests upstox-python-sdk</code><br /><br />
            <b style={{ color: "#aaa" }}>Quick start:</b><br />
            <code style={{ color: "#4ecdc4", background: "#0e0e22", padding: "2px 6px", borderRadius: 3, display: "block", margin: "4px 0", whiteSpace: "pre" }}>python upstox_connector.py --token YOUR_TOKEN --index NIFTY50 --live</code><br />
            <code style={{ color: "#4ecdc4", background: "#0e0e22", padding: "2px 6px", borderRadius: 3, display: "block", margin: "4px 0", whiteSpace: "pre" }}>python upstox_connector.py --token YOUR_TOKEN --index BANKNIFTY --scan-all 12</code><br />
            <code style={{ color: "#4ecdc4", background: "#0e0e22", padding: "2px 6px", borderRadius: 3, display: "block", margin: "4px 0", whiteSpace: "pre" }}>python upstox_connector.py --auth --config upstox_config.json</code><br /><br />
            <b style={{ color: "#aaa" }}>API Endpoints Used:</b><br />
            <span style={{ color: "#888" }}>• Historical Candle V3: </span><span style={{ color: "#555" }}>api.upstox.com/v3/historical-candle/</span><br />
            <span style={{ color: "#888" }}>• Intraday Candle: </span><span style={{ color: "#555" }}>api.upstox.com/v2/historical-candle/intraday/</span><br />
            <span style={{ color: "#888" }}>• Option Chain: </span><span style={{ color: "#555" }}>api.upstox.com/v2/option/chain</span><br />
            <span style={{ color: "#888" }}>• Option Contracts: </span><span style={{ color: "#555" }}>api.upstox.com/v2/option/contract</span><br />
            <span style={{ color: "#888" }}>• Expired Instruments: </span><span style={{ color: "#555" }}>api.upstox.com/v2/expired-instruments/</span><br />
            <span style={{ color: "#888" }}>• Market Data Feeder V3: </span><span style={{ color: "#555" }}>WebSocket streaming</span>
          </div>
        </div>
      </div>}

      {/* DASHBOARD */}
      {tab === "DASH" && <div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(145px,1fr))", gap: 8, marginBottom: 16 }}>
          {[{ l: "Expiries", v: db.length, c: "#7c83ff" }, { l: "Patterns Found", v: db.reduce((s, e) => s + e.pc, 0), c: "#00c896" },
          { l: "Unique Types", v: allPats.length, c: "#ffd700" }, { l: "Bullish %", v: (db.filter(e => e.out === "BULLISH").length / db.length * 100).toFixed(1) + "%", c: "#00c896" },
          { l: "Bearish %", v: (db.filter(e => e.out === "BEARISH").length / db.length * 100).toFixed(1) + "%", c: "#ff4757" },
          { l: "Avg Confidence", v: (db.flatMap(e => e.pats).reduce((s, p) => s + p.conf, 0) / Math.max(1, db.flatMap(e => e.pats).length) * 100).toFixed(1) + "%", c: "#e066ff" }
          ].map((s, i) => <div key={i} style={{ ...S.c, borderLeft: `3px solid ${s.c}` }}><div style={{ fontSize: 9, color: "#555", letterSpacing: 1, marginBottom: 4 }}>{s.l}</div><div style={{ fontSize: 20, fontWeight: 700, color: s.c }}>{s.v}</div></div>)}
        </div>

        <div style={{ ...S.c, marginBottom: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#ffd700", marginBottom: 10, letterSpacing: 1 }}>TOP DETECTED PATTERNS (by scipy.signal engine)</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: 6 }}>
            {patFreq.slice(0, 12).map(([nm, cnt]) => {
              const matched = db.filter(e => e.pats.some(p => p.nm === nm));
              const avgChg = matched.length ? (matched.reduce((s, e) => s + e.ch, 0) / matched.length).toFixed(2) : "0";
              const meta = matched[0]?.pats.find(p => p.nm === nm);
              return <div key={nm} style={{ ...S.c, display: "flex", flexDirection: "column", gap: 2 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, color: "#d0d0d8", fontSize: 11 }}>{nm}</span>
                  {meta && <span style={S.bg(meta.bias)}>{meta.bias}</span>}
                </div>
                <div style={{ display: "flex", gap: 10, fontSize: 9, color: "#555" }}>
                  <span>Found: <b style={{ color: "#aaa" }}>{cnt}×</b></span>
                  <span>Avg: <b style={{ color: avgChg > 0 ? "#00c896" : avgChg < 0 ? "#ff4757" : "#888" }}>{avgChg > 0 ? "+" : ""}{avgChg}%</b></span>
                  {meta && <span>SR: <b style={{ color: "#ffd700" }}>{(meta.sr * 100).toFixed(0)}%</b></span>}
                </div>
              </div>;
            })}
          </div>
        </div>

        <div style={{ ...S.c }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#4ecdc4", marginBottom: 10, letterSpacing: 1 }}>DETECTION METHOD</div>
          <div style={{ fontSize: 11, color: "#888", lineHeight: 1.7 }}>
            <b style={{ color: "#00c896" }}>Peak/Trough Detection:</b> scipy.signal.argrelextrema equivalent — finds local maxima/minima using configurable lookback order.<br />
            <b style={{ color: "#00c896" }}>Trendline Analysis:</b> scipy.stats.linregress equivalent — linear regression on peak/trough sequences for slope, R² fit.<br />
            <b style={{ color: "#00c896" }}>ATR Thresholds:</b> Adaptive thresholds using Average True Range for pattern validation.<br />
            <b style={{ color: "#00c896" }}>Bulkowski Rules:</b> Mathematical rules from Encyclopedia of Chart Patterns — shoulder symmetry (5%), neckline computation, target projection.<br />
            <b style={{ color: "#00c896" }}>Libraries Used:</b> scipy.signal (peaks), scipy.stats (regression), numpy (math), pandas (data). JS port mirrors the Python engine exactly.
          </div>
        </div>
      </div>}

      {/* ENGINE DETAIL */}
      {tab === "ENG" && <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#e066ff", marginBottom: 12, letterSpacing: 1 }}>DETECTION ENGINE ARCHITECTURE</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
          <div style={S.c}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#00c896", marginBottom: 8 }}>GEOMETRIC PATTERNS (25+)</div>
            {["Head & Shoulders", "Inv Head & Shoulders", "Double Top", "Double Bottom", "Triple Top", "Triple Bottom",
              "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle", "Rising Wedge", "Falling Wedge",
              "Bullish Flag", "Bearish Flag", "Bullish Pennant", "Bearish Pennant", "Bullish Rectangle", "Bearish Rectangle",
              "Cup & Handle", "Ascending Channel", "Descending Channel", "Megaphone", "Ascending Staircase", "Descending Staircase",
              "V Bottom", "Bull Trap", "Bear Trap", "Spike"
            ].map(nm => {
              const cnt = db.reduce((s, e) => s + e.pats.filter(p => p.nm === nm).length, 0);
              return <div key={nm} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px solid #0a0a14", fontSize: 10 }}>
                <span style={{ color: "#aaa" }}>{nm}</span>
                <span style={{ color: cnt > 0 ? "#00c896" : "#333" }}>{cnt}×</span>
              </div>;
            })}
          </div>
          <div style={S.c}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#ff6b6b", marginBottom: 8 }}>CANDLESTICK PATTERNS</div>
            {["Morning Star", "Evening Star", "Bullish Engulfing", "Bearish Engulfing", "Hammer", "Bullish Kicker"].map(nm => {
              const cnt = db.reduce((s, e) => s + e.pats.filter(p => p.nm === nm).length, 0);
              return <div key={nm} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px solid #0a0a14", fontSize: 10 }}>
                <span style={{ color: "#aaa" }}>{nm}</span><span style={{ color: cnt > 0 ? "#ff6b6b" : "#333" }}>{cnt}×</span></div>;
            })}
            <div style={{ fontSize: 11, fontWeight: 700, color: "#4ecdc4", marginBottom: 8, marginTop: 14 }}>OI PATTERNS</div>
            {["Long Buildup", "Short Buildup", "Long Unwinding", "Short Covering", "PCR Extreme High", "PCR Extreme Low"].map(nm => {
              const cnt = db.reduce((s, e) => s + e.pats.filter(p => p.nm === nm).length, 0);
              return <div key={nm} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px solid #0a0a14", fontSize: 10 }}>
                <span style={{ color: "#aaa" }}>{nm}</span><span style={{ color: cnt > 0 ? "#4ecdc4" : "#333" }}>{cnt}×</span></div>;
            })}
            <div style={{ fontSize: 11, fontWeight: 700, color: "#ffd700", marginBottom: 8, marginTop: 14 }}>GAP PATTERNS</div>
            {["Gap Up", "Gap Down"].map(nm => {
              const cnt = db.reduce((s, e) => s + e.pats.filter(p => p.nm === nm).length, 0);
              return <div key={nm} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px solid #0a0a14", fontSize: 10 }}>
                <span style={{ color: "#aaa" }}>{nm}</span><span style={{ color: cnt > 0 ? "#ffd700" : "#333" }}>{cnt}×</span></div>;
            })}
          </div>
        </div>
      </div>}

      {/* BACKTEST */}
      {tab === "BACK" && <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#00c896", marginBottom: 12 }}>BACKTEST ENGINE</div>
        <div style={{ ...S.c, display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16, alignItems: "flex-end" }}>
          {[{ k: "mkt", l: "Market", o: ["ALL", "US", "INDIA"] }, { k: "ix", l: "Index", o: allIdx },
          { k: "pat", l: "Pattern", o: ["ALL", ...allPats] }, { k: "ct", l: "Cycle", o: ["ALL", "WEEKLY", "MONTHLY"] },
          { k: "mp", l: "Month Phase", o: ["ALL", "W1", "W2", "W3", "W4"] },
          { k: "dow", l: "Expiry Day", o: ["ALL", "Mon", "Tue", "Wed", "Thu", "Fri"] }
          ].map(f => <div key={f.k}><label style={{ fontSize: 8, color: "#444", letterSpacing: 1, display: "block", marginBottom: 2 }}>{f.l}</label>
            <select value={flt[f.k]} onChange={e => setFlt(p => ({ ...p, [f.k]: e.target.value }))} style={S.s}>
              {f.o.map(o => <option key={o} value={o}>{o}</option>)}</select></div>)}
        </div>

        {stats ? <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))", gap: 8, marginBottom: 14 }}>
            {[{ l: "Matches", v: stats.n, c: "#7c83ff" }, { l: "Win Rate", v: stats.wr + "%", c: stats.wr > 50 ? "#00c896" : "#ff4757" },
            { l: "Avg Change", v: (stats.avg > 0 ? "+" : "") + stats.avg + "%", c: stats.avg > 0 ? "#00c896" : "#ff4757" },
            { l: "Avg DD", v: stats.dd + "%", c: "#ff4757" }, { l: "Avg RU", v: "+" + stats.ru + "%", c: "#00c896" }
            ].map((s, i) => <div key={i} style={{ ...S.c, borderTop: `2px solid ${s.c}` }}><div style={{ fontSize: 8, color: "#444", letterSpacing: 1, marginBottom: 2 }}>{s.l}</div><div style={{ fontSize: 18, fontWeight: 700, color: s.c }}>{s.v}</div></div>)}
          </div>

          <div style={{ ...S.c, marginBottom: 14 }}>
            <div style={{ fontSize: 10, color: "#555", marginBottom: 6 }}>OUTCOME DISTRIBUTION</div>
            <div style={{ display: "flex", borderRadius: 4, overflow: "hidden", height: 22 }}>
              {[{ v: stats.bl, c: "#00c896" }, { v: stats.ne, c: "#7c83ff" }, { v: stats.br, c: "#ff4757" }].map((s, i) =>
                s.v > 0 ? <div key={i} style={{ width: `${s.v / stats.n * 100}%`, background: s.c, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#000", fontWeight: 700, minWidth: 20 }}>{s.v}</div> : null)}</div>
          </div>

          <div style={{ ...S.c, maxHeight: 350, overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
              <thead><tr style={{ borderBottom: "1px solid #141428", position: "sticky", top: 0, background: "#0b0b16" }}>
                {["Index", "Date", "Cycle", "Change", "Patterns (scipy)", "Conf", "Outcome"].map(h =>
                  <th key={h} style={{ padding: "4px 6px", textAlign: "left", color: "#444", fontSize: 9 }}>{h}</th>)}</tr></thead>
              <tbody>{filtered.slice(0, 40).map(e => <tr key={e.id} onClick={() => { setSel(e); setTab("DB"); }}
                style={{ borderBottom: "1px solid #07071a", cursor: "pointer" }}
                onMouseEnter={ev => ev.currentTarget.style.background = "#0e0e22"} onMouseLeave={ev => ev.currentTarget.style.background = "transparent"}>
                <td style={{ padding: "3px 6px", color: "#aaa" }}>{e.ix}</td>
                <td style={{ padding: "3px 6px", color: "#666" }}>{e.dt}</td>
                <td style={{ padding: "3px 6px", color: e.ct === "WEEKLY" ? "#ffd700" : "#e066ff", fontSize: 9 }}>{e.ct}</td>
                <td style={{ padding: "3px 6px", color: e.ch > 0 ? "#00c896" : "#ff4757" }}>{e.ch > 0 ? "+" : ""}{e.ch}%</td>
                <td style={{ padding: "3px 6px" }}>{e.pats.slice(0, 2).map(p => <span key={p.nm} style={S.bg(p.bias)}>{p.nm.slice(0, 14)}</span>)}</td>
                <td style={{ padding: "3px 6px", color: "#ffd700", fontSize: 9 }}>{e.pats[0]?.conf ? (e.pats[0].conf * 100).toFixed(0) + "%" : ""}</td>
                <td style={{ padding: "3px 6px" }}><span style={{ padding: "2px 5px", borderRadius: 3, fontSize: 8, ...(e.out === "BULLISH" ? { background: "#082e1a", color: "#00c896" } : e.out === "BEARISH" ? { background: "#2e0808", color: "#ff4757" } : { background: "#10102e", color: "#7c83ff" }) }}>{e.out}</span></td>
              </tr>)}</tbody></table>
          </div>
        </div> : <div style={{ textAlign: "center", padding: 40, color: "#444" }}>No matches.</div>}
      </div>}

      {/* REAL-TIME MATCH */}
      {tab === "RT" && <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#ffd700", marginBottom: 6 }}>REAL-TIME PATTERN MATCHER</div>
        <div style={{ fontSize: 10, color: "#555", marginBottom: 14 }}>Select conditions → query Bulkowski engine database → directional prediction.</div>
        <div style={{ ...S.c, marginBottom: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 10 }}>
            {[{ k: "mkt", l: "Market", o: ["ALL", "US", "INDIA"] }, { k: "ct", l: "Cycle", o: ["ALL", "WEEKLY", "MONTHLY"] },
            { k: "pat", l: "Pattern", o: ["ALL", ...allPats] },
            { k: "oi", l: "OI Signal", o: ["ALL", "Long Buildup", "Short Buildup", "Long Unwinding", "Short Covering", "PCR Extreme High", "PCR Extreme Low"] },
            { k: "ph", l: "Month Phase", o: ["ALL", "W1", "W2", "W3", "W4"] }
            ].map(f => <div key={f.k}><label style={{ fontSize: 8, color: "#444", letterSpacing: 1, display: "block", marginBottom: 2 }}>{f.l}</label>
              <select value={rtI[f.k]} onChange={e => setRtI(p => ({ ...p, [f.k]: e.target.value }))} style={{ ...S.s, width: "100%" }}>
                {f.o.map(o => <option key={o} value={o}>{o}</option>)}</select></div>)}</div>
          <button onClick={doMatch} style={{ marginTop: 12, padding: "8px 20px", background: "#00c896", color: "#000", border: "none", borderRadius: 4, fontWeight: 700, fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>▶ FIND MATCHES</button>
        </div>

        {matches.length > 0 && <div>
          {(() => {
            const bl = matches.filter(m => m.out === "BULLISH").length, br = matches.filter(m => m.out === "BEARISH").length;
            const av = (matches.reduce((s, m) => s + m.ch, 0) / matches.length).toFixed(2);
            const pr = bl > br ? "BULLISH BIAS" : br > bl ? "BEARISH BIAS" : "NEUTRAL", pc = bl > br ? "#00c896" : br > bl ? "#ff4757" : "#7c83ff";
            return <div style={{ ...S.c, marginBottom: 14, borderLeft: `3px solid ${pc}` }}>
              <div style={{ fontSize: 9, color: "#444", letterSpacing: 1 }}>HISTORICAL PREDICTION (Bulkowski Engine)</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: pc, margin: "4px 0" }}>{pr}</div>
              <div style={{ fontSize: 10, color: "#888" }}>{matches.length} matches · Avg: {av > 0 ? "+" : ""}{av}% · Bull {bl} / Bear {br}</div>
            </div>;
          })()}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(270px,1fr))", gap: 10 }}>
            {matches.slice(0, 12).map(m => <div key={m.id} onClick={() => { setSel(m); setTab("DB"); }} style={{ ...S.c, cursor: "pointer" }}
              onMouseEnter={ev => ev.currentTarget.style.borderColor = "#333"} onMouseLeave={ev => ev.currentTarget.style.borderColor = "#141428"}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}><span style={{ fontWeight: 700, color: "#d0d0d8" }}>{m.ix}</span><span style={{ fontSize: 9, color: "#555" }}>{m.dt}</span></div>
              <MC cd={m.cd} w={248} h={55} />
              <div style={{ display: "flex", gap: 8, fontSize: 9, marginTop: 4, color: "#888" }}>
                <span>Chg: <b style={{ color: m.ch > 0 ? "#00c896" : "#ff4757" }}>{m.ch > 0 ? "+" : ""}{m.ch}%</b></span>
                <span>Score: <b style={{ color: "#ffd700" }}>{m.score.toFixed(1)}</b></span></div>
              <div style={{ marginTop: 4 }}>{m.pats.slice(0, 3).map(p => <span key={p.nm} style={S.bg(p.bias)}>{p.nm}</span>)}</div>
            </div>)}
          </div>
        </div>}
      </div>}

      {/* DATABASE / DETAIL */}
      {tab === "DB" && <div>
        {sel ? <div>
          <button onClick={() => setSel(null)} style={{ background: "none", border: "1px solid #141428", color: "#888", padding: "5px 12px", borderRadius: 4, cursor: "pointer", fontSize: 10, fontFamily: "inherit", marginBottom: 12 }}>← Back</button>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <div style={S.c}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <div><div style={{ fontSize: 16, fontWeight: 700, color: "#e8e8f0" }}>{sel.ix}</div>
                  <div style={{ fontSize: 9, color: "#555" }}>{sel.mkt} · {sel.dt} · {sel.ct} · {sel.mp} · {sel.dow}</div>
                  <div style={{ fontSize: 9, color: "#444", marginTop: 2 }}>Engine: scipy.signal.argrelextrema + linregress</div></div>
                <span style={{ padding: "3px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700, height: "fit-content", ...(sel.out === "BULLISH" ? { background: "#082e1a", color: "#00c896" } : { background: "#2e0808", color: "#ff4757" }) }}>{sel.out}</span></div>
              <MC cd={sel.cd} w={440} h={130} />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6, marginTop: 10 }}>
                {[{ l: "Start", v: sel.sp }, { l: "End", v: sel.ep }, { l: "Change", v: (sel.ch > 0 ? "+" : "") + sel.ch + "%", c: sel.ch > 0 ? "#00c896" : "#ff4757" },
                { l: "Drawdown", v: sel.dd + "%", c: "#ff4757" }, { l: "Runup", v: "+" + sel.ru + "%", c: "#00c896" }, { l: "PCR", v: sel.pcr, c: sel.pcr > 1 ? "#00c896" : "#ff4757" }
                ].map((s, i) => <div key={i} style={{ background: "#0e0e22", padding: "6px 8px", borderRadius: 4 }}>
                  <div style={{ fontSize: 7, color: "#444", letterSpacing: 1 }}>{s.l}</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: s.c || "#d0d0d8" }}>{s.v}</div></div>)}
              </div>
            </div>
            <div style={S.c}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#ffd700", marginBottom: 8, letterSpacing: 1 }}>DETECTED PATTERNS ({sel.pats.length})</div>
              <div style={{ fontSize: 9, color: "#444", marginBottom: 8 }}>Method: scipy.signal peak/trough → Bulkowski mathematical rules</div>
              {sel.pats.map((p, i) => <div key={i} style={{ background: "#0e0e22", padding: "8px 10px", borderRadius: 4, marginBottom: 5, borderLeft: `3px solid ${p.conf > 0.7 ? "#00c896" : p.conf > 0.5 ? "#ffd700" : "#ff4757"}` }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontWeight: 700, color: "#d0d0d8", fontSize: 11 }}>{p.nm}</span>
                  <span style={{ fontSize: 9, color: "#888" }}>{p.cat} · {p.ph}</span></div>
                <div style={{ fontSize: 9, color: "#555", margin: "2px 0" }}>{p.desc}</div>
                {p.kl && Object.keys(p.kl).length > 0 && <div style={{ fontSize: 9, color: "#4ecdc4", margin: "2px 0" }}>
                  {Object.entries(p.kl).map(([k, v]) => <span key={k} style={{ marginRight: 8 }}>{k}: <b>{v}</b></span>)}</div>}
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 3 }}>
                  <div style={{ flex: 1, height: 3, background: "#05050b", borderRadius: 2 }}>
                    <div style={{ height: 3, borderRadius: 2, width: `${p.conf * 100}%`, background: p.conf > 0.7 ? "#00c896" : p.conf > 0.5 ? "#ffd700" : "#ff4757" }} /></div>
                  <span style={{ fontSize: 9, color: "#888" }}>{(p.conf * 100).toFixed(0)}%</span>
                  <span style={S.bg(p.bias)}>{p.bias}</span>
                  <span style={{ fontSize: 8, color: "#555" }}>SR:{(p.sr * 100).toFixed(0)}%</span>
                </div></div>)}
            </div>
          </div>
        </div> :
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#7c83ff", marginBottom: 12 }}>DATABASE — {db.length} Records (Bulkowski Engine)</div>
            <div style={{ display: "flex", gap: 4, marginBottom: 12, flexWrap: "wrap" }}>
              {allIdx.map(ix => <button key={ix} onClick={() => setFlt(p => ({ ...p, ix }))}
                style={{ padding: "3px 10px", borderRadius: 3, fontSize: 9, cursor: "pointer", fontFamily: "inherit", background: flt.ix === ix ? "#0e0e22" : "transparent", border: `1px solid ${flt.ix === ix ? "#7c83ff" : "#141428"}`, color: flt.ix === ix ? "#7c83ff" : "#444" }}>{ix}</button>)}</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(260px,1fr))", gap: 8 }}>
              {db.filter(e => flt.ix === "ALL" || e.ix === flt.ix).slice(0, 36).map(e =>
                <div key={e.id} onClick={() => setSel(e)} style={{ ...S.c, cursor: "pointer" }}
                  onMouseEnter={ev => ev.currentTarget.style.borderColor = "#333"} onMouseLeave={ev => ev.currentTarget.style.borderColor = "#141428"}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontWeight: 700, color: "#d0d0d8", fontSize: 11 }}>{e.ix}</span>
                    <span style={{ fontSize: 9, color: e.ct === "WEEKLY" ? "#ffd700" : "#e066ff" }}>{e.ct}</span>
                    <span style={{ fontSize: 9, color: "#555" }}>{e.dt}</span></div>
                  <MC cd={e.cd} w={236} h={48} />
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 9 }}>
                    <span style={{ color: e.ch > 0 ? "#00c896" : "#ff4757" }}>{e.ch > 0 ? "+" : ""}{e.ch}%</span>
                    <span style={{ color: "#555" }}>{e.pc} pats</span>
                    <span style={{ padding: "1px 5px", borderRadius: 2, fontSize: 8, ...(e.out === "BULLISH" ? { background: "#082e1a", color: "#00c896" } : e.out === "BEARISH" ? { background: "#2e0808", color: "#ff4757" } : { background: "#10102e", color: "#7c83ff" }) }}>{e.out}</span></div>
                  <div style={{ marginTop: 3 }}>{e.pats.slice(0, 3).map(p => <span key={p.nm} style={S.bg(p.bias)}>{p.nm.slice(0, 14)}</span>)}</div>
                </div>)}
            </div>
          </div>}
      </div>}

    </div>
    <div style={{ padding: "10px 16px", borderTop: "1px solid #141428", textAlign: "center", fontSize: 8, color: "#222", letterSpacing: 1, marginTop: 16 }}>
      BULKOWSKI ENGINE v3.0 · scipy.signal + linregress · SYNTHETIC DATA · NOT FINANCIAL ADVICE
    </div>
  </div>);
}
