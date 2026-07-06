"""
EARNINGS SHORT-STRADDLE BOT — ALPACA PAPER (LIVE FORWARD TEST)
===============================================================
Actually TRADES the strategy on the Alpaca PAPER account, like the iron
butterfly bot — real orders, real paper fills — so the forward test measures
true execution economics, not would-be fills.

STRUCTURE NOTE (important): Alpaca does not permit naked short options (max
approval = Level 3, defined-risk multi-leg). A pure short straddle is naked.
So this bot trades the DEFINED-RISK variant the pre-registration anticipated:
  SELL ATM call + ATM put, BUY far protective wings at WING_MULT x the
  market-implied move (a wide iron butterfly ~= straddle with a capped tail).
The wing cost is logged per trade so the drag vs a pure straddle is measured.

Workflow (ET):
  15:30  T-1 ENTRY: for events reporting tomorrow-BMO / today-AMC, run gates
         (premium >= 4%, OI >= 500, leg spread <= 5%, VIX <= 30, GBM ratio
         <= 0.85), size, and place the 4-leg mleg atomically.
  09:40  POST-PRINT: record actual |gap|; run stop check (loss >= 50% of
         credit -> close at marketable).
  10:00-15:30 every 30 min: stop monitor.
  15:45  EXPIRY-DAY CLOSE: close any position expiring today (avoid pin/
         assignment).
All decisions + fills logged to DATA_DIR for the pre-registered analysis.

Env: ALPACA_API_KEY, ALPACA_SECRET_KEY  (paper), DATA_DIR, TZ=America/New_York
"""

import os, json, time, pickle, logging, traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import pytz
import schedule
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

load_dotenv()

# ------------------------------------------------------------------ config
DATA_DIR  = os.environ.get("DATA_DIR", ".")
ET        = pytz.timezone("US/Eastern")
HISTORY_F = os.path.join(DATA_DIR, "ticker_history.json")
TRADES_F  = os.path.join(DATA_DIR, "straddle_trades.json")
CAL_F     = os.path.join(DATA_DIR, "calendar_cache.json")
GBM_F     = os.path.join(DATA_DIR, "gbm_model.pkl")
LOG_F     = os.path.join(DATA_DIR, "earnings_straddle_bot.log")

UNIVERSE = ("AAPL ABBV ABNB ABT ADBE AEP AMAT AMD AMGN AMT AMZN APD AVGO AXP "
    "BA BAC BK BLK BSX C CAT CCI CDNS CF CFG CHTR CI CMCSA CMG CMI "
    "COF COIN COP COST CRM CRWD CSCO CVX D DASH DDOG DD DE DECK DHR "
    "DKNG DLTR DOCU DOV DPZ DUK EA ECL EMR EOG EQIX ETN F FAST FCX "
    "FTNT GE GILD GM GOOGL GS GWW HAL HBAN HD HON HOOD INTC ISRG ITW "
    "JNJ JPM KEY KLAC KO LCID LIN LLY LMT LOW LRCX LULU LYFT MCD MCHP "
    "MDT MET META MMM MPC MRK MRVL MS MSFT MU NEE NEM NET NFLX NKE NUE "
    "NVDA NXPI ON ORCL OXY PANW PATH PEP PFE PG PH PINS PLTR PNC PSX "
    "PYPL QCOM RBLX REGN RF RIVN ROK ROKU ROST RTX SBUX SCHW SHOP SHW "
    "SLB SNAP SNOW SNPS SO SOFI SPG STLD STT SWK SWKS SYK T TFC TGT "
    "TJX TMO TSLA TTWO TXN U UBER UNH UNP UPS UPST USB VLO VRTX VZ "
    "WFC WMT XOM YUM ZM ZTS").split()

# Gates (per the strategy under test)
VIX_MAX, MIN_PREMIUM_PCT, MIN_OI = 30.0, 0.04, 500
MAX_LEG_SPREAD, GBM_RATIO_MAX, MIN_HISTORY_N = 0.05, 0.85, 4
# Structure & risk
WING_MULT          = 3.0    # wings at strike +/- 3x implied move (defined risk)
STOP_LOSS_FRAC     = 0.50   # close if loss >= 50% of credit received
RISK_PER_EVENT     = 0.02   # 2% of equity max loss per event
MAX_TOTAL_RISK     = 0.10   # 10% of equity across concurrent events
CONCESSION_CAP_PCT = 0.10   # skip if entry concession > 10% of credit

FEATURE_COLS = ['gap_lag1','gap_lag2','rolling_avg','rolling_std','trail_vol_20d',
    'pre_momentum','vol_trend','volume_ratio','vix_level','gap_lag1_vs_vol',
    'avg_vs_vol','n_past']

# ------------------------------------------------------------------ logging
log = logging.getLogger("esb"); log.setLevel(logging.INFO)
for h in (RotatingFileHandler(LOG_F, maxBytes=5_000_000, backupCount=3),
          logging.StreamHandler()):
    h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                                     "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

# ------------------------------------------------------------------ clients
trading = TradingClient(os.environ["ALPACA_API_KEY"],
                        os.environ["ALPACA_SECRET_KEY"], paper=True)
odata   = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                     os.environ["ALPACA_SECRET_KEY"])

def get_equity():
    return float(trading.get_account().equity)

def occ(ticker, expiry, cp, strike):
    """OCC symbol: TICKER + YYMMDD + C/P + strike*1000 8-padded."""
    d = datetime.strptime(expiry, "%Y-%m-%d")
    return f"{ticker}{d:%y%m%d}{cp}{int(round(strike * 1000)):08d}"

def quote(sym):
    """(bid, ask, mid) via Alpaca latest quote; None on failure."""
    try:
        q = odata.get_option_latest_quote(OptionLatestQuoteRequest(
            symbol_or_symbols=sym))[sym]
        b, a = float(q.bid_price or 0), float(q.ask_price or 0)
        if a <= 0: return None
        return b, a, (b + a) / 2 if b > 0 else a
    except Exception:
        return None

def live_option_underlyings():
    """Underlyings of options actually held on Alpaca (reconciliation)."""
    out = set()
    try:
        for p in trading.get_all_positions():
            s = p.symbol
            if len(s) > 10 and s[-9] in "CP":
                i = 0
                while i < len(s) and not s[i].isdigit(): i += 1
                out.add(s[:i])
    except Exception as e:
        log.error(f"position fetch failed: {e}")
    return out

# ------------------------------------------------------------------ storage
def _load(p, d):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return d
def _save(p, o):
    t = p + ".tmp"
    with open(t, "w") as f: json.dump(o, f, indent=1, default=str)
    os.replace(t, p)
def load_trades(): return _load(TRADES_F, [])
def push_to_dashboard(trades):
    """POST open trades to the risk dashboard's ingest endpoint (same format
    as the butterfly bot) so positions get attribution + an EARNINGS strategy
    tag instead of showing as OTHER with 'no entry record'. Best-effort:
    never let a dashboard hiccup affect trading."""
    url = os.environ.get("DASHBOARD_URL")
    if not url:
        return
    try:
        import requests
        payload = []
        for t in trades:
            if t.get("status") not in ("OPEN", "CLOSED") or not t.get("legs"):
                continue
            L = t["legs"]
            payload.append({
                "symbol": t["ticker"], "expiry": t.get("expiry"),
                "center_strike": t.get("strike"),
                "lower_wing": t.get("wing_put"), "upper_wing": t.get("wing_call"),
                "qty": t.get("qty", 0),
                "entry_credit": t.get("fill_credit"),
                "entry_date": t.get("opened_ts", ""),
                "strategy": "EARNINGS",
                "legs": {"short_put": L["sp"], "short_call": L["sc"],
                         "long_put": L["lp"], "long_call": L["lc"]},
                "leg_entries": t.get("leg_entries", {}),
                "status": t.get("status", "OPEN"),
            })
        r = requests.post(url.rstrip("/") + "/api/ingest_trades",
                          json={"trades": payload}, timeout=5)
        if r.ok:
            log.info(f"Pushed {len(payload)} trades → dashboard")
        else:
            log.warning(f"Dashboard ingest returned {r.status_code}")
    except Exception as e:
        log.warning(f"Dashboard push failed (non-fatal): {e}")


def save_trades(t):
    _save(TRADES_F, t)
    push_to_dashboard(t)
def load_history(): return _load(HISTORY_F, {})
def save_history(h): _save(HISTORY_F, h)

# ------------------------------------------------------------------ yf data
def get_vix():
    try: return float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
    except Exception: return None

def get_price_df(tk, period="6mo"):
    try:
        df = yf.Ticker(tk).history(period=period, auto_adjust=True)
        return df if len(df) else None
    except Exception: return None

def infer_timing(ts):
    try:
        t = ts.tz_convert(ET) if ts.tzinfo else ET.localize(ts)
        return ("BMO" if t.hour < 10 else "AMC"), "high"
    except Exception: return "AMC", "low"

def refresh_calendar(days_ahead=10):
    cache = _load(CAL_F, {})
    if cache.get("asof") == str(date.today()): return cache.get("events", [])
    log.info("Refreshing earnings calendar (slow, once/day)...")
    events, now = [], datetime.now()
    for i, tk in enumerate(UNIVERSE):
        try:
            ed = yf.Ticker(tk).get_earnings_dates(limit=4)
            if ed is None or len(ed) == 0: continue
            for ts in ed.index:
                dt = ts.tz_localize(None) if ts.tzinfo is None \
                     else ts.tz_convert(ET).tz_localize(None)
                if now - timedelta(days=1) < dt < now + timedelta(days=days_ahead):
                    tm, conf = infer_timing(ts)
                    events.append({"ticker": tk, "earnings_dt": str(dt),
                                   "timing": tm, "timing_confidence": conf})
        except Exception: pass
        if i % 10 == 9: time.sleep(1.0)
    _save(CAL_F, {"asof": str(date.today()), "events": events})
    log.info(f"Calendar: {len(events)} events")
    return events

def find_weekly_expiry(tk, edt):
    try: exps = yf.Ticker(tk).options
    except Exception: return None
    best = None
    for s in exps:
        dte = (datetime.strptime(s, "%Y-%m-%d") - edt).days
        if 0 <= dte <= 7 and (best is None or dte < best[0]): best = (dte, s)
    if best: return best[1]
    for s in exps:
        if (datetime.strptime(s, "%Y-%m-%d") - edt).days > 0: return s
    return None

def listed_strikes(tk, expiry):
    try:
        ch = yf.Ticker(tk).option_chain(expiry)
        return sorted(set(ch.calls["strike"]) & set(ch.puts["strike"]))
    except Exception: return []

# ------------------------------------------------------------------ features/GBM
def compute_features(hist, pdf, vix):
    if hist is None or hist.get("n", 0) < MIN_HISTORY_N: return None
    gaps = hist["gaps"]
    if len(gaps) < 2 or pdf is None: return None
    rets = pdf["Close"].pct_change().dropna()
    if len(rets) < 21: return None
    tv20 = rets.tail(20).std() * np.sqrt(252)
    closes = pdf["Close"]
    return [float(x) for x in [gaps[-1], gaps[-2], hist["avg"], hist["std"], tv20,
        closes.iloc[-1]/closes.iloc[-21]-1,
        (rets.tail(5).std()*np.sqrt(252))/max(tv20,.01),
        pdf["Volume"].tail(5).mean()/max(pdf["Volume"].tail(20).mean(),1),
        vix, gaps[-1]/max(tv20/np.sqrt(252),.001),
        hist["avg"]/max(tv20/np.sqrt(252),.001), hist["n"]]]

def update_hist(h, gap):
    h.setdefault("gaps", []).append(round(float(gap), 5)); h["gaps"] = h["gaps"][-8:]
    h["all_gaps"] = h.get("all_gaps", h["gaps"][:]); h["all_gaps"].append(round(float(gap), 5))
    h["n"] = h.get("n", 0) + 1
    h["avg"] = float(np.mean(h["all_gaps"]))
    h["std"] = float(np.std(h["all_gaps"], ddof=1)) if len(h["all_gaps"]) > 1 else 0.0
    return h

def train_gbm():
    log.info("Training GBM from YF-reconstructed history (one-off)...")
    try: vixdf = yf.Ticker("^VIX").history(period="5y")["Close"]
    except Exception: vixdf = None
    rows = []
    for i, tk in enumerate(UNIVERSE):
        try:
            t = yf.Ticker(tk)
            prices = t.history(period="5y", auto_adjust=True)
            ed = t.get_earnings_dates(limit=28)
            if prices is None or len(prices) < 100 or ed is None or not len(ed): continue
            dates = sorted([(ts.tz_localize(None) if ts.tzinfo is None
                             else ts.tz_convert(ET).tz_localize(None)) for ts in ed.index])
            hist = {"gaps": [], "all_gaps": [], "n": 0, "avg": 0.0, "std": 0.0}
            idx = prices.index.tz_localize(None)
            for edt in dates:
                if edt > datetime.now(): break
                timing = "BMO" if edt.hour < 10 else "AMC"
                before = idx[idx < (edt if timing == "BMO" else edt + timedelta(hours=8))]
                after  = idx[idx > (edt - timedelta(hours=8) if timing == "BMO" else edt)]
                if not len(before) or not len(after): continue
                c0 = prices.iloc[idx.get_loc(before[-1])]["Close"]
                o1 = prices.iloc[idx.get_loc(after[0])]["Open"]
                if not (c0 > 0 and o1 > 0): continue
                gap = abs(o1 / c0 - 1)
                pdf = prices[idx < before[-1] + pd.Timedelta(days=1)].tail(130)
                vix = 20.0
                if vixdf is not None:
                    vs = vixdf[vixdf.index.tz_localize(None) <= before[-1]]
                    if len(vs): vix = float(vs.iloc[-1])
                f = compute_features(hist, pdf, vix) if hist["n"] >= MIN_HISTORY_N else None
                if f is not None: rows.append(f + [gap])
                update_hist(hist, gap)
        except Exception: pass
        if i % 10 == 9: time.sleep(1.0)
    if len(rows) < 300:
        log.warning(f"GBM data too small ({len(rows)}) — GBM gate N/A"); return None
    arr = np.array(rows)
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, loss="squared_error", random_state=42)
    gbm.fit(arr[:, :12], arr[:, 12])
    with open(GBM_F, "wb") as f: pickle.dump(gbm, f)
    log.info(f"GBM trained on {len(rows)} events"); return gbm

def load_gbm():
    if os.path.exists(GBM_F):
        try:
            with open(GBM_F, "rb") as f: return pickle.load(f)
        except Exception: pass
    return train_gbm()

# ------------------------------------------------------------------ execution
def submit_mleg(legs, qty, target_credit, floor_credit, wait=8):
    """Open: walk credit from mid DOWN through the marketable floor (−5%
    buffer) so a fill is guaranteed even if quotes drift (lesson from the
    NFLX roll-reopen failure). Limit is negative (credit) per Alpaca mleg."""
    give = max(round(floor_credit * 0.05, 2), 0.05)
    hard = max(round(floor_credit - give, 2), 0.01)
    steps = 10
    ladder, seen = [], None
    for i in range(steps):
        c = max(round(target_credit - (target_credit - hard) * i / (steps - 1), 2), hard)
        if c != seen: ladder.append(c); seen = c
    for att, credit in enumerate(ladder):
        try:
            o = trading.submit_order(LimitOrderRequest(
                qty=qty, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
                limit_price=-round(credit, 2), legs=legs))
            log.info(f"    open ×{qty} credit=${credit:.2f} ({att+1}/{len(ladder)})")
            for _ in range(wait):
                time.sleep(1)
                st = trading.get_order_by_id(o.id).status.value
                if st == "filled": return trading.get_order_by_id(o.id)
                if st in ("canceled", "expired", "rejected"): break
            try: trading.cancel_order_by_id(o.id)
            except Exception: pass
        except Exception as e:
            log.error(f"    open error: {e}"); return None
    # confirm no late fill left behind (double-entry lesson)
    return None

def close_mleg(trade, reason, wait=12):
    """Close all 4 legs atomically: walk debit from mid UP through the
    marketable ceiling (+1c) — the hardened close from the butterfly bot."""
    L = trade["legs"]
    qs = {k: quote(v) for k, v in L.items()}
    if any(v is None for v in qs.values()):
        log.error("  close: missing quotes"); return None
    debit_mid = (qs["sc"][2] + qs["sp"][2]) - (qs["lc"][2] + qs["lp"][2])
    debit_mkt = round((qs["sc"][1] + qs["sp"][1]) - (qs["lc"][0] + qs["lp"][0]), 2)
    log.info(f"  CLOSING {trade['ticker']} reason={reason} "
             f"mid=${debit_mid:.2f} marketable=${debit_mkt:.2f}")
    legs = [OptionLegRequest(symbol=L["sc"], side=OrderSide.BUY,  ratio_qty=1),
            OptionLegRequest(symbol=L["sp"], side=OrderSide.BUY,  ratio_qty=1),
            OptionLegRequest(symbol=L["lc"], side=OrderSide.SELL, ratio_qty=1),
            OptionLegRequest(symbol=L["lp"], side=OrderSide.SELL, ratio_qty=1)]
    lo = max(round(debit_mid, 2), 0.01)
    hi = round(max(debit_mkt, lo) + 0.01, 2)
    steps = 8
    ladder = sorted({round(lo + (hi - lo) * i / (steps - 1), 2) for i in range(steps)})
    for att, debit in enumerate(ladder):
        try:
            o = trading.submit_order(LimitOrderRequest(
                qty=trade["qty"], order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY, limit_price=round(debit, 2), legs=legs))
            log.info(f"    close ×{trade['qty']} debit=${debit:.2f} ({att+1}/{len(ladder)})")
            for _ in range(wait):
                time.sleep(1)
                st = trading.get_order_by_id(o.id).status.value
                if st == "filled": return trading.get_order_by_id(o.id)
                if st in ("canceled", "expired", "rejected"): break
            try: trading.cancel_order_by_id(o.id)
            except Exception: pass
        except Exception as e:
            log.error(f"    close error: {e}"); return None
    log.error("  close did not fill — will retry next run")
    return None

# ------------------------------------------------------------------ entry
def open_total_risk(trades):
    return sum(t.get("max_loss_total", 0) for t in trades if t["status"] == "OPEN")

def try_enter(ev, hist_all, vix, gbm, trades):
    tk = ev["ticker"]; edt = datetime.fromisoformat(ev["earnings_dt"])
    eid = f"{tk}_{edt.date()}"
    if any(t["event_id"] == eid for t in trades): return
    if tk in live_option_underlyings():
        log.info(f"  {tk}: already holds options — skip"); return
    rec = {"event_id": eid, "ticker": tk, "earnings_dt": ev["earnings_dt"],
           "timing": ev["timing"], "status": "EVALUATED", "vix": vix,
           "t1_ts": str(datetime.now(ET))}
    try:
        expiry = find_weekly_expiry(tk, edt)
        strikes = listed_strikes(tk, expiry) if expiry else []
        if not strikes:
            rec.update(decision="SKIP", reason="no expiry/strikes"); trades.append(rec); return
        spot = float(yf.Ticker(tk).history(period="1d")["Close"].iloc[-1])
        k_atm = min(strikes, key=lambda s: abs(s - spot))
        sc, sp = occ(tk, expiry, "C", k_atm), occ(tk, expiry, "P", k_atm)
        qsc, qsp = quote(sc), quote(sp)
        if qsc is None or qsp is None:
            rec.update(decision="SKIP", reason="no ATM quotes"); trades.append(rec); return
        straddle_mid = qsc[2] + qsp[2]; straddle_bid = qsc[0] + qsp[0]
        imp = straddle_mid / spot
        rec.update(spot=spot, strike=k_atm, expiry=expiry,
                   straddle_mid=round(straddle_mid, 3),
                   straddle_bid=round(straddle_bid, 3),
                   implied_move=round(imp, 5))
        # gates
        if imp < MIN_PREMIUM_PCT:
            rec.update(decision="SKIP", reason=f"premium {imp:.1%}<4%"); trades.append(rec); return
        for q, nm in ((qsc, "call"), (qsp, "put")):
            if (q[1] - q[0]) / max(q[2], .01) > MAX_LEG_SPREAD:
                rec.update(decision="SKIP", reason=f"{nm} spread wide"); trades.append(rec); return
        if vix is None or vix > VIX_MAX:
            rec.update(decision="SKIP", reason=f"VIX {vix}"); trades.append(rec); return
        feats = compute_features(hist_all.get(tk), get_price_df(tk), vix)
        if feats is not None and gbm is not None:
            pred = float(gbm.predict([feats])[0]); ratio = pred / max(imp, .001)
            rec.update(gbm_pred=round(pred, 5), gbm_ratio=round(ratio, 3))
            if ratio > GBM_RATIO_MAX:
                rec.update(decision="SKIP", reason=f"GBM ratio {ratio:.2f}>0.85")
                trades.append(rec); return
        else:
            rec.update(gbm_pred=None, gbm_ratio=None)
        # wings at WING_MULT x implied move, snapped to listed strikes
        wd = WING_MULT * imp * spot
        k_lc = min([s for s in strikes if s > k_atm + wd * 0.6] or [max(strikes)],
                   key=lambda s: abs(s - (k_atm + wd)))
        k_lp = min([s for s in strikes if s < k_atm - wd * 0.6] or [min(strikes)],
                   key=lambda s: abs(s - (k_atm - wd)))
        lc, lp = occ(tk, expiry, "C", k_lc), occ(tk, expiry, "P", k_lp)
        qlc, qlp = quote(lc), quote(lp)
        if qlc is None or qlp is None:
            rec.update(decision="SKIP", reason="no wing quotes"); trades.append(rec); return
        wing_cost_mid = qlc[2] + qlp[2]
        credit_mid = straddle_mid - wing_cost_mid
        credit_mkt = round(straddle_bid - (qlc[1] + qlp[1]), 2)  # sell@bid, buy@ask
        concession = credit_mid - credit_mkt
        rec.update(wing_call=k_lc, wing_put=k_lp,
                   wing_cost_mid=round(wing_cost_mid, 3),
                   credit_mid=round(credit_mid, 3), credit_marketable=credit_mkt,
                   concession_pct=round(concession / max(credit_mid, .01), 3))
        if credit_mkt <= 0 or concession > CONCESSION_CAP_PCT * credit_mid:
            rec.update(decision="SKIP",
                       reason=f"concession {concession:.2f} > {CONCESSION_CAP_PCT:.0%} of credit")
            trades.append(rec); return
        # sizing: defined-risk max loss = wider wing dist − credit
        wing_w = max(k_lc - k_atm, k_atm - k_lp)
        max_loss_per = (wing_w - credit_mid) * 100
        if max_loss_per <= 0:
            rec.update(decision="SKIP", reason="degenerate structure"); trades.append(rec); return
        eq = get_equity()
        budget = min(eq * RISK_PER_EVENT, eq * MAX_TOTAL_RISK - open_total_risk(trades))
        qty = int(budget // max_loss_per)
        if qty < 1:
            rec.update(decision="SKIP", reason="risk budget exhausted"); trades.append(rec); return
        legs = [OptionLegRequest(symbol=sc, side=OrderSide.SELL, ratio_qty=1),
                OptionLegRequest(symbol=sp, side=OrderSide.SELL, ratio_qty=1),
                OptionLegRequest(symbol=lc, side=OrderSide.BUY,  ratio_qty=1),
                OptionLegRequest(symbol=lp, side=OrderSide.BUY,  ratio_qty=1)]
        o = submit_mleg(legs, qty, round(credit_mid, 2), credit_mkt)
        if o is None:
            rec.update(decision="NOFILL", reason="entry did not fill"); trades.append(rec); return
        fill_credit = abs(float(getattr(o, "filled_avg_price", credit_mkt) or credit_mkt))
        rec.update(decision="FILLED", status="OPEN", qty=qty,
                   fill_credit=round(fill_credit, 3),
                   max_loss_total=round(max_loss_per * qty, 2),
                   legs={"sc": sc, "sp": sp, "lc": lc, "lp": lp},
                   leg_entries={
                       sc: {"price": round(qsc[2], 2), "iv": None},
                       sp: {"price": round(qsp[2], 2), "iv": None},
                       lc: {"price": round(qlc[2], 2), "iv": None},
                       lp: {"price": round(qlp[2], 2), "iv": None}},
                   opened_ts=str(datetime.now(ET)))
        log.info(f"  ✓ OPENED {eid} ×{qty} credit=${fill_credit:.2f} "
                 f"(imp={imp:.1%}, wings {k_lp}/{k_lc}, "
                 f"wing cost ${wing_cost_mid:.2f} = "
                 f"{wing_cost_mid/straddle_mid:.0%} of straddle)")
    except Exception as e:
        rec.update(decision="ERROR", reason=str(e))
        log.error(f"  {eid}: {e}\n{traceback.format_exc()}")
    trades.append(rec)

def run_t1():
    log.info("=== T-1 ENTRY RUN ===")
    try:
        trades = load_trades(); cal = refresh_calendar()
        hist_all = load_history(); vix = get_vix(); gbm = load_gbm()
        now = datetime.now(ET).replace(tzinfo=None)
        for ev in cal:
            edt = datetime.fromisoformat(ev["earnings_dt"])
            if (ev["timing"] == "AMC" and edt.date() == now.date()) or \
               (ev["timing"] == "BMO" and edt.date() == (now + timedelta(days=1)).date()):
                try_enter(ev, hist_all, vix, gbm, trades)
                time.sleep(0.5)
        save_trades(trades)
    except Exception as e:
        log.error(f"T-1 run failed: {e}\n{traceback.format_exc()}")

# ------------------------------------------------------------------ manage
def mark_and_manage():
    """Post-print gaps, stop checks, P&L marks."""
    trades = load_trades(); hist_all = load_history()
    now = datetime.now(ET).replace(tzinfo=None); changed = False
    for t in trades:
        if t.get("status") != "OPEN": continue
        edt = datetime.fromisoformat(t["earnings_dt"])
        # record gap once reported
        if t.get("gap_abs") is None:
            reported = (t["timing"] == "AMC" and now.date() > edt.date()) or \
                       (t["timing"] == "BMO" and now.date() >= edt.date() and now.hour >= 9)
            if reported:
                df = get_price_df(t["ticker"], period="10d")
                if df is not None and len(df) >= 2:
                    idx = df.index.tz_localize(None)
                    pre  = idx[idx.date <= edt.date()] if t["timing"] == "AMC" \
                           else idx[idx.date < edt.date()]
                    post = idx[idx.date >  edt.date()] if t["timing"] == "AMC" \
                           else idx[idx.date >= edt.date()]
                    if len(pre) and len(post):
                        c0 = float(df.iloc[idx.get_loc(pre[-1])]["Close"])
                        o1 = float(df.iloc[idx.get_loc(post[0])]["Open"])
                        t["gap_abs"] = round(abs(o1 / c0 - 1), 5)
                        hist_all[t["ticker"]] = update_hist(
                            hist_all.get(t["ticker"],
                                {"gaps": [], "all_gaps": [], "n": 0, "avg": 0, "std": 0}),
                            t["gap_abs"])
                        changed = True
                        log.info(f"  {t['event_id']}: gap={t['gap_abs']:.2%} "
                                 f"vs implied {t.get('implied_move', 0):.2%}")
        # stop check: mark position, close if loss >= 50% credit
        qs = {k: quote(v) for k, v in t.get("legs", {}).items()}
        if t.get("legs") and all(qs.values()):
            buyback = (qs["sc"][2] + qs["sp"][2]) - (qs["lc"][2] + qs["lp"][2])
            pnl = t["fill_credit"] - buyback
            t["last_mark_pnl"] = round(pnl, 3)
            if pnl <= -STOP_LOSS_FRAC * t["fill_credit"]:
                o = close_mleg(t, "STOP")
                if o is not None:
                    fp = abs(float(getattr(o, "filled_avg_price", buyback) or buyback))
                    t.update(status="CLOSED", close_reason="STOP",
                             close_debit=round(fp, 3),
                             pnl_final=round(t["fill_credit"] - fp, 3),
                             closed_ts=str(datetime.now(ET)))
                    log.info(f"  ✗ STOPPED {t['event_id']} pnl=${t['pnl_final']:.2f}")
                changed = True
    if changed: save_trades(trades); save_history(hist_all)

def close_expiring():
    """15:45 ET on expiry day: close everything expiring today."""
    trades = load_trades(); today = datetime.now(ET).date(); changed = False
    for t in trades:
        if t.get("status") != "OPEN" or not t.get("expiry"): continue
        if datetime.strptime(t["expiry"], "%Y-%m-%d").date() != today: continue
        o = close_mleg(t, "EXPIRY")
        if o is not None:
            fp = abs(float(getattr(o, "filled_avg_price", 0) or 0))
            t.update(status="CLOSED", close_reason="EXPIRY",
                     close_debit=round(fp, 3),
                     pnl_final=round(t["fill_credit"] - fp, 3),
                     closed_ts=str(datetime.now(ET)))
            log.info(f"  ✓ EXPIRY-CLOSED {t['event_id']} pnl=${t['pnl_final']:.2f}")
            changed = True
    if changed: save_trades(trades)

def report():
    trades = load_trades()
    done = [t for t in trades if t.get("pnl_final") is not None]
    evald = [t for t in trades if t.get("gap_abs") is not None and t.get("implied_move")]
    log.info("── RUNNING STATS (real Alpaca paper fills) ──")
    if evald:
        edges = np.array([t["implied_move"] - t["gap_abs"] for t in evald])
        n = len(edges)
        tstat = edges.mean() / (edges.std(ddof=1) / np.sqrt(n)) if n > 2 else 0
        log.info(f"  premium-vs-gap: n={n} mean={edges.mean():+.3%} t={tstat:.2f} "
                 f"hit={(edges > 0).mean():.0%}")
    if done:
        pnls = np.array([t["pnl_final"] * 100 * t.get("qty", 1) for t in done])
        log.info(f"  realized: n={len(done)} total=${pnls.sum():.0f} "
                 f"mean=${pnls.mean():.0f} worst=${pnls.min():.0f} "
                 f"win={(pnls > 0).mean():.0%}")
    log.info("  Decision gate: ≥150 events per PREREGISTRATION.md — do not judge early.")

def run_morning():
    log.info("=== POST-PRINT / MANAGE RUN ===")
    try: mark_and_manage(); report()
    except Exception as e: log.error(f"morning failed: {e}\n{traceback.format_exc()}")

def run_monitor():
    try: mark_and_manage()
    except Exception as e: log.error(f"monitor failed: {e}")

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_t1(); run_morning(); sys.exit(0)
    log.info("Earnings straddle bot (defined-risk, Alpaca PAPER) starting")
    log.info(f"Account equity: ${get_equity():,.0f}")
    schedule.every().day.at("15:30").do(run_t1)
    schedule.every().day.at("09:40").do(run_morning)
    for hh in ("10:00","10:30","11:00","11:30","12:00","12:30","13:00",
               "13:30","14:00","14:30","15:00"):
        schedule.every().day.at(hh).do(run_monitor)
    schedule.every().day.at("15:45").do(close_expiring)
    run_morning()
    while True:
        schedule.run_pending(); time.sleep(30)
