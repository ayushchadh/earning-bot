"""
Alpaca data layer — PRIMARY market-data source (proven working live).
======================================================================
Mirrors yahoo_data's interface so the bot can prefer Alpaca and fall back to
Yahoo only when Alpaca returns nothing. Pricing now comes from the SAME venue
the bot executes on, which fixes the fill mismatch and gives proper near-money
strikes (e.g. NFLX $73 instead of Yahoo's coarse $75).

Functions: get_spot, get_option_expirations, get_listed_strikes, get_atm_iv,
get_option_quote (bid,ask,mid), get_option_mid, get_realized_vol, clear_caches.
"""

import logging
from datetime import datetime, timedelta, date
import numpy as np

from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.data.requests import (
    OptionSnapshotRequest, StockSnapshotRequest, StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

log = logging.getLogger("iron_butterfly")

# Injected by the bot at import time (so we reuse its authenticated clients)
trading_client = None
stock_data_client = None
option_data_client = None

_contracts_cache = {}   # (symbol, exp) -> {strike: {C/P: occ}}
_snap_cache = {}        # occ -> snapshot
_spot_cache = {}
_exp_cache = {}


def init(tc, sdc, odc):
    global trading_client, stock_data_client, option_data_client
    trading_client, stock_data_client, option_data_client = tc, sdc, odc


def clear_caches():
    _contracts_cache.clear(); _snap_cache.clear()
    _spot_cache.clear(); _exp_cache.clear()


def get_spot(symbol: str):
    if symbol in _spot_cache:
        return _spot_cache[symbol]
    try:
        snap = stock_data_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=[symbol])).get(symbol)
        px = None
        if snap:
            tr = getattr(snap, "latest_trade", None)
            q  = getattr(snap, "latest_quote", None)
            if tr and getattr(tr, "price", None):
                px = float(tr.price)
            elif q and getattr(q, "bid_price", None) and getattr(q, "ask_price", None):
                px = (float(q.bid_price) + float(q.ask_price)) / 2
        if px and px > 0:
            _spot_cache[symbol] = round(px, 2)
            log.info(f"  Spot ${px:.2f}  (source: alpaca)")
            return _spot_cache[symbol]
    except Exception as e:
        log.warning(f"  alpaca spot failed for {symbol}: {e}")
    return None      # caller falls back to Yahoo


def _load_contracts(symbol, expiry):
    key = (symbol, expiry)
    if key in _contracts_cache:
        return _contracts_cache[key]
    out = {}
    try:
        exp_d = datetime.strptime(expiry, "%Y-%m-%d").date()
        contracts = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date_gte=exp_d, expiration_date_lte=exp_d,
            limit=1000,
        )).option_contracts
        for c in contracts:
            k = float(c.strike_price)
            typ = "C" if str(c.type).lower().endswith("call") else "P"
            out.setdefault(k, {})[typ] = c.symbol
    except Exception as e:
        log.warning(f"  alpaca contracts failed for {symbol} {expiry}: {e}")
    _contracts_cache[key] = out
    return out


def get_option_expirations(symbol: str):
    if symbol in _exp_cache:
        return _exp_cache[symbol]
    out = []
    try:
        today = date.today()
        contracts = trading_client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date_gte=today + timedelta(days=2),
            expiration_date_lte=today + timedelta(days=60),
            limit=1000,
        )).option_contracts
        out = sorted({str(c.expiration_date) for c in contracts})
    except Exception as e:
        log.warning(f"  alpaca expirations failed for {symbol}: {e}")
    _exp_cache[symbol] = out
    return out


def get_listed_strikes(symbol: str, expiry: str):
    return sorted(_load_contracts(symbol, expiry).keys())


def snap_to_strike(target, strikes):
    if not strikes:
        return round(target)
    return min(strikes, key=lambda s: abs(s - target))


def _occ_for(symbol, expiry, strike, opt_type):
    row = _load_contracts(symbol, expiry).get(float(strike))
    if not row:
        return None
    return row.get(opt_type.upper()[:1])


def _snapshot(occ):
    if not occ:
        return None
    if occ in _snap_cache:
        return _snap_cache[occ]
    try:
        s = option_data_client.get_option_snapshot(
            OptionSnapshotRequest(symbol_or_symbols=[occ])).get(occ)
    except Exception as e:
        log.warning(f"  alpaca snapshot failed for {occ}: {e}"); s = None
    _snap_cache[occ] = s
    return s


def _best_expiry(symbol):
    exps = get_option_expirations(symbol)
    if not exps:
        return None
    today = date.today()
    scored = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days) for e in exps]
    inwin = [t for t in scored if 25 <= t[1] <= 45]
    def is_third_fri(e):
        d = datetime.strptime(e, "%Y-%m-%d").date()
        return d.weekday() == 4 and 15 <= d.day <= 21
    if inwin:
        monthlies = [t for t in inwin if is_third_fri(t[0])]
        pool = monthlies or inwin
        return min(pool, key=lambda t: abs(t[1] - 35))[0]
    oks = [t for t in scored if t[1] >= 25]
    return min(oks, key=lambda t: abs(t[1] - 35))[0] if oks else None


def get_atm_iv(symbol: str):
    spot = get_spot(symbol)
    if spot is None:
        return None, None, None
    best_exp = _best_expiry(symbol)
    if not best_exp:
        return None, spot, None
    strikes = get_listed_strikes(symbol, best_exp)
    if not strikes:
        return None, spot, best_exp
    atm = snap_to_strike(spot, strikes)
    log.info(f"  ATM strike: spot=${spot:.2f} → ${atm} exp={best_exp} ({len(strikes)} strikes)")
    ivs = []
    for typ in ("C", "P"):
        occ = _occ_for(symbol, best_exp, atm, typ)
        snap = _snapshot(occ)
        iv = getattr(snap, "implied_volatility", None) if snap else None
        if iv and iv > 0:
            ivs.append(float(iv))
    if ivs:
        iv = sum(ivs) / len(ivs)
        log.info(f"  IV source: alpaca ({iv*100:.1f}%)")
        return iv, spot, best_exp
    return None, spot, best_exp


def _quote_from_snap(snap):
    if not snap:
        return 0.0, 0.0, 0.0
    q = getattr(snap, "latest_quote", None)
    bid = float(getattr(q, "bid_price", 0) or 0) if q else 0.0
    ask = float(getattr(q, "ask_price", 0) or 0) if q else 0.0
    if bid > 0 and ask > 0:
        mid = round((bid + ask) / 2, 2)
    else:
        tr = getattr(snap, "latest_trade", None)
        mid = round(float(tr.price), 2) if (tr and getattr(tr, "price", None)) else round(bid or ask or 0.0, 2)
    return bid, ask, mid


def get_option_quote(symbol, expiry, strike, opt_type):
    occ = _occ_for(symbol, expiry, strike, opt_type)
    return _quote_from_snap(_snapshot(occ))


def get_option_mid(symbol, expiry, strike, opt_type):
    return get_option_quote(symbol, expiry, strike, opt_type)[2]


def get_realized_vol(symbol: str, days: int = 30):
    try:
        end = datetime.now(); start = end - timedelta(days=90)
        bars = stock_data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
            start=start, end=end, feed="iex", adjustment=Adjustment.ALL,
        ))
        data = bars.data.get(symbol, []) if hasattr(bars, "data") else []
        closes = [float(b.close) for b in data if getattr(b, "close", 0)]
        if len(closes) < 2:
            return None
        rets = np.diff(np.log(np.array(closes)))[-days:]
        rets = np.clip(rets, -0.35, 0.35)
        return float(rets.std() * np.sqrt(252))
    except Exception as e:
        log.warning(f"  alpaca RV failed for {symbol}: {e}")
        return None
