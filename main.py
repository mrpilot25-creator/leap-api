from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# =========================
# CREATE APP (MUST BE FIRST)
# =========================
app = FastAPI()

# =========================
# CORS (for Base44)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROOT ROUTE (VISIBLE IN BROWSER)
# =========================
@app.get("/")
def home():
    return {
        "status": "LEAP API LIVE",
        "endpoints": {
            "radar": "/radar",
            "scan": "/scan?ticker=AAPL"
        }
    }

# =========================
# BLACK-SCHOLES FUNCTIONS
# =========================
def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - price, 1e-5, 5)
    except:
        return np.nan


def delta(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# =========================
# CORE SCANNER
# =========================
def scan_stock(ticker):

    t = yf.Ticker(ticker)

    try:
        hist = t.history(period="1d")
        if hist.empty:
            return []
        S = hist["Close"].iloc[-1]
    except:
        return []

    expiries = t.options
    if not expiries:
        return []

    expiry = expiries[-1]

    T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365
    r = 0.04

    try:
        chain = t.option_chain(expiry).calls
    except:
        return []

    results = []

    for _, row in chain.iterrows():
        K = row.get("strike")
        price = row.get("lastPrice")

        if not price or price <= 0:
            continue

        iv = implied_vol(price, S, K, T, r)
        if np.isnan(iv):
            continue

        d = delta(S, K, T, r, iv)
        if np.isnan(d):
            continue

        score = d / (iv + 1e-6)

        signal = "STRONG BUY" if score > 2.5 else "BUY" if score > 1.8 else "WATCH"

        results.append({
            "ticker": ticker,
            "strike": float(K),
            "iv": float(iv),
            "delta": float(d),
            "score": float(score),
            "signal": signal
        })

    if not results:
        return []

    df = pd.DataFrame(results)

    df = df[(df["delta"] > 0.65) & (df["delta"] < 0.95)]

    df = df.sort_values(by="score", ascending=False)

    return df.to_dict(orient="records")

# =========================
# RADAR ENDPOINT
# =========================
TICKERS = ["AMZN", "NVDA", "INTC", "EXPE", "GOOG", "ABDE", "NFLX", "ORCL", "NOW", "BKNG" ]

@app.get("/radar")
def radar():

    all_results = []

    for t in TICKERS:
        try:
            data = scan_stock(t)
            all_results.extend(data)
        except:
            continue

    if not all_results:
        return {"best_trade": None, "top_trades": []}

    all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    return {
        "best_trade": all_results[0],
        "top_trades": all_results[:20]
    }

# =========================
# SINGLE STOCK SCAN
# =========================
@app.get("/scan")
def scan(ticker: str):

    data = scan_stock(ticker)

    return {
        "ticker": ticker,
        "best_trade": data[0] if data else None,
        "top_10": data[:10]
    }
