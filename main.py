from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

app = FastAPI()

# =========================
# CORS (IMPORTANT for Base44)
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# BLACK-SCHOLES
# =========================

def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - price, 1e-5, 5)
    except:
        return np.nan

# =========================
# DELTA
# =========================

def delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# =========================
# SCAN ENGINE
# =========================

def scan_stock(ticker):

    t = yf.Ticker(ticker)

    try:
        S = t.history(period="1d")["Close"].iloc[-1]
    except:
        return []

    expiries = t.options
    if len(expiries) == 0:
        return []

    expiry = expiries[-1]

    T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365
    r = 0.04

    chain = t.option_chain(expiry).calls

    results = []

    for _, row in chain.iterrows():

        K = row["strike"]
        price = row["lastPrice"]

        if price <= 0:
            continue

        iv = implied_vol(price, S, K, T, r)
        if np.isnan(iv):
            continue

        d = delta(S, K, T, r, iv)

        score = d / (iv + 1e-6)

        signal = (
            "STRONG BUY" if score > 2.5 else
            "BUY" if score > 1.8 else
            "WATCH"
        )

        results.append({
            "ticker": ticker,
            "strike": K,
            "iv": float(iv),
            "delta": float(d),
            "score": float(score),
            "signal": signal
        })

    df = pd.DataFrame(results)

    if df.empty:
        return []

    df = df[(df["delta"] > 0.65) & (df["delta"] < 0.95)]

    df = df.sort_values(by="score", ascending=False)

    return df.to_dict(orient="records")

# =========================
# API ENDPOINTS
# =========================

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","SPY","QQQ"]

@app.get("/radar")
def radar():

    all_results = []

    for t in TICKERS:
        try:
            all_results.extend(scan_stock(t))
        except:
            continue

    all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    return {
        "best_trade": all_results[0] if all_results else None,
        "top_trades": all_results[:20]
    }


@app.get("/scan")
def scan(ticker: str):

    data = scan_stock(ticker)

    return {
        "ticker": ticker,
        "best_trade": data[0] if data else None,
        "top_10": data[:10]
    }
