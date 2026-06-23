"""
One-Click Company Valuation Engine — FastAPI Backend
=====================================================
Exposes: GET /api/valuate?ticker=XYZ
Returns: JSON with price, shares, EPS, and three valuation estimates.

Deploy to Render or Fly.io (see DEPLOYMENT.md).
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import math

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Valuation Engine API",
    description="Programmatic stock valuation via yfinance",
    version="1.0.0",
)

# ── CORS — allow your Framer domain (and localhost for dev) ───────────────────
# Replace the Framer URL with your actual published domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # local dev
        "https://*.framer.app",           # Framer preview domains
        "https://yoursite.framer.website", # ← replace with your real domain
        "*",                              # wide-open while prototyping; tighten before launch
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_PE_RATIO        = 18.0   # Normalized P/E anchor (long-run S&P average)
DCF_GROWTH_RATE        = 0.08   # 8% annual FCF growth assumption (conservative)
DCF_TERMINAL_RATE      = 0.03   # 3% terminal/perpetuity growth (≈ long-run GDP)
DCF_DISCOUNT_RATE      = 0.10   # 10% WACC / required return
DCF_YEARS              = 5      # Projection horizon
GRAHAM_GROWTH_MULTIPLE = 8.5    # Graham's base P/E for zero-growth company
GRAHAM_BOND_YIELD_AAA  = 4.4    # Current AAA corporate bond yield (update periodically)

# ── Helper: safe division ─────────────────────────────────────────────────────

def safe_div(a, b, fallback=0.0):
    """Return a/b or fallback if b is None/zero."""
    try:
        if b is None or b == 0:
            return fallback
        return a / b
    except Exception:
        return fallback

# ── Valuation calculators ─────────────────────────────────────────────────────

def calc_multiples_price(trailing_eps: float) -> float | None:
    """
    Multiples Style Valuation
    ─────────────────────────
    Logic: If the market were to price this stock at a 'normal' 18x earnings
    multiple, what would the share price be?

    Formula: Target Price = EPS × Target P/E (18x)
    """
    if trailing_eps is None or trailing_eps <= 0:
        return None  # Negative/zero earnings make P/E meaningless
    return round(trailing_eps * TARGET_PE_RATIO, 2)


def calc_dcf_price(
    operating_cash_flow: float,
    capital_expenditures: float,
    shares_outstanding: float,
) -> float | None:
    """
    Intrinsic DCF Style Valuation (5-Year FCF Projection)
    ───────────────────────────────────────────────────────
    Logic: Project Free Cash Flow for 5 years at a fixed growth rate,
    discount each year back to present value, add a terminal value,
    then divide total intrinsic value by shares outstanding.

    FCF  = Operating Cash Flow − |CapEx|
    PV   = FCF_year / (1 + discount_rate)^year
    TV   = FCF_year5 × (1 + terminal_rate) / (discount_rate − terminal_rate)
    """
    if operating_cash_flow is None or shares_outstanding is None or shares_outstanding <= 0:
        return None

    # CapEx is usually reported as negative in yfinance; take absolute value
    capex = abs(capital_expenditures) if capital_expenditures else 0
    fcf = operating_cash_flow - capex

    if fcf <= 0:
        return None  # Negative FCF: DCF model breaks down

    total_pv = 0.0
    current_fcf = fcf

    for year in range(1, DCF_YEARS + 1):
        current_fcf *= (1 + DCF_GROWTH_RATE)
        pv = current_fcf / ((1 + DCF_DISCOUNT_RATE) ** year)
        total_pv += pv

    # Terminal value (Gordon Growth Model) discounted to today
    terminal_value = (current_fcf * (1 + DCF_TERMINAL_RATE)) / (DCF_DISCOUNT_RATE - DCF_TERMINAL_RATE)
    terminal_pv    = terminal_value / ((1 + DCF_DISCOUNT_RATE) ** DCF_YEARS)

    intrinsic_total = total_pv + terminal_pv
    return round(intrinsic_total / shares_outstanding, 2)


def calc_graham_price(trailing_eps: float, eps_growth_rate_pct: float) -> float | None:
    """
    Benjamin Graham Margin of Safety Formula
    ─────────────────────────────────────────
    The revised Graham Number from 'The Intelligent Investor':

    Intrinsic Value = EPS × (8.5 + 2g) × (4.4 / AAA_Bond_Yield)

    Where:
      8.5  = P/E for a no-growth company
      g    = expected annual EPS growth rate (next 7–10 years)
      4.4  = Graham's original AAA bond yield baseline
      AAA  = current AAA corporate bond yield (adjust GRAHAM_BOND_YIELD_AAA above)
    """
    if trailing_eps is None or trailing_eps <= 0:
        return None

    # Cap growth rate to avoid absurd outputs for hyper-growth stocks
    g = max(0, min(eps_growth_rate_pct, 25))

    intrinsic = trailing_eps * (GRAHAM_GROWTH_MULTIPLE + 2 * g) * (4.4 / GRAHAM_BOND_YIELD_AAA)
    return round(intrinsic, 2)

# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.get("/api/valuate")
async def valuate(ticker: str = Query(..., description="Stock ticker symbol, e.g. AAPL")):
    """
    Pull live data from Yahoo Finance and return three valuation estimates.

    Returns a JSON object with:
      - meta       : ticker, company name, sector
      - market     : current_price, shares_outstanding, trailing_eps
      - valuations : multiples_price, dcf_price, graham_price
      - inputs     : raw inputs used for each model (for frontend transparency)
      - errors     : list of any models that couldn't be calculated and why
    """

    ticker_symbol = ticker.strip().upper()
    errors = []

    # ── 1. Fetch yfinance data ────────────────────────────────────────────────
    try:
        stock = yf.Ticker(ticker_symbol)
        info  = stock.info or {}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"yfinance fetch failed: {str(e)}")

    # Validate the ticker returned real data
    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker_symbol}' not found or returned no data. Check the symbol."
        )

    # ── 2. Extract core market data ───────────────────────────────────────────

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    shares_outstanding = (
        info.get("sharesOutstanding")
        or info.get("impliedSharesOutstanding")
    )

    trailing_eps = info.get("trailingEps")

    # EPS growth: use analyst 5y estimate; fall back to trailing revenue growth
    eps_growth_pct = (
        info.get("earningsGrowth", 0) or
        info.get("revenueGrowth", 0) or
        0.10  # last-resort fallback: 10%
    ) * 100  # yfinance returns as decimal; convert to percent

    company_name = info.get("shortName") or info.get("longName") or ticker_symbol
    sector       = info.get("sector", "Unknown")

    # ── 3. Pull cash flow statement for DCF inputs ───────────────────────────
    operating_cash_flow = None
    capital_expenditures = None

    try:
        cf = stock.cashflow  # DataFrame, columns = fiscal periods
        if cf is not None and not cf.empty:
            # yfinance labels vary; try both camelCase and label-style keys
            def get_cf_row(df, *keys):
                for key in keys:
                    if key in df.index:
                        val = df.loc[key].iloc[0]  # most recent period
                        return float(val) if not math.isnan(float(val)) else None
                return None

            operating_cash_flow  = get_cf_row(cf,
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
                "operatingCashflow",
            )
            capital_expenditures = get_cf_row(cf,
                "Capital Expenditure",
                "Capital Expenditures",
                "capitalExpenditures",
            )
    except Exception as e:
        errors.append(f"Cash flow data unavailable ({str(e)}); DCF model skipped.")

    # ── 4. Run the three valuation models ────────────────────────────────────

    multiples_price = calc_multiples_price(trailing_eps)
    if multiples_price is None:
        errors.append("Multiples model skipped: EPS is negative or unavailable.")

    dcf_price = calc_dcf_price(operating_cash_flow, capital_expenditures, shares_outstanding)
    if dcf_price is None:
        errors.append("DCF model skipped: FCF is negative or cash flow data unavailable.")

    graham_price = calc_graham_price(trailing_eps, eps_growth_pct)
    if graham_price is None:
        errors.append("Graham model skipped: EPS is negative or unavailable.")

    # ── 5. Format shares outstanding for display ─────────────────────────────
    def fmt_shares(n):
        if n is None: return "N/A"
        if n >= 1e9:  return f"{n/1e9:.2f}B"
        if n >= 1e6:  return f"{n/1e6:.2f}M"
        return str(int(n))

    # ── 6. Return structured payload ─────────────────────────────────────────
    return JSONResponse(content={
        "meta": {
            "ticker":       ticker_symbol,
            "company_name": company_name,
            "sector":       sector,
        },
        "market": {
            "current_price":       round(float(current_price), 2) if current_price else None,
            "shares_outstanding":  fmt_shares(shares_outstanding),
            "trailing_eps":        round(float(trailing_eps), 2) if trailing_eps else None,
            "eps_growth_rate_pct": round(float(eps_growth_pct), 1),
        },
        "valuations": {
            "multiples_price": multiples_price,
            "dcf_price":       dcf_price,
            "graham_price":    graham_price,
        },
        "inputs": {
            "target_pe":             TARGET_PE_RATIO,
            "dcf_growth_rate_pct":   DCF_GROWTH_RATE * 100,
            "dcf_discount_rate_pct": DCF_DISCOUNT_RATE * 100,
            "dcf_years":             DCF_YEARS,
            "operating_cash_flow":   operating_cash_flow,
            "capital_expenditures":  capital_expenditures,
        },
        "errors": errors,
    })


# ── Health check — useful for Render/Fly.io uptime pings ─────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Local dev entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
