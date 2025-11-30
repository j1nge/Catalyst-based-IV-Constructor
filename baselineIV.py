import numpy as np
import pandas as pd
from datetime import date
from scipy.optimize import curve_fit

from dataProcessing import preprocess, get_next_earnings_date

def add_tenor_days(atm_df):
    atm = atm_df.copy()
    atm["expiration"] = pd.to_datetime(atm["expiration"]).dt.date
    today = date.today()
    atm["tenor_days"] = (atm["expiration"] - today).apply(lambda d: d.days)

    # Keep only future expirations
    atm = atm[atm["tenor_days"] > 0].copy()

    if atm.empty:
        raise ValueError("No positive-tenor expirations available for baseline IV fitting.")

    return atm


def fit_exponential_baseline(atm_df):
    atm = add_tenor_days(atm_df)

    T = atm["tenor_days"].values / 365.0
    IV = atm["impliedVolatility"].values

    def exp_decay(t, a, b, c):
        return a + b * np.exp(-c * t)

    a0 = IV.mean()
    b0 = IV.std() if IV.std() > 0 else 0.05
    c0 = 1.0
    p0 = [a0, b0, c0]

    params, _ = curve_fit(exp_decay, T, IV, p0=p0, maxfev=5000)
    a, b, c = params

    print("\n=== Diffusive Baseline IV Curve Parameters ===")
    print(f"a = {a:.4f}")
    print(f"b = {b:.4f}")
    print(f"c = {c:.4f}")
    print(f"Model: IV(T) = {a:.4f} + {b:.4f} * exp(-{c:.4f} * T)\n")

    return params, exp_decay, atm


def construct_baseline_iv(ticker, n_exp=4):
    print("=" * 60)
    print(f"Constructing Baseline Diffusive IV Curve for {ticker}")
    print("=" * 60)

    # Use existing pipeline to get ATM IV snapshot
    df_clean, atm_df, earnings, exps = preprocess(ticker, n_exp=n_exp)

    print("\nATM IV snapshot:")
    print(atm_df[["expiration", "strike", "impliedVolatility"]])

    params, model_func, atm_with_tenor = fit_exponential_baseline(atm_df)

    return atm_with_tenor, (params, model_func), earnings

if __name__ == "__main__":
    ticker = "ANET"

    atm_with_tenor, (params, model_func), earnings = construct_baseline_iv(ticker, n_exp=4)

    print("ATM with tenor (days):")
    print(atm_with_tenor[["expiration", "tenor_days", "strike", "impliedVolatility"]])

    print("\nBaseline diffusive IV at standard DTE:")
    for dte in [7, 14, 21, 30]:
        T_years = dte / 365.0
        iv_pred = model_func(T_years, *params)
        print(f"DTE = {dte:2d} days -> IV ≈ {iv_pred:.4f} ({iv_pred*100:.2f}%)")

    next_evt = get_next_earnings_date(earnings)
    print("\nNext earnings date:", next_evt)
