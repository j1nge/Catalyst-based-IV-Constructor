import numpy as np
import pandas as pd

from dataProcessing import preprocess, get_next_earnings_date
from baselineIV import construct_baseline_iv

def compute_event_iv_for_rows(atm_with_tenor, params, model_func):

    a, b, c = params

    df = atm_with_tenor.copy()

    T_years = df["tenor_days"].values / 365.0
    sigma_actual = df["impliedVolatility"].values

    sigma_diff = model_func(T_years, a, b, c)

    # Annualized variance * T
    var_actual_T = (sigma_actual ** 2) * T_years
    var_diff_T   = (sigma_diff   ** 2) * T_years

    event_var = np.maximum(var_actual_T - var_diff_T, 0.0)

    event_iv = np.sqrt(event_var)

    df["iv_diff"]   = sigma_diff
    df["event_var"] = event_var
    df["event_iv"]  = event_iv

    positive_events = df["event_iv"][df["event_iv"] > 0]
    avg_event_iv = positive_events.mean() if not positive_events.empty else 0.0

    return df, avg_event_iv

def construct_event_iv(ticker, n_exp=4):
    atm_with_tenor, (params, model_func), earnings = construct_baseline_iv(
        ticker,
        n_exp=n_exp
    )

    # Step 2: compute event IV for each tenor
    event_table, avg_event_iv = compute_event_iv_for_rows(
        atm_with_tenor,
        params,
        model_func
    )

    return event_table, avg_event_iv, earnings


if __name__ == "__main__":
    ticker = "ANET"

    print("=" * 60)
    print(f"Event IV Decomposition for {ticker}")
    print("=" * 60)

    event_table, avg_event_iv, earnings = construct_event_iv(ticker, n_exp=4)

    # Show the table with key columns
    print("\nPer-tenor event IV (today):")
    print(
        event_table[
            ["expiration", "tenor_days", "strike",
             "impliedVolatility", "iv_diff", "event_iv"]
        ]
    )

    print("\nAverage event IV across tenors:")
    print(f"  {avg_event_iv:.4f} ({avg_event_iv * 100:.2f}%)")

    next_evt = get_next_earnings_date(earnings)
    print("\nNext earnings date:", next_evt)
