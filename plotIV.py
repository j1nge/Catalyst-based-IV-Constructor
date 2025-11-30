import numpy as np
import matplotlib.pyplot as plt

from baselineIV import construct_baseline_iv
from eventIV import compute_event_iv_for_rows


def plot_baseline_and_event_iv(ticker="ANET", n_exp=4):

    atm_with_tenor, (params, model_func), earnings = construct_baseline_iv(
        ticker,
        n_exp=n_exp,
    )
    a, b, c = params

    tenor_days = atm_with_tenor["tenor_days"].values
    T_years = tenor_days / 365.0
    iv_actual = atm_with_tenor["impliedVolatility"].values

    T_grid = np.linspace(T_years.min() * 0.8, T_years.max() * 1.2, 200)
    iv_baseline_grid = model_func(T_grid, a, b, c)

    event_table, avg_event_iv = compute_event_iv_for_rows(
        atm_with_tenor,
        params,
        model_func,
    )
    event_iv = event_table["event_iv"].values
    tenor_days_event = event_table["tenor_days"].values

    plt.figure(figsize=(7, 4))
    plt.title(f"{ticker} ATM IV vs Diffusive Baseline")
    plt.xlabel("Time to Expiry (days)")
    plt.ylabel("Implied Volatility")

    plt.scatter(tenor_days, iv_actual, label="Actual ATM IV")
    plt.plot(T_grid * 365.0, iv_baseline_grid, label="Baseline Diffusive IV")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.title(f"{ticker} Event IV by Tenor (Today)")
    plt.xlabel("Time to Expiry (days)")
    plt.ylabel("Event IV")

    plt.bar(tenor_days_event, event_iv, width=2)
    plt.grid(True, axis="y")
    plt.tight_layout()

    print("\nEvent IV per tenor:")
    print(event_table[["expiration", "tenor_days", "impliedVolatility",
                       "iv_diff", "event_iv"]])
    print(f"\nAverage event IV: {avg_event_iv:.4f} ({avg_event_iv * 100:.2f}%)")

    return atm_with_tenor, event_table, avg_event_iv


if __name__ == "__main__":
    ticker = "ANET"
    plot_baseline_and_event_iv(ticker, n_exp=4)
    plt.show()
