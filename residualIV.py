import numpy as np
import matplotlib.pyplot as plt

from baselineIV import construct_baseline_iv


def get_event_iv_table(ticker, n_exp=4):
    atm_with_tenor, (params, model_func), earnings = construct_baseline_iv(
        ticker, n_exp=n_exp
    )

    T_years = atm_with_tenor["tenor_days"].values / 365.0
    baseline_iv = model_func(T_years, *params)

    event_iv_df = atm_with_tenor.copy()
    event_iv_df["iv_diff"] = event_iv_df["impliedVolatility"] - baseline_iv
    event_iv_df["event_iv"] = np.maximum(event_iv_df["iv_diff"], 0.0)

    print("\nEvent IV per tenor:")
    print(event_iv_df[["expiration", "tenor_days", "impliedVolatility", "iv_diff", "event_iv"]])

    avg_event_iv = event_iv_df["event_iv"].mean()
    print(f"\nAverage event IV: {avg_event_iv:.4f} ({avg_event_iv*100:.2f}%)")

    return event_iv_df, (params, model_func)


def build_residual_surface(event_iv_df, decay_lambda=0.3, max_days=10, n_tenor_points=50):
    tenor_src = event_iv_df["tenor_days"].values.astype(float)
    event_src = event_iv_df["event_iv"].values

    # Finer tenor grid between min and max observed tenor
    tenor_grid = np.linspace(tenor_src.min(), tenor_src.max(), n_tenor_points)

    # 1D linear interpolation of event IV across tenor
    event_iv_grid = np.interp(tenor_grid, tenor_src, event_src)

    # Days after event
    days_grid = np.arange(0, max_days + 1)

    # Residual IV surface: event_iv(T) * exp(-lambda * days)
    residual_grid = np.zeros((len(days_grid), len(tenor_grid)))

    for i, d in enumerate(days_grid):
        residual_grid[i, :] = event_iv_grid * np.exp(-decay_lambda * d)

    return days_grid, tenor_grid, residual_grid, event_iv_grid


def plot_residual_surface(days_grid, tenor_grid, residual_grid):
    plt.figure(figsize=(9, 5))
    plt.imshow(
        residual_grid,
        aspect="auto",
        origin="lower",
        extent=[tenor_grid.min(), tenor_grid.max(), days_grid.min(), days_grid.max()],
    )
    plt.colorbar(label="Residual IV")
    plt.xlabel("Tenor (days)")
    plt.ylabel("Days after event")
    plt.title("Residual IV Surface (event-related IV)")
    plt.tight_layout()
    plt.show()


def plot_single_tenor_path(days_grid, tenor_grid, residual_grid, event_iv_grid,
                           params, model_func, target_tenor=5):
    idx = np.argmin(np.abs(tenor_grid - target_tenor))

    residual_path = residual_grid[:, idx]

    # Baseline diffusive IV for that tenor (constant over days)
    T_years = tenor_grid[idx] / 365.0
    baseline_iv = model_func(np.array([T_years]), *params)[0]

    total_iv_path = baseline_iv + residual_path

    plt.figure(figsize=(9, 5))
    plt.plot(days_grid, residual_path, "o-", label="Residual IV (event-related)")
    plt.plot(days_grid, total_iv_path, "o-", label="Total post-event IV")
    plt.axhline(baseline_iv, linestyle="--", label="Baseline IV")

    plt.xlabel("Days after event")
    plt.ylabel("Implied Volatility")
    plt.title(f"Residual and Total IV after Event (Tenor ~{tenor_grid[idx]:.1f} days)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticker = "ANET"

    event_iv_df, (params, model_func) = get_event_iv_table(ticker, n_exp=4)

    days_grid, tenor_grid, residual_grid, event_iv_grid = build_residual_surface(
        event_iv_df,
        decay_lambda=0.30,   # speed of decay per day
        max_days=10,
        n_tenor_points=50,
    )

    plot_residual_surface(days_grid, tenor_grid, residual_grid)

    plot_single_tenor_path(
        days_grid, tenor_grid, residual_grid, event_iv_grid,
        params, model_func,
        target_tenor=5,
    )
