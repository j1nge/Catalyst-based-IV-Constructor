import yfinance as yf
import pandas as pd
import numpy as np


def load_option_data(ticker, n_exp=4):
    
    tk = yf.Ticker(ticker)

    # Earnings
    try:
        earnings = tk.get_earnings_dates(limit=8)
    except:
        earnings = None

    # expirations
    exps = tk.options[:n_exp]

    chains = []
    for exp in exps:
        raw = tk.option_chain(exp)
        calls = raw.calls.copy()
        calls["expiration"] = exp
        chains.append(calls)

    df = pd.concat(chains).reset_index(drop=True)

    return df, earnings, exps


def clean_option_quotes(df):

    df = df.copy()

    # Remove no-volume strikes
    df = df[df["volume"] > 0]

    # Remove missing IV
    df = df[~df["impliedVolatility"].isna()]

    # Remove IV ≤ 0
    df = df[df["impliedVolatility"] > 0]

    # Spread filter: ask/bid not crazy
    df = df[df["bid"] > 0]
    df = df[df["ask"] > df["bid"]]
    df = df[(df["ask"] - df["bid"]) / df["bid"] < 0.5]  # 50% spread cap

    return df.reset_index(drop=True)


def extract_atm(df, spot):

    df = df.copy()
    df["atm_diff"] = (df["strike"] - spot).abs()

    atm = df.loc[df.groupby("expiration")["atm_diff"].idxmin()]

    return atm.reset_index(drop=True)


def preprocess(ticker, n_exp=4):
    
    print(f"Fetching data for {ticker}...")
    df, earnings, exps = load_option_data(ticker, n_exp)

    # fetch spot price (last close)
    spot = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    print(f"Spot price = {spot:.2f}")

    print("Cleaning quotes...")
    df_clean = clean_option_quotes(df)

    print("Extracting ATM IV...")
    atm = extract_atm(df_clean, spot)

    print("Done.")
    return df_clean, atm, earnings, exps


if __name__ == "__main__":
    ticker = "ANET"

    df_clean, atm_df, earnings, exps = preprocess(ticker)

    print("\nCleaned data sample:")
    print(df_clean.head())

    print("\nATM IV per expiration:")
    print(atm_df[["expiration", "strike", "impliedVolatility"]])

    print("\nEarnings Dates:")
    print(earnings)
