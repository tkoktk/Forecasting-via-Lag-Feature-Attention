import numpy as np
import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_prices(tickers, start, end):
    all_data = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end, auto_adjust=False)

        if hist.empty:
            print(f"WARNING: No data returned for {ticker}")
            continue

        hist["ticker"] = ticker
        hist = hist.reset_index()
        all_data.append(hist)
        print(f"{ticker}: {len(hist)} rows downloaded")

    if not all_data:
        raise ValueError("No data downloaded for any ticker")

    prices_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows before cleaning: {len(prices_df)}")
    return prices_df


def format_prices(prices_df):
    prices_df = prices_df.copy()

    # Set Date as the index
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.tz_localize(None)
    prices_df = prices_df.set_index("Date")
    # Set to 'date'
    prices_df.index.name = "date"

    prices_df[REQUIRED_COLUMNS] = prices_df[REQUIRED_COLUMNS].replace(0, np.nan)

    prices_df = prices_df.sort_values(["ticker", "date"])

    missing = prices_df[REQUIRED_COLUMNS].isna().sum()
    print("\nMissing values per column after formatting:")
    print(missing)

    return prices_df


def get_ticker_data(prices_df, ticker):
    if prices_df.index.name == "date":
        df = prices_df.reset_index()
    else:
        df = prices_df.copy()

    ticker_data = df[df["ticker"] == ticker].copy()

    if ticker_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    ticker_data = ticker_data.drop(columns=["ticker"])

    if "date" in ticker_data.columns:
        ticker_data = ticker_data.set_index("date")

    ticker_data.index = pd.to_datetime(ticker_data.index)
    ticker_data = ticker_data.sort_index()

    print(
        f"{ticker}: {len(ticker_data)} rows, {ticker_data.index[0].date()} to {ticker_data.index[-1].date()}"
    )
    return ticker_data
