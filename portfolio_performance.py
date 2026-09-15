import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

START_DATE = "2026-07-02"

# Use None to download data until today.
# You can also specify a date, for example:
# END_DATE = "2026-08-31"

END_DATE = "2026-09-04"


# Starting value of your portfolio

INITIAL_PORTFOLIO_VALUE = 10000


# ============================================================
# PORTFOLIO ASSET ALLOCATION
# ============================================================

# Only assets with NON-ZERO allocation are included.

#allocations absed on backtesting data from 1 Jan2016 to 2 July 2026

allocations = {

"IAU": 5.50,
"BRK-B": 8.38,
"GOOG": 0.81,
"MSFT": 4.50,
"AAPL": 6.61,
"META": 6.99,
"ARKQ": 0.67,
"SOXQ": 0.39,
"WTV": 2.51,
"MCHI": 6.55,
"EWT": 3.18,
"EWY": 0.34,
"SCHP": 5.11,
"XLU": 1.73,
"BIL": 5.32,
"XAR": 1.23,
"XLE": 7.00,
"CIBR": 1.62,
"XMMO": 1.81,
"IJR": 0.84,
"RTX": 2.23,
"VEA": 5.80,
"ASHR": 2.70,
"SCHD": 1.36,
"VOO": 6.50,
"GILD": 7.11,
"AZN": 3.20


}




# ============================================================
# CONVERT ALLOCATIONS INTO PORTFOLIO WEIGHTS
# ============================================================

weights = pd.Series(allocations, dtype=float)

# Convert percentage into decimal
weights = weights / 100


# Check total allocation

total_allocation = weights.sum()

print("=" * 60)
print("PORTFOLIO ALLOCATION")
print("=" * 60)

print(weights)

print("\nTotal Portfolio Allocation:", total_allocation)


# Safety check

if not np.isclose(total_allocation, 1.0):

    print(
        "\nWARNING: Portfolio allocation does not equal 100%."
    )

    print(
        "The program will normalize the weights automatically."
    )

    weights = weights / weights.sum()


# ============================================================
# DOWNLOAD HISTORICAL DATA
# ============================================================

tickers = list(weights.index)

print("\nDownloading historical data...")


data = yf.download(
    tickers,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)


# ============================================================
# EXTRACT CLOSING PRICES
# ============================================================

prices = data["Close"]


# In case only one ticker is downloaded

if isinstance(prices, pd.Series):

    prices = prices.to_frame()


# Arrange columns according to portfolio weights

prices = prices.reindex(columns=weights.index)


# ============================================================
# HANDLE MISSING DATA
# ============================================================

# Forward-fill missing prices.
# This is useful when some markets have holidays
# on different dates.

prices = prices.ffill()


# Remove rows where data is still incomplete.

prices = prices.dropna()


# ============================================================
# DISPLAY DOWNLOADED PRICES
# ============================================================

print("\n" + "=" * 60)
print("PRICE DATA")
print("=" * 60)

print(prices.head())

print("\nLatest Prices:")

print(prices.tail())


# ============================================================
# CALCULATE DAILY RETURNS OF EACH ASSET
# ============================================================

asset_returns = prices.pct_change()


# Remove first row because it will contain NaN

asset_returns = asset_returns.dropna()


# ============================================================
# CALCULATE WEIGHTED PORTFOLIO DAILY RETURN
# ============================================================

portfolio_returns = asset_returns.dot(weights)


portfolio_returns.name = "Portfolio Daily Return"


# ============================================================
# CALCULATE PORTFOLIO VALUE
# ============================================================

portfolio_value = (
    INITIAL_PORTFOLIO_VALUE
    * (1 + portfolio_returns).cumprod()
)


# Add the initial portfolio value
# on the first available trading date

first_date = prices.index[0]


starting_value = pd.Series(
    [INITIAL_PORTFOLIO_VALUE],
    index=[first_date],
    name="Portfolio Value"
)


portfolio_value.name = "Portfolio Value"


portfolio_value = pd.concat(
    [
        starting_value,
        portfolio_value
    ]
)


# Remove duplicate dates

portfolio_value = portfolio_value[
    ~portfolio_value.index.duplicated(
        keep="first"
    )
]


# ============================================================
# CREATE DATE-WISE PORTFOLIO PERFORMANCE TABLE
# ============================================================

results = pd.DataFrame()


results["Portfolio Value"] = portfolio_value


# Daily portfolio return

results["Daily Return"] = (
    results["Portfolio Value"]
    .pct_change()
)


# Cumulative return from the beginning

results["Cumulative Return"] = (
    results["Portfolio Value"]
    / INITIAL_PORTFOLIO_VALUE
    - 1
)


# ============================================================
# CALCULATE STANDARD DEVIATION
# ============================================================

# Daily standard deviation

daily_standard_deviation = (
    portfolio_returns.std()
)


# Annualized standard deviation

annualized_standard_deviation = (
    daily_standard_deviation
    * np.sqrt(252)
)


# ============================================================
# CALCULATE TOTAL RETURN
# ============================================================

final_portfolio_value = (
    results["Portfolio Value"].iloc[-1]
)


total_return = (
    final_portfolio_value
    / INITIAL_PORTFOLIO_VALUE
    - 1
)


# ============================================================
# CALCULATE BEST AND WORST DAYS
# ============================================================

best_daily_return = (
    portfolio_returns.max()
)


worst_daily_return = (
    portfolio_returns.min()
)


best_day = (
    portfolio_returns.idxmax()
)


worst_day = (
    portfolio_returns.idxmin()
)


# ============================================================
# PRINT PORTFOLIO SUMMARY
# ============================================================

print("\n")

print("=" * 60)

print("PORTFOLIO PERFORMANCE SUMMARY")

print("=" * 60)


print(
    f"\nStart Date: {START_DATE}"
)


print(
    f"Initial Portfolio Value: "
    f"{INITIAL_PORTFOLIO_VALUE:,.2f}"
)


print(
    f"Final Portfolio Value: "
    f"{final_portfolio_value:,.2f}"
)


print(
    f"Total Portfolio Return: "
    f"{total_return * 100:.2f}%"
)


print(
    f"Daily Standard Deviation: "
    f"{daily_standard_deviation * 100:.4f}%"
)


print(
    f"Annualized Standard Deviation: "
    f"{annualized_standard_deviation * 100:.2f}%"
)


print(
    f"\nBest Daily Return: "
    f"{best_daily_return * 100:.2f}%"
)


print(
    f"Best Day: "
    f"{best_day.date()}"
)


print(
    f"\nWorst Daily Return: "
    f"{worst_daily_return * 100:.2f}%"
)


print(
    f"Worst Day: "
    f"{worst_day.date()}"
)


# ============================================================
# DATE-WISE RESULTS
# ============================================================

print("\n")

print("=" * 60)

print("DATE-WISE PORTFOLIO PERFORMANCE")

print("=" * 60)


print(results)


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

results.to_csv(
    "portfolio_performance.csv"
)


print(
    "\nPortfolio performance saved successfully!"
)


print(
    "File name: portfolio_performance.csv"
)



