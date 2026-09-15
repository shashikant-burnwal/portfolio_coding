import pandas as pd
import numpy as np
import yfinance as yf

tickers = ["IAU",
    "BRK-B",

    "GOOG",
    "MSFT",
    "AAPL",
    "META",

    "ARKQ",
    "SOXQ",
    "WTV",

    "MCHI",
    "EWT",
    "EWY",

    "SCHP",
    "XLU",
    "BIL",

    "XAR",
    "XLE",

    "CIBR",
    "XMMO",
    "IJR",

    "RTX",

    "VEA",
    "ASHR",
    "SCHD",

    "VOO",

    "GILD",
    "AZN"]
df = yf.download(tickers, start="2026-07-02", end="2026-09-04")["Close"].reindex(columns=tickers)
returns = df.pct_change().dropna()

#now get random weights for each ticker

def get_random_weights(tickers):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    return weights


pf_exp,pf_stds,weights=[],[],[]
for i in range(20000):
    W=get_random_weights(tickers)
    weights.append(W)
    pf_exp.append(W.dot(returns.mean())*252)
    pf_stds.append(np.sqrt(W.T.dot(returns.cov().dot(W) * 252)))


tog=pd.DataFrame({"Returns":pf_exp,"Volatility":pf_stds,'weights':weights})


   
sharpes = (tog.Returns - 0.02) / tog.Volatility
max_sharpe_index = sharpes.idxmax()
max_sharpe_portfolio = tog.loc[max_sharpe_index]

print("max sharpe ratio: " + str(max_sharpe_portfolio) + "\n")
print("weights for max Sharpe ratio portfolio:")
for ticker, weight in zip(tickers, max_sharpe_portfolio["weights"]):
    print(f"{ticker}: {weight:.2%}")

print("max return: " + str(tog.nlargest(1, 'Returns').iloc[0])+"\n")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(tog.Volatility, tog.Returns, marker='o')

plt.title("Efficient Frontier")
plt.xlabel("Volatility")
plt.ylabel("Returns")

plt.show()



