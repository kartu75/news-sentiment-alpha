

---

# News Sentiment Alpha — Project Writeup

## Objective
Build a news-driven trading strategy for JPMorgan (JPM) using NLP sentiment analysis on financial headlines, backtest it, and evaluate whether public news sentiment can generate alpha across different market regimes.

---

## Stack
- **Python** — pandas, numpy, yfinance, sklearn, backtesting.py
- **NLP** — CountVectorizer, ComplementNB
- **Backtesting** — backtesting.py with custom Strategy class

---

## Methodology

### 1. Sentiment Model
Trained a Naive Bayes classifier (ComplementNB) on the Financial PhraseBank dataset (4,846 labelled financial headlines). Used CountVectorizer with 10,000 features to convert text to word count vectors.

```
Training data distribution:
neutral     2879 (59%)
positive    1363 (28%)
negative     604 (12%)

Model accuracy: 71% (3-class: positive / negative / neutral)
```

ComplementNB was chosen over MultinomialNB due to better recall on the minority class (negative), which is the most critical signal for downside protection in trading.

---

### 2. Data
- **News data**: 2,873 JPM headlines from `raw_partner_headlines.csv` (Oct 2018 — Jun 2020)
- **Price data**: JPM OHLCV pulled via yfinance for the same period
- **Merged**: 401 trading days with both price and sentiment data
- **Note**: Dataset availability was limited to this period. This window unfortunately coincides with the COVID crash, introducing regime bias. A longer dataset spanning multiple market cycles would give more statistically robust results.

---

### 3. Sentiment Aggregation
Multiple headlines per day were aggregated into a single daily sentiment score:

```
sentiment_score = average(sentiment scores per day)
where: positive=+1, neutral=0, negative=-1
```

---

### 4. Signal Generation
Conservative signal thresholds to avoid low-conviction trades:

```
buy      → sentiment_score > 0.5   (strongly positive day)
sell     → sentiment_score < -0.3  (negative day)
no_trade → everything in between
```

```
Signal distribution:
no_trade    440 (79%)
sell         79 (14%)
buy          35  (6%)
```

---

### 5. Trading Strategy
**Entry**: Buy JPM on strongly positive sentiment days (score > 0.5)

**Exit**:
- Take profit → price rises 2% above 20-day SMA
- Stop loss → price falls 3% below 20-day SMA

**Parameters**: Starting capital $10,000, commission 0.2% per trade

---

## Results

### Overall (Oct 2018 — Jun 2020)

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total Return | -19.0% | -4.6% |
| Sharpe Ratio | -1.29 | — |
| Max Drawdown | -21.3% | — |
| Win Rate | 47.1% | — |
| # Trades | 17 | — |
| Exposure Time | 16.7% | 100% |
| Commissions | $644 | — |

---

### Regime Analysis

| Metric | Pre-COVID (Oct 2018 — Jan 2020) | COVID (Feb 2020 — Jun 2020) |
|---|---|---|
| Strategy Return | -8.81% | -11.21% |
| Buy & Hold Return | +12.67% | -19.09% |
| Sharpe Ratio | -0.89 | -2.95 |
| # Trades | 14 | 3 |

---

## Conclusion

### Overall Performance
The strategy underperformed a simple buy-and-hold benchmark (-19% vs -4.6%) over the full period, empirically consistent with the **semi-strong form of the Efficient Market Hypothesis (EMH)**.

### Regime Analysis — Key Finding
The regime split reveals a more nuanced story:

**Bull market (Pre-COVID):** Strategy lost 8.81% vs buy and hold gain of 12.67%. Confirms EMH in normal markets — by the time headlines are published, institutional traders and HFT algorithms have already acted on the information.

**Crisis period (COVID):** Strategy lost only 11.21% vs buy and hold loss of 19.09% — **outperforming by 7.88% during the crash.** The strategy's conservative nature (16.7% market exposure) acted as a natural hedge, preserving capital during extreme volatility.

**Core insight:** News sentiment is not an alpha-generating signal in efficient bull markets, but demonstrates meaningful downside protection properties during market crises. This suggests sentiment may be more useful as a **risk-off filter** than a return-generating signal.

---

## What Would Actually Work

| Approach | Why |
|---|---|
| Alternative data (satellite, credit card) | Not yet reflected in prices |
| Earnings surprise models | Predict before announcement |
| HFT news parsing | Act in milliseconds before market adjusts |
| Sentiment on small-cap stocks | Less efficient, news not priced in as fast |

---

## Limitations
- Dataset limited to Oct 2018 — Jun 2020 due to data availability constraints
- Period coincides with COVID crash, introducing significant regime bias
- Sentiment model trained on generic financial news, not JPM-specific language
- Only 35 buy signals over 2 years — too conservative to capture meaningful upside
- $644 commissions on $10,000 capital (6.4% drag) highlights cost of low-conviction active trading
- No transaction cost optimisation
- Does not account for macro regime (bull vs bear market) in signal generation
