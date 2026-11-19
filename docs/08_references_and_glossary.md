# Chapter VIII — References and Glossary

**Research only:** Check the current API documentation, licensing and regional availability before implementation.

## Technical references
- [Python](https://docs.python.org/3/) — language and standard library
- [pandas](https://pandas.pydata.org/docs/) — time series, grouping, missing data
- [NumPy](https://numpy.org/doc/stable/) — numerical methods
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — sequential validation (extra purging may be needed)
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [PyTorch](https://pytorch.org/docs/stable/) — optional sequence models
- [Binance Public Data](https://data.binance.vision/) — example historical crypto venue data; verify usage rights
- [CCXT](https://docs.ccxt.com/) — exchange connector library
- [TradingView Pine Script](https://www.tradingview.com/pine-script-docs/) — custom indicators
- [OANDA developer portal](https://developer.oanda.com/) — example FX API; access depends on jurisdiction
- [pytest](https://docs.pytest.org/en/stable/) — testing

## Glossary

| Term | Meaning |
|---|---|
| OHLCV | Open, high, low, close and source-specific volume in an interval |
| Log return | Natural logarithm of the price ratio between two observations |
| Forecast horizon | Time distance between prediction and its target |
| EMA | Exponential moving average |
| RSI | Relative Strength Index, a momentum oscillator |
| ATR | Average True Range, based on historical price ranges |
| Look-ahead bias | Accidental use of information unavailable at decision time |
| Data leakage | Training or features contaminated by evaluation/future information |
| Walk-forward | Test successive future windows after fitting on the past |
| Calibration | Reliability of predicted probabilities vs actual event frequencies |
| Backtest | Historical rule simulation subject to market/cost assumptions |
| Spread | Difference between buy and sell quotes |
| Slippage | Price difference between planned and realized/simulated execution |
| Drawdown | Equity decline from prior peak |
| Idempotency | Repeat requests do not create duplicate side effects |
| Kill switch | Control that stops new simulated execution |
| Paper trading | Simulated positions, orders and money, not real capital |

## Revision exercise
Choose one glossary term and write **definition → formula → example → typical mistake → test**. Add it to your experiment journal. The repository should remain a reusable technical textbook as well as a portfolio project.
