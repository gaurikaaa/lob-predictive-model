# Benchmark results

Forward-VWAP regression at the **5s** horizon (`--timestep 3`). Lower RMSE/MAE is better; higher R² is better.

- data: `./features` (18192 train / 4549 val snapshots, 44 features per snapshot)
- device (mlp): cpu, mlp epochs: 30
- generated: 2026-07-20

| model | RMSE | MAE | R² | train time (s) |
|---|---|---|---|---|
| Linear Regression ⭐ | 2.0969 | 1.5907 | -0.0129 | 0 |
| k-Nearest Neighbors | 2.1977 | 1.6708 | -0.1127 | 2 |
| Decision Tree | 2.1481 | 1.6154 | -0.0630 | 0 |
| MLP (neural net) | 2.2010 | 1.6791 | -0.1159 | 12 |

Labels are sigma-normalized log-returns, so RMSE is in units of trailing volatility (≈2.0 means the model is off by about two standard deviations of recent returns).

R² near zero (or slightly negative) is the expected starting point here: on a single trading day of one symbol, short-horizon price moves are close to random, so no simple model beats just predicting the average. This table is the baseline that feature work, more data, and tuning get measured against.
