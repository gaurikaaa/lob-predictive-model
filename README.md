# LOB Modelling -- short-horizon price prediction from limit order book data 

A small, beginner-friendly benchmark of regression models that predict
short-horizon price movement from limit-order-book (LOB) data on US equities.
Given one order-book snapshot, each model forecasts the **forward
volume-weighted average price (VWAP) change** at a chosen horizon (1–300 s).

The goal is a clean, reproducible testbed for comparing a few simple modelling
techniques on the same task, features, and data.

### Problem Statement
High-frequency Limit Order Book (LOB) data generates massive amounts of hidden financial information, but traditional predictive models fail to accurately capture the rapid, short-term price and volume changes needed for reliable trading strategies.

### USP
This project uniquely benchmarks advanced deep learning architectures against baseline regression models on real-world LOB data, capturing both spatial and temporal market features to accurately forecast short-term Volume-Weighted Average Price (VWAP) movements.

### Problem Description
This research utilizes high-frequency Limit Order Book data to forecast short-term stock price and volume movements. By comparing traditional Linear Regression against advanced Deep Learning neural networks, the project identifies the most accurate model for predicting market microstructure changes and optimizing automated trading strategies.

## Models

Four models, kept deliberately simple and easy to explain:

| model | `--model` | library | idea |
|---|---|---|---|
| Linear Regression | `linear` | scikit-learn | fit a straight-line relationship |
| k-Nearest Neighbors | `knn` | scikit-learn | average the most similar past snapshots |
| Decision Tree | `tree` | scikit-learn | a flowchart of yes/no rules |
| MLP (neural net) | `mlp` | PyTorch | one small neural network (2 hidden layers) |

The first three are one line of scikit-learn each. The MLP is the only
deep-learning model — the simplest kind (a fully-connected network). All are
trained as continuous regressors and evaluated with RMSE, MAE and R².

## Data

Free real order-book data, no signup:
[LOBSTER](https://lobsterdata.com) publishes a free sample trading day
(2012-06-21) of real reconstructed NASDAQ order books for AAPL, AMZN, GOOG,
INTC, MSFT — genuine exchange-reconstructed 10-level L2 data.
`data_prep/fetch_lobster_sample.py` pulls it from a public mirror (no signup).
One day per symbol, so train/validation is a within-day time split — ideal for
a working benchmark, not a production backtest.

## Feature spec

44 features per snapshot (column 0 doubles as the label's reference price, so
it stays first):

| index | feature |
|---|---|
| 0 | mid_price |
| 1 | spread (ask_1 − bid_1) |
| 2 | weighted_mid_price (microprice) |
| 3 | order_imbalance (aggregate across all levels) |
| 4–13 | bid_price_1..10 |
| 14–23 | bid_size_1..10 |
| 24–33 | ask_price_1..10 |
| 34–43 | ask_size_1..10 |

Each snapshot is one training example: **44 features in → one predicted price
change out.** The label is the forward VWAP at 12 horizons (1, 2, 3, 5, 10, 20,
30, 60, 90, 120, 180, 300 s), expressed as a sigma-normalized log-return (see
`data.py`); `--timestep` selects which one to predict.

## Usage

```bash
pip install -r requirements.txt

# 1. fetch real order book data (LOBSTER sample, free, no signup)
python data_prep/fetch_lobster_sample.py --tickers AAPL --levels 10

# 2. build feature/label CSVs + train/val file lists
python data_prep/build_features.py --symbols AAPL --levels 10

# 3. train a model (choose from the table above)
python train.py --model linear --timestep 3
python train.py --model mlp --timestep 3 --epochs 30

# 4. evaluate a saved model
python eval.py --checkpoint ./checkpoints/linear.joblib --timestep 3
```

`--timestep` selects which of the 12 horizons to predict (0 = 1 s … 3 = 5 s …
11 = 300 s); each run trains one model for one horizon.

To reproduce the full benchmark across all four models:

```bash
python run_benchmark.py --timestep 3 --epochs 30
```

## Results

See [RESULTS.md](RESULTS.md) for the current benchmark table.

## Project layout

```
model.py                     the four models (3 scikit-learn + 1 MLP)
data.py                      loads feature CSVs into (X, y) arrays
train.py                     train one model, report metrics, save it
eval.py                      load a saved model and evaluate it
run_benchmark.py             train+eval all models, write RESULTS.md
data_prep/
  fetch_lobster_sample.py    download LOBSTER sample order books
  build_features.py          raw book/trades -> feature+label CSVs
```
