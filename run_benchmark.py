"""
Train and evaluate all four models on the same horizon and data, then write a
benchmark table to RESULTS.md.

Assumes features have already been built (see README steps 1-2):
    python data_prep/fetch_lobster_sample.py --tickers AAPL --levels 10
    python data_prep/build_features.py --symbols AAPL --levels 10

Usage:
    python run_benchmark.py --timestep 3 --epochs 30
"""
import argparse
import time
from datetime import date

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import HORIZONS_SECONDS, load_xy
from model import build_sklearn_model
from train import predict, train_mlp

MODELS = ['linear', 'knn', 'tree', 'mlp']
MODEL_LABELS = {
    'linear': 'Linear Regression',
    'knn': 'k-Nearest Neighbors',
    'tree': 'Decision Tree',
    'mlp': 'MLP (neural net)',
}


def run_one(name, data, args):
    X_train, y_train, X_val, y_val = data
    t0 = time.time()
    if name == 'mlp':
        model = train_mlp(X_train, y_train, X_val, y_val, args, verbose=False)
    else:
        model = build_sklearn_model(name)
        model.fit(X_train, y_train)

    preds = predict(name, model, X_val)
    return {
        'rmse': mean_squared_error(y_val, preds) ** 0.5,
        'mae': mean_absolute_error(y_val, preds),
        'r2': r2_score(y_val, preds),
        'seconds': time.time() - t0,
        'train_n': len(X_train),
        'val_n': len(X_val),
    }


def write_results(results, args):
    horizon = HORIZONS_SECONDS[args.timestep]
    any_r = next(iter(results.values()))
    best_name = min(results, key=lambda n: results[n]['rmse'])

    lines = []
    lines.append('# Benchmark results\n')
    lines.append(f'Forward-VWAP regression at the **{horizon}s** horizon '
                 f'(`--timestep {args.timestep}`). Lower RMSE/MAE is better; '
                 f'higher R² is better.\n')
    lines.append(f'- data: `{args.dataset_root}` '
                 f'({any_r["train_n"]} train / {any_r["val_n"]} val snapshots, '
                 f'{args.in_features} features per snapshot)')
    lines.append(f'- device (mlp): {args.device}, mlp epochs: {args.epochs}')
    lines.append(f'- generated: {date.today().isoformat()}\n')
    lines.append('| model | RMSE | MAE | R² | train time (s) |')
    lines.append('|---|---|---|---|---|')
    for name in MODELS:
        r = results[name]
        label = MODEL_LABELS[name] + (' ⭐' if name == best_name else '')
        lines.append(f'| {label} | {r["rmse"]:.4f} | {r["mae"]:.4f} '
                     f'| {r["r2"]:.4f} | {r["seconds"]:.0f} |')
    lines.append('')
    lines.append('Labels are sigma-normalized log-returns, so RMSE is in units of '
                 'trailing volatility (≈2.0 means the model is off by about two '
                 'standard deviations of recent returns).\n')
    lines.append('R² near zero (or slightly negative) is the expected starting '
                 'point here: on a single trading day of one symbol, short-horizon '
                 'price moves are close to random, so no simple model beats just '
                 'predicting the average. This table is the baseline that feature '
                 'work, more data, and tuning get measured against.\n')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'wrote {args.out}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_root', default='./features')
    p.add_argument('--train_list', default='lob_us_train.txt')
    p.add_argument('--val_list', default='lob_us_val.txt')
    p.add_argument('--in_features', type=int, default=44)
    p.add_argument('--timestep', type=int, default=3)
    p.add_argument('--epochs', type=int, default=30, help='only used for the mlp')
    p.add_argument('--batch_size', type=int, default=256, help='only used for the mlp')
    p.add_argument('--lr', type=float, default=1e-3, help='only used for the mlp')
    p.add_argument('--device', default='cpu', help='only used for the mlp (cpu or cuda)')
    p.add_argument('--seed', type=int, default=222)
    p.add_argument('--out', default='RESULTS.md')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    X_train, y_train = load_xy(args.dataset_root, args.train_list, args.timestep)
    X_val, y_val = load_xy(args.dataset_root, args.val_list, args.timestep)
    data = (X_train, y_train, X_val, y_val)

    results = {}
    for name in MODELS:
        print(f'=== {name} ===', flush=True)
        results[name] = run_one(name, data, args)
        r = results[name]
        print(f'  RMSE {r["rmse"]:.4f}  MAE {r["mae"]:.4f}  R2 {r["r2"]:.4f}  '
              f'({r["seconds"]:.0f}s)', flush=True)

    write_results(results, args)


if __name__ == '__main__':
    main()
