"""
Turn raw LOBSTER order-book event data into the feature+label CSVs that
data.py reads. Input is the LOBSTER message/orderbook CSV pairs fetched by
fetch_lobster_sample.py (real NASDAQ L2 data, one trading day per symbol).

Feature vector (44 dims, fixed order -- column 0 doubles as the label's
reference price in data.py, so it must stay first):
    [0]      mid_price
    [1]      spread                    (ask_price_1 - bid_price_1)
    [2]      weighted_mid_price        (microprice: size-weighted between
                                         best bid/ask)
    [3]      order_imbalance           (all levels: (bid_sz - ask_sz) /
                                         (bid_sz + ask_sz))
    [4:14]   bid_price_1..10
    [14:24]  bid_size_1..10
    [24:34]  ask_price_1..10
    [34:44]  ask_size_1..10

Label: forward VWAP (raw dollar price -- data.py applies the log/sigma
normalization) at 12 horizons: 1,2,3,5,10,20,30,60,90,120,180,300 seconds
after each 1-second snapshot. sigma is the trailing realized volatility of
1-second mid-price log-returns, used by data.py to z-score the label.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

LEVELS = 10
HORIZONS_SECONDS = [1, 2, 3, 5, 10, 20, 30, 60, 90, 120, 180, 300]
SENTINEL = 1e6  # LOBSTER's "no such level" price sentinel is +/-9999999999 (i.e. /1e4 -> +/-999999.9999)


# ---------------------------------------------------------------- loaders --

def load_lobster(ticker, levels, raw_dir):
    """Returns (book_df indexed by time, trades_df with time/price/size)."""
    pattern = os.path.join(raw_dir, f"LOBSTER_SampleFile_{ticker}_*_{levels}",
                            f"{ticker}_*_message_{levels}.csv")
    message_paths = glob.glob(pattern)
    if not message_paths:
        raise FileNotFoundError(
            f"no LOBSTER message file found under {raw_dir} for {ticker} at {levels} levels "
            f"-- run fetch_lobster_sample.py first")
    message_path = message_paths[0]
    orderbook_path = message_path.replace('_message_', '_orderbook_')

    msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
    msg = pd.read_csv(message_path, header=None, names=msg_cols)
    msg['price'] = msg['price'] / 1e4

    ob_cols = []
    for lvl in range(1, levels + 1):
        ob_cols += [f'ask_price_{lvl}', f'ask_size_{lvl}', f'bid_price_{lvl}', f'bid_size_{lvl}']
    book = pd.read_csv(orderbook_path, header=None, names=ob_cols)
    price_cols = [c for c in ob_cols if 'price' in c]
    book[price_cols] = book[price_cols] / 1e4
    book[price_cols] = book[price_cols].where(book[price_cols].abs() < SENTINEL).ffill().bfill()

    book['time'] = msg['time'].values
    book = book.set_index('time')

    trades = msg[msg['type'].isin([4, 5])][['time', 'price', 'size']].reset_index(drop=True)
    return book, trades


# ------------------------------------------------------------- features ----

def reindex_to_grid(book, grid_times):
    """Last known book state at or before each grid timestamp (as-of join)."""
    book = book.sort_index()
    idx = np.searchsorted(book.index.values, grid_times, side='right') - 1
    idx = np.clip(idx, 0, len(book) - 1)
    snap = book.iloc[idx].reset_index(drop=True)
    snap['time'] = grid_times
    return snap


def compute_features(snap, levels=LEVELS):
    bid_px = snap[[f'bid_price_{i}' for i in range(1, levels + 1)]].values
    bid_sz = snap[[f'bid_size_{i}' for i in range(1, levels + 1)]].values
    ask_px = snap[[f'ask_price_{i}' for i in range(1, levels + 1)]].values
    ask_sz = snap[[f'ask_size_{i}' for i in range(1, levels + 1)]].values

    best_bid, best_ask = bid_px[:, 0], ask_px[:, 0]
    best_bid_sz, best_ask_sz = bid_sz[:, 0], ask_sz[:, 0]

    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    weighted_mid = (best_bid * best_ask_sz + best_ask * best_bid_sz) / (best_bid_sz + best_ask_sz + 1e-8)
    total_bid_sz = bid_sz.sum(axis=1)
    total_ask_sz = ask_sz.sum(axis=1)
    imbalance = (total_bid_sz - total_ask_sz) / (total_bid_sz + total_ask_sz + 1e-8)

    features = np.column_stack([
        mid_price, spread, weighted_mid, imbalance,
        bid_px, bid_sz, ask_px, ask_sz,
    ])
    return features, mid_price


def forward_vwap(trade_times, trade_price, trade_size, snap_times, horizons, mid_lookup_times, mid_lookup_values):
    """For every snapshot time, the trade-VWAP over (t, t+h] for each horizon.

    Falls back to the prevailing mid-price at t+h when no trades occur in
    the window (documented deviation from a "pure" VWAP; keeps the label
    defined everywhere instead of dropping illiquid windows).
    """
    order = np.argsort(trade_times)
    trade_times = trade_times[order]
    notional = (trade_price * trade_size)[order]
    size = trade_size[order]
    cum_notional = np.concatenate([[0.0], np.cumsum(notional)])
    cum_size = np.concatenate([[0.0], np.cumsum(size)])

    n_snap, n_h = len(snap_times), len(horizons)
    out = np.empty((n_snap, n_h), dtype=np.float64)

    lo_idx = np.searchsorted(trade_times, snap_times, side='right')
    for j, h in enumerate(horizons):
        hi_idx = np.searchsorted(trade_times, snap_times + h, side='right')
        window_size = cum_size[hi_idx] - cum_size[lo_idx]
        window_notional = cum_notional[hi_idx] - cum_notional[lo_idx]
        vwap = np.divide(window_notional, window_size, out=np.full(n_snap, np.nan), where=window_size > 0)

        fallback_idx = np.searchsorted(mid_lookup_times, snap_times + h, side='right') - 1
        fallback_idx = np.clip(fallback_idx, 0, len(mid_lookup_values) - 1)
        fallback = mid_lookup_values[fallback_idx]

        out[:, j] = np.where(window_size > 0, vwap, fallback)
    return out


def build_dataset(book, trades, levels=LEVELS, horizons=HORIZONS_SECONDS,
                   snapshot_interval=1.0, vol_window=300, warmup_seconds=60):
    session_start = book.index.min() + warmup_seconds
    session_end = book.index.max() - max(horizons)
    if session_end <= session_start:
        raise ValueError('not enough session time to build a single snapshot with warmup+horizon margins')

    grid_times = np.arange(session_start, session_end, snapshot_interval)
    snap = reindex_to_grid(book, grid_times)
    features, mid_price = compute_features(snap, levels=levels)

    log_ret = np.diff(np.log(mid_price), prepend=np.log(mid_price[0]))
    sigma = pd.Series(log_ret).rolling(window=int(vol_window / snapshot_interval),
                                        min_periods=int(vol_window / snapshot_interval)).std().values

    raw_mid_price = ((book['bid_price_1'] + book['ask_price_1']) / 2.0).values
    labels = forward_vwap(trades['time'].values, trades['price'].values, trades['size'].values,
                           grid_times, horizons,
                           mid_lookup_times=book.index.values, mid_lookup_values=raw_mid_price)

    rows = np.column_stack([features, sigma, labels])
    valid = ~np.isnan(rows).any(axis=1)
    return grid_times[valid], rows[valid]


# ------------------------------------------------------------------ I/O ----

def write_feature_csv(out_path, symbol, times, rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        for t, row in zip(times, rows):
            f.write(f"{symbol},{t:.3f},{'|'.join(f'{v:.8g}' for v in row)}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbols', nargs='+', default=['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT'])
    p.add_argument('--levels', type=int, default=LEVELS)
    p.add_argument('--raw_dir', default='./raw/lobster', help='dir passed to fetch_lobster_sample.py')
    p.add_argument('--out_dir', default='./features')
    p.add_argument('--val_fraction', type=float, default=0.2,
                    help='fraction of each symbol-day held out (from the end) for validation')
    p.add_argument('--vol_window', type=float, default=300, help='seconds, trailing window for sigma')
    p.add_argument('--warmup_seconds', type=float, default=60)
    args = p.parse_args()

    train_files, val_files = [], []
    for symbol in args.symbols:
        print(f'Building features for {symbol}...')
        book, trades = load_lobster(symbol, args.levels, args.raw_dir)

        times, rows = build_dataset(book, trades, levels=args.levels,
                                     vol_window=args.vol_window, warmup_seconds=args.warmup_seconds)
        print(f'  {len(times)} usable snapshots')

        split_idx = int(len(times) * (1 - args.val_fraction))

        train_name = f'{symbol}_train.csv'
        val_name = f'{symbol}_val.csv'
        write_feature_csv(os.path.join(args.out_dir, train_name), symbol,
                           times[:split_idx], rows[:split_idx])
        write_feature_csv(os.path.join(args.out_dir, val_name), symbol,
                           times[split_idx:], rows[split_idx:])
        train_files.append(train_name)
        val_files.append(val_name)

    with open(os.path.join(args.out_dir, 'lob_us_train.txt'), 'w') as f:
        f.write('\n'.join(train_files) + '\n')
    with open(os.path.join(args.out_dir, 'lob_us_val.txt'), 'w') as f:
        f.write('\n'.join(val_files) + '\n')

    print(f'wrote {len(train_files)} train / {len(val_files)} val feature files to {args.out_dir}')


if __name__ == '__main__':
    main()
