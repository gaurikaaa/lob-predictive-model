"""
Evaluate a saved model on the validation set: load the checkpoint written by
train.py, run predictions, and print RMSE / MAE / R^2. Optionally dump the
per-example predictions to CSV.

    python eval.py --checkpoint ./checkpoints/linear.joblib --timestep 3
    python eval.py --checkpoint ./checkpoints/mlp.pth       --timestep 3

--timestep must match the horizon the model was trained on.
"""
import argparse
import csv

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import HORIZONS_SECONDS, load_xy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True,
                    help='a .joblib file (linear/knn/tree) or a .pth file (mlp)')
    p.add_argument('--dataset_root', default='./features')
    p.add_argument('--val_list', default='lob_us_val.txt')
    p.add_argument('--timestep', type=int, default=3,
                    help=f'horizon the model was trained on (0=1s ... {len(HORIZONS_SECONDS) - 1}=300s)')
    p.add_argument('--out_csv', default=None,
                    help='optional path to dump per-example predictions vs targets')
    return p.parse_args()


def main():
    args = parse_args()
    X_val, y_val = load_xy(args.dataset_root, args.val_list, args.timestep)

    if args.checkpoint.endswith('.pth'):
        import torch

        from model import MLP
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model = MLP(in_features=ckpt['in_features'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_val)).view(-1).cpu().numpy()
    else:
        model = joblib.load(args.checkpoint)
        preds = model.predict(X_val)

    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f'checkpoint: {args.checkpoint}')
    print(f'examples:   {len(y_val)}')
    print(f'RMSE:       {rmse:.6f}')
    print(f'MAE:        {mae:.6f}')
    print(f'R^2:        {r2:.4f}')

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prediction', 'target'])
            writer.writerows(zip(preds.tolist(), y_val.tolist()))
        print(f'predictions written to {args.out_csv}')


if __name__ == '__main__':
    main()
