"""
Train one model to predict forward price change at one horizon, then report
RMSE / MAE / R^2 on the validation set and save the trained model.

    python train.py --model linear --timestep 3
    python train.py --model knn    --timestep 3
    python train.py --model tree   --timestep 3
    python train.py --model mlp    --timestep 3 --epochs 30

Models: linear, knn, tree (scikit-learn) and mlp (a small neural network).
--timestep picks which of the 12 horizons to predict (0 = 1s ... 11 = 300s).
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import HORIZONS_SECONDS, load_xy
from model import build_sklearn_model


def metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def train_sklearn(name, X_train, y_train):
    model = build_sklearn_model(name)
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train, X_val, y_val, args, verbose=True):
    """Train the small neural network with a plain mini-batch loop."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from model import MLP

    torch.manual_seed(args.seed)
    device = args.device
    model = MLP(in_features=X_train.shape[1]).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    x_val_t = torch.from_numpy(X_val).to(device)
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).view(-1), yb)
            loss.backward()
            optimizer.step()
        if verbose:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_t).view(-1).cpu().numpy()
            rmse, mae, r2 = metrics(y_val, val_pred)
            print(f'epoch {epoch:3d}  val_rmse {rmse:.6f}  val_mae {mae:.6f}  val_r2 {r2:.4f}')
    return model


def predict(name, model, X):
    if name == 'mlp':
        import torch
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(X).to(device)).view(-1).cpu().numpy()
    return model.predict(X)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=['linear', 'knn', 'tree', 'mlp'])
    p.add_argument('--dataset_root', default='./features')
    p.add_argument('--train_list', default='lob_us_train.txt')
    p.add_argument('--val_list', default='lob_us_val.txt')
    p.add_argument('--timestep', type=int, default=3,
                    help=f'index into the {len(HORIZONS_SECONDS)} horizons to predict '
                         f'({", ".join(f"{i}={h}s" for i, h in enumerate(HORIZONS_SECONDS))})')
    p.add_argument('--epochs', type=int, default=30, help='only used for --model mlp')
    p.add_argument('--batch_size', type=int, default=256, help='only used for --model mlp')
    p.add_argument('--lr', type=float, default=1e-3, help='only used for --model mlp')
    p.add_argument('--device', default='cpu', help='only used for --model mlp (cpu or cuda)')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--seed', type=int, default=222)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    X_train, y_train = load_xy(args.dataset_root, args.train_list, args.timestep)
    X_val, y_val = load_xy(args.dataset_root, args.val_list, args.timestep)
    print(f'train examples: {len(X_train)}  val examples: {len(X_val)}  features: {X_train.shape[1]}')

    if args.model == 'mlp':
        model = train_mlp(X_train, y_train, X_val, y_val, args)
    else:
        model = train_sklearn(args.model, X_train, y_train)

    rmse, mae, r2 = metrics(y_val, predict(args.model, model, X_val))
    print(f'\n{args.model}  RMSE {rmse:.6f}  MAE {mae:.6f}  R2 {r2:.4f}')

    if args.model == 'mlp':
        import torch
        path = os.path.join(args.checkpoint_dir, 'mlp.pth')
        torch.save({'state_dict': model.state_dict(), 'in_features': X_train.shape[1]}, path)
    else:
        path = os.path.join(args.checkpoint_dir, f'{args.model}.joblib')
        joblib.dump(model, path)
    print(f'saved model to {path}')


if __name__ == '__main__':
    main()
