import argparse
import torch
import numpy as np

from gessl_net import GeSSL
from tasks_cifar_ssl import build_cifar_unlabeled, sample_ssl_tasks


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_num', type=int, default=4)
    p.add_argument('--n_pairs', type=int, default=32)

    p.add_argument('--imgc', type=int, default=3)
    p.add_argument('--imgsz', type=int, default=32)

    p.add_argument('--update_lr', type=float, default=0.01)
    p.add_argument('--general_lr', type=float, default=1e-3)

    p.add_argument('--update_step', type=int, default=1)

    p.add_argument('--mu', type=float, default=0.7)
    p.add_argument('--k_disc', type=float, default=10.0)
    p.add_argument('--temp', type=float, default=0.2)

    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--data_root', type=str, default='XXX')

    args = p.parse_args()
    return args


def build_config(feat_dim=64):
    cfg = [
        ('conv2d', [64, 3, 1, 1]),
        ('bn', [64]),
        ('relu', []),
        ('maxpool2d', [2, 2]),

        ('conv2d', [64, 3, 1, 1]),
        ('bn', [64]),
        ('relu', []),
        ('maxpool2d', [2, 2]),

        ('conv2d', [64, 3, 1, 1]),
        ('bn', [64]),
        ('relu', []),
        ('maxpool2d', [2, 2]),

        ('conv2d', [64, 3, 1, 1]),
        ('bn', [64]),
        ('relu', []),
        ('maxpool2d', [2, 2]),

        ('flatten', []),
        ('linear', [feat_dim]),
    ]
    return cfg


def main():
    args = build_args()
    config = build_config(feat_dim=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeSSL(args, config).to(device)

    dataset = build_cifar_unlabeled(root=args.data_root, train=True, download=True)

    for epoch in range(1, args.epochs + 1):
        x_spt, x_qry = sample_ssl_tasks(args, dataset, device)
        losses = model(x_spt, x_qry)
        print(f'epoch {epoch:03d} | ssl_losses = {np.round(losses, 4)}')

    print('done')


if __name__ == '__main__':
    main()
