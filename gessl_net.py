import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from learner import Learner


def simclr_loss(z1, z2, temp=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temp

    eye = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    exp_sim = torch.exp(sim)
    idx = torch.arange(2 * N, device=z.device)
    pos_idx = (idx + N) % (2 * N)

    pos = exp_sim[idx, pos_idx]
    den = exp_sim.sum(dim=1)

    loss = -torch.log(pos / den).mean()
    return loss


class GeSSL(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.update_lr = args.update_lr
        self.general_lr = args.general_lr

        self.task_num = args.task_num
        self.n_pairs = args.n_pairs
        self.update_step = args.update_step

        self.mu = getattr(args, "mu", 0.7)
        self.k_disc = getattr(args, "k_disc", 10.0)
        self.temp = getattr(args, "temp", 0.2)

        self.net = Learner(config, args.imgc, args.imgsz)

        embed_dim = config[-1][1][0]
        self.g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

        self.general_optim = optim.Adam(
            list(self.net.parameters()) + list(self.g.parameters()),
            lr=self.general_lr
        )

    def encode(self, x, vars=None):
        return self.net(x, vars=vars, bn_training=True)

    def ssl_loss_batch(self, x, vars=None):
        B = x.size(0) // 2
        x1 = x[:B]
        x2 = x[B:]
        f1 = self.encode(x1, vars)
        f2 = self.encode(x2, vars)
        return simclr_loss(f1, f2, temp=self.temp)

    def disc_loss(self, x_view, x_anchor, vars=None):
        f_v = self.encode(x_view, vars)
        f_a = self.encode(x_anchor, vars)

        f_v_n = F.normalize(f_v, dim=1)
        f_a_n = F.normalize(f_a, dim=1)

        dist = 1.0 - torch.mm(f_a_n, f_v_n.t())

        a = self.g(f_a).squeeze(-1)
        a = a.unsqueeze(1)

        w = torch.sigmoid(self.k_disc * (a - dist))

        loss = w * dist + (1.0 - w) * (-dist)
        loss = loss.mean()
        return loss

    def forward(self, x_spt, x_qry):
        task_num, setsz, c, h, w = x_spt.size()
        B = self.n_pairs
        assert setsz == 2 * B

        losses_q = [0.0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            xs = x_spt[i]
            x_view_s = xs[:B]
            x_anchor_s = xs[B:]

            loss_ssl_sup = self.ssl_loss_batch(xs, vars=None)
            loss_disc_sup = self.disc_loss(x_view_s, x_anchor_s, vars=None)
            inner_loss = loss_ssl_sup + self.mu * loss_disc_sup

            grads = torch.autograd.grad(
                inner_loss,
                self.net.parameters(),
                create_graph=True
            )

            fast_weights = [
                p - self.update_lr * g_
                for (p, g_) in zip(self.net.parameters(), grads)
            ]

            loss_q0 = self.ssl_loss_batch(x_qry[i], vars=None)
            losses_q[0] += loss_q0

            loss_q1 = self.ssl_loss_batch(x_qry[i], vars=fast_weights)
            losses_q[1] += loss_q1

            for k in range(1, self.update_step):
                xs = x_spt[i]
                x_view_s = xs[:B]
                x_anchor_s = xs[B:]

                loss_ssl_sup = self.ssl_loss_batch(xs, vars=fast_weights)
                loss_disc_sup = self.disc_loss(x_view_s, x_anchor_s, vars=fast_weights)
                inner_loss = loss_ssl_sup + self.mu * loss_disc_sup

                grads = torch.autograd.grad(
                    inner_loss,
                    fast_weights,
                    create_graph=True
                )

                fast_weights = [
                    w_ - self.update_lr * g_
                    for (w_, g_) in zip(fast_weights, grads)
                ]

                loss_q = self.ssl_loss_batch(x_qry[i], vars=fast_weights)
                losses_q[k + 1] += loss_q

        losses_q = [l / task_num for l in losses_q]
        meta_loss = losses_q[self.update_step]

        self.general_optim.zero_grad()
        meta_loss.backward()
        self.general_optim.step()

        vals = [l.item() for l in losses_q]
        return np.array(vals)
