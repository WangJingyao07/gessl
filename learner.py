import torch
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):
    def __init__(self, config, imgc, imgsz):
        super().__init__()
        self.config = config
        self.vars = nn.ParameterList()

        c = imgc
        h = imgsz
        w = imgsz
        feat_dim = None

        for name, params in self.config:
            if name == 'conv2d':
                out_c, k, s, p = params
                w_param = nn.Parameter(torch.empty(out_c, c, k, k))
                b_param = nn.Parameter(torch.zeros(out_c))
                nn.init.kaiming_normal_(w_param)
                self.vars.append(w_param)
                self.vars.append(b_param)
                h = (h + 2 * p - k) // s + 1
                w = (w + 2 * p - k) // s + 1
                c = out_c

            elif name == 'bn':
                num = params[0]
                w_param = nn.Parameter(torch.ones(num))
                b_param = nn.Parameter(torch.zeros(num))
                self.vars.append(w_param)
                self.vars.append(b_param)

            elif name == 'maxpool2d':
                k, s = params
                h = (h - k) // s + 1
                w = (w - k) // s + 1

            elif name == 'flatten':
                feat_dim = c * h * w

            elif name == 'linear':
                out_f = params[0]
                if feat_dim is None:
                    in_f = params[1]
                else:
                    in_f = feat_dim
                    feat_dim = out_f
                w_param = nn.Parameter(torch.empty(out_f, in_f))
                b_param = nn.Parameter(torch.zeros(out_f))
                nn.init.kaiming_normal_(w_param)
                self.vars.append(w_param)
                self.vars.append(b_param)

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        for name, params in self.config:
            if name == 'conv2d':
                out_c, k, s, p = params
                w = vars[idx]
                b = vars[idx + 1]
                idx += 2
                x = F.conv2d(x, w, b, stride=s, padding=p)

            elif name == 'bn':
                num = params[0]
                w = vars[idx]
                b = vars[idx + 1]
                idx += 2
                x = F.batch_norm(
                    x,
                    running_mean=None,
                    running_var=None,
                    weight=w,
                    bias=b,
                    training=bn_training
                )

            elif name == 'relu':
                x = F.relu(x, inplace=True)

            elif name == 'maxpool2d':
                k, s = params
                x = F.max_pool2d(x, kernel_size=k, stride=s)

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'linear':
                out_f = params[0]
                w = vars[idx]
                b = vars[idx + 1]
                idx += 2
                x = F.linear(x, w, b)

        return x
