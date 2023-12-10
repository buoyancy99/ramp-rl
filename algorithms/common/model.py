import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch, activation_fn=nn.ReLU, squash_output=False):
        super(Mlp, self).__init__()
        if len(net_arch) > 0:
            modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(nn.Linear(last_layer_dim, output_dim))
        if squash_output:
            modules.append(nn.Tanh())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
