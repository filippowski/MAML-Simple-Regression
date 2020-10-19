from model import *


def make_hidden_layer(dim_input, dim_output, relu=True):
    layers = [Linear(dim_input, dim_output)]
    if relu:
        layers.append(BatchNorm1d(dim_output))
        layers.append(ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_hidden_layerBN(dim_input, dim_output, relu=True):
    layers = [Linear(dim_input, dim_output),
              BatchNorm1d(dim_output)]
    if relu:
        layers.append(ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_layers(dim_input, dim_hidden, dim_output, bn=False):
    dims = [dim_input] + list(dim_hidden) + [dim_output]
    inouts = [dims[i:i + 2] for i in range(len(dims) - 1)]
    relus = [True] * len(inouts)
    relus[-1] = False  # last layer w/o non-linear activation function
    f = make_hidden_layerBN if bn else make_hidden_layer
    layers = [f(nin, nout, relu) for (nin, nout), relu in zip(inouts, relus)]
    return nn.Sequential(*layers)


class Regressor(nn.Module):
    def __init__(self,
                 dim_input=1,
                 dim_output=1,
                 dim_hidden=(40, 40)):
        super(Regressor, self).__init__()

        self.layers = make_layers(dim_input, dim_hidden, dim_output)

    def forward(self, x):
        out = self.layers(x)
        return out


class RegressorBN(nn.Module):
    def __init__(self,
                 dim_input=1,
                 dim_output=1,
                 dim_hidden=(40, 40)):
        super(RegressorBN, self).__init__()

        self.layers = make_layers(dim_input, dim_hidden, dim_output, bn=True)

    def forward(self, x):
        out = self.layers(x)
        return out
