from model import *


def weights_initialization(m, verbose=False):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if verbose: print("Classname:", classname)
        if m.weight is not None:
            xavier_normal_(m.weight.data, gain=calculate_gain('relu'))
        if m.bias is not None:
            constant_(m.bias.data, 0.02)
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if verbose: print("Classname:", classname)
        if m.weight is not None:  # gammas filling ~1.0
            normal_(m.weight.data, mean=1.0, std=0.01)
        if m.bias is not None:    # betas filling  ~0.0
            zeros_(m.bias.data)
