import torch
import torch.nn.functional as F

from torch import nn

class PatchNet(nn.Module):
    """
    Simple non-deep 28x28 pixel classification CNN
    """
    def __init__(self):
        super(PatchNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Dropout())
        self.linear = nn.Linear(26*26*4, 4)
        self.init_weights()

    def forward(self, input):
        o = self.conv(input)
        o = o.view(input.size(0), -1)
        return self.linear(o)

    def init_weights(self):
        def _wi(m):
            if isinstance(m, torch.nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, torch.nn.LSTM):
                for p in m.parameters():
                    # weights
                    if p.data.dim() == 2:
                        nn.init.orthogonal_(p.data)
                    # initialize biases to 1 (jozefowicz 2015)
                    else:
                        nn.init.constant_(p.data[len(p)//4:len(p)//2], 1.0)
            elif isinstance(m, torch.nn.GRU):
                for p in m.parameters():
                    nn.init.orthogonal_(p.data)
            elif isinstance(m, torch.nn.Conv2d):
                for p in m.parameters():
                    nn.init.uniform_(p.data, -0.1, 0.1)
        self.conv.apply(_wi)
        self.linear.apply(_wi)
