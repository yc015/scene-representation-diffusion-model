import torch
from torch import nn

class probeLinearDense(nn.Module):
    def __init__(self, in_channels, out_channels, factor=8, use_bias=False, interpolate=True):
        super().__init__()
        self.dense1 = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.factor = factor
        self.out_channels = out_channels
        self.interpolate = interpolate
    
    def forward(self, x):
        x = self.dense1(x)
        x = x.view(-1, 512 // self.factor, 512 // self.factor, self.out_channels).permute([0, 3, 1, 2])
        # Interpolate the prediction to the same spatial size as that of the original image
        if self.interpolate:
            x = nn.functional.interpolate(x, scale_factor=self.factor)
        return x
    

class probeTwoNonLinearDense(nn.Module):
    def __init__(self, in_channels, out_channels, factor=8, use_bias=False, mid_channels=640):
        super().__init__()
        self.dense1 = nn.Linear(in_channels, mid_channels, bias=use_bias)
        self.factor = factor
        self.nonlinearity = torch.nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(mid_channels, out_channels, bias=use_bias)
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.nonlinearity(x)
        x = self.dense2(x)
        x = x.view(-1, 512 // self.factor, 512 // self.factor, self.out_channels).permute([0, 3, 1, 2])
        
        x = nn.functional.interpolate(x, scale_factor=self.factor)
        return x
