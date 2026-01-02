import torch.nn as nn
class ResNet18(nn.Module):
    def __init__(self): super().__init__(); self.conv=nn.Conv2d(3,64,3,padding=1)
