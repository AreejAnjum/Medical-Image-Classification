

import torch
from Code.models import AlexNet

model = AlexNet(in_channels=3, num_classes=8)

x = torch.randn(1, 3, 64, 64)
out = model.features(x)

print(out.shape)
