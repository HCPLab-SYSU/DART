import torch
import torch.nn as nn
from models_deit import *
model = darvit_tiny(num_classes=1000,img_size=224, high_res=False).to("cuda:0")
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
from fvcore.nn import FlopCountAnalysis
input_tensor = torch.randn(1, 3, 224, 224).to("cuda:0")

flops = FlopCountAnalysis(model, input_tensor)
print("FLOPs: ", flops.total())  

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params}")