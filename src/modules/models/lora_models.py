import torch


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# your code here
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev) #норм распределение с параметрами 0 и сигма квадрат
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        x = x @ self.A @ self.B
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank)

        device = linear.weight.device
        dtype = linear.weight.dtype

        self.lora.A = torch.nn.Parameter(self.lora.A.to(device=device, dtype=dtype))
        self.lora.B = torch.nn.Parameter(self.lora.B.to(device=device, dtype=dtype))

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def _replace_module(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                lora_layer = LinearWithLoRA(child, rank=8)
                setattr(module, name, lora_layer)
            else:
                _replace_module(child)
                
def replace_with_lora(model, rank=8):  
    for param in model.parameters():
        param.requires_grad = False  
    _replace_module(model)
    return model




