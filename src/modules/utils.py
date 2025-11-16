import torch
import gc


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_cache():
    torch.cuda.empty_cache()
    gc.collect()