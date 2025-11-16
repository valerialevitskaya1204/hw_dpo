# # your code here
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from .lora_models import replace_with_lora
from ..utils import device
import kagglehub



import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_kaggle_lora(model_path="valery1891/qwen-lora/Transformers/default/1"):
    download_dir = kagglehub.model_download(model_path)
    return download_dir

def download_models_from_hf(model_name, 
                            torch_dtype=None, 
                            device_map=None, 
                            trust_remote_code=None,
                            replace_lora=False,
                            rank=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch_dtype and device_map and trust_remote_code:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    torch_dtype=torch_dtype,
                                                    device_map=device_map,  
                                                    trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    if replace_lora:
        model = replace_with_lora(model, rank=rank)
    return tokenizer, model

def inference_pretrained(model, tokenizer, prompt="tell me a story about a red hat."):
    lora_dir = download_kaggle_lora("valery1891/qwen-lora/Transformers/default/1")

    part1 = os.path.join(lora_dir, "pytorch_model-00001-of-00002.bin")
    part2 = os.path.join(lora_dir, "pytorch_model-00002-of-00002.bin")

    state_dict = torch.load(part1, map_location="cpu")
    state_dict2 = torch.load(part2, map_location="cpu")
    state_dict.update(state_dict2)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(output[0], skip_special_tokens=True)




