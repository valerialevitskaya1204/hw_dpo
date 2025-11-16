import time
import torch


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def get_batch(dataset, batch_size=2, max_length=16):
    idx = torch.randint(0, len(dataset), (batch_size,)).tolist()
    
    batch = {
        "input_ids": torch.stack([torch.tensor(dataset[i]["input_ids"][:max_length]) for i in idx]).to("cuda"),
        "labels": torch.stack([torch.tensor(dataset[i]["labels"][:max_length]) for i in idx]).to("cuda"),
        "attention_mask": torch.stack([torch.tensor(dataset[i]["attention_mask"][:max_length]) for i in idx]).to("cuda"),
    }
    return batch


def check_time(model, tokenized_train_dataset, name="", enable_backward=True):
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    model.train()
    
    if enable_backward:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4
        )
    else:
        optimizer = None

    for _ in range(2):
        batch = get_batch(tokenized_train_dataset, batch_size=1, max_length=32)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        if enable_backward:
            optimizer.zero_grad()
            loss.backward()

    torch.cuda.synchronize()

    batch = get_batch(tokenized_train_dataset, batch_size=1, max_length=32)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    start = time.time()
    out = model(**batch)
    torch.cuda.synchronize()
    fwd = time.time() - start

    if enable_backward:
        loss = out.loss
        optimizer.zero_grad()
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd = time.time() - start
    else:
        bwd = 0

    mem = torch.cuda.max_memory_allocated() / 1024**2  
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{name}")
    print(f"Forward:  {fwd:.4f}s")
    if enable_backward:
        print(f"Backward: {bwd:.4f}s")
    print(f"Memory:   {mem:.2f} MB")
   
    return fwd, bwd, mem
