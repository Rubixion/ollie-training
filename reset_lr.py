import torch

ckpt = torch.load("app_checkpoint.pt", map_location="cpu", weights_only=False)
ckpt["finetune"] = True
torch.save(ckpt, "app_checkpoint.pt")
print(f"Done — will resume from epoch {ckpt['epoch'] + 1} with fresh lr=1e-4, T_max=150")
