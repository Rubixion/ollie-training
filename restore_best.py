import torch
import shutil
import os

# 1. Back up the current VGGFace2 run
shutil.copy("app_checkpoint.pt", "app_checkpoint_vgg_epoch124.pt")
print("Backed up: app_checkpoint_vgg_epoch124.pt")

# 2. Load the 83% weights and wrap them as a training checkpoint
best_weights = torch.load("app_best.pt", map_location="cpu", weights_only=False)
new_ckpt = {
    "epoch":    -1,     # so training starts at epoch 0
    "model":    best_weights,
    "best_acc": 0.83,   # bar stays at 83% — only overwrites app_best.pt if we beat it
    "threshold": 0.984,
    "finetune": True,   # resets lr=1e-4, fresh cosine schedule (T_max=150)
}
torch.save(new_ckpt, "app_checkpoint.pt")
print("app_checkpoint.pt restored to 83% weights (finetune=True, starts at epoch 0, lr=1e-4)")

# 3. Disable VGGFace2 so training uses LFW-only pairs
if os.path.exists("vggface2_path.txt"):
    os.rename("vggface2_path.txt", "vggface2_path.txt.bak")
    print("VGGFace2 disabled (vggface2_path.txt → .bak)")
else:
    print("VGGFace2 already not active")

print("\nReady — hit Start Training to fine-tune the 83% model on LFW pairs.")
print("To re-enable VGGFace2 later: rename vggface2_path.txt.bak back to vggface2_path.txt")
