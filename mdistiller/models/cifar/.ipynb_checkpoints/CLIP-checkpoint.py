import torch
import torch.nn as nn

class CLIPTeacher(nn.Module):
    def __init__(self, logits_path, features_path, device="cuda"):
        super().__init__()
        cache = torch.load(logits_path)
        logits = cache["logits"]

        cache = torch.load(features_path)
        feats = cache["features"]

        self.register_buffer("logits", logits.to(device))
        self.register_buffer("feats", feats.to(device))

    def forward(self, indices):
        return self.logits[indices], self.feats[indices]