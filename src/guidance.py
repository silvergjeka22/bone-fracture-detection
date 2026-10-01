"""Guided training: teach the best model to look at the fracture.

loss = weighted cross-entropy + box * Energy loss + contrast * contrastive loss
- Energy loss (Rao et al. 2023): share of the Grad-CAM heat outside the radiologist's box.
- Contrastive loss (SimCLR / SupCon style): features inside fracture boxes alike, background and healthy apart.
Both only after a warm-up, growing slowly; fractured X-rays only (healthy ones have no box).
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import COLOR, GEOMETRY, MEAN, STD, make_loader
from .models import build_model
from .training import class_weights, compute_metrics, predict, save_json, save_model
from .utils import count_parameters, set_seed

CHANNELS = 256  # channels of the guide layer (CNN last block, ResNet18 layer3)


class BoxDataset(Dataset):
    """X-ray, label and a mask of its fracture boxes, grown by `margin` px; flips and rotations move the mask too."""

    def __init__(self, image_set, augment=False, margin=8):
        self.set, self.augment, self.margin = image_set, augment, margin

    def __len__(self):
        return len(self.set)

    def __getitem__(self, i):
        size = self.set.images.shape[1]
        x = torch.from_numpy(self.set.images[i]).float().div(255)[None]
        mask = torch.zeros(1, size, size)
        for x0, y0, x1, y1 in self.set.boxes[i]:
            m = self.margin
            mask[0, max(int(y0) - m, 0):int(np.ceil(y1)) + m, max(int(x0) - m, 0):int(np.ceil(x1)) + m] = 1
        if self.augment:
            x, mask = GEOMETRY(torch.cat([x, mask])).split(1)
            x = COLOR(x).clamp(0, 1)
        return (x - MEAN) / STD, int(self.set.labels[i]), mask


def contrastive_loss(head, features, mask, fractured, temperature=0.1):
    """Supervised contrastive loss on regions: fracture boxes are positives of each other, everything else negative."""
    area_in = mask.sum((1, 2)).clamp_min(1e-6)[:, None]
    area_out = (1 - mask).sum((1, 2)).clamp_min(1e-6)[:, None]
    inside = (features * mask[:, None]).sum((2, 3)) / area_in
    outside = (features * (1 - mask)[:, None]).sum((2, 3)) / area_out  # background, or the whole healthy X-ray
    z = F.normalize(head(torch.cat([inside[fractured], outside])), dim=1)
    label = torch.cat([torch.ones(int(fractured.sum())), torch.zeros(len(outside))]).to(z.device)
    self_pair = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = (z @ z.T / temperature).masked_fill(self_pair, -1e9)
    positive = (label[:, None] == label[None]) & ~self_pair
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    keep = positive.any(1)
    return -((log_prob * positive).sum(1)[keep] / positive.sum(1)[keep]).mean()


def guided_losses(model, head, x, y, mask, box=True, contrast=True, temperature=0.1):
    """Logits, Energy loss and contrastive loss of a batch (a loss is 0 when it is off or has nothing to guide)."""
    saved = {}
    hook = model.guide_layer.register_forward_hook(lambda module, inputs, output: saved.update(a=output))
    logits = model(x)
    hook.remove()
    a = saved["a"]
    m = F.adaptive_avg_pool2d(mask, a.shape[-2:])[:, 0]  # box mask on the 14x14 grid, soft edges
    fractured = (y == 0) & (m.flatten(1).sum(1) > 0)
    energy = con = logits.new_zeros(())
    if box and fractured.any():
        grads = torch.autograd.grad(logits[fractured, 0].sum(), a, create_graph=True)[0]
        cam = F.relu((grads.mean((2, 3), keepdim=True) * a).sum(1))[fractured]  # differentiable Grad-CAM
        energy = (1 - (cam * m[fractured]).sum((1, 2)) / (cam.sum((1, 2)) + 1e-6)).mean()
    if contrast and fractured.any():
        con = contrastive_loss(head, a, m, fractured, temperature)
    return logits, energy, con


def train_guided(name, key, train_set, val_set, device, box=1.0, contrast=0.1, warmup=3, temperature=0.1, margin=8,
                 epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-4, seed=42, num_workers=2, image_size=224,
                 results_dir=None):
    """Train model type `name` with the guided loss, saved as `key`; the val split picks the epoch."""
    print(f"=== {key} | box {box} | contrast {contrast} | warm-up {warmup} epochs ===")
    set_seed(seed)
    model = build_model(name, image_size).to(device)
    head = nn.Sequential(nn.Linear(CHANNELS, CHANNELS), nn.ReLU(), nn.Linear(CHANNELS, 128)).to(device)
    loader = DataLoader(BoxDataset(train_set, augment=True, margin=margin), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = make_loader(val_set, batch_size=2 * batch_size, num_workers=num_workers)
    weight = class_weights(train_set.labels).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {k: [] for k in ("train_loss", "train_acc", "energy", "contrast", "val_loss", "val_acc", "val_f1", "seconds")}
    best_acc, best_epoch, best_state = -1.0, epochs, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        ramp = 1.0 if warmup == 0 else min(1.0, max(0.0, (epoch - warmup) / warmup))  # 0 in the warm-up, then up to 1
        model.train()
        head.train()
        sums, seen, correct = np.zeros(3), 0, 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits, energy, con = guided_losses(model, head, x, y, mask, box > 0 and ramp > 0,
                                                contrast > 0 and ramp > 0, temperature)
            ce = F.cross_entropy(logits, y, weight=weight)
            loss = ce + ramp * (box * energy + contrast * con)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += np.array([ce.item(), energy.item(), con.item()]) * len(y)
            seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        probs, labels, val_loss = predict(model, val_loader, device, weight)
        val = compute_metrics(labels, probs)
        scheduler.step()
        for k, v in zip(history, (*sums[:1] / seen, correct / seen, *sums[1:] / seen, val_loss, val["accuracy"],
                                  val["f1"], time.time() - start)):
            history[k].append(float(v))
        print(f"[{epoch:2d}/{epochs}] loss {history['train_loss'][-1]:.3f} energy {history['energy'][-1]:.3f} "
              f"contrast {history['contrast'][-1]:.3f} | val acc {val['accuracy']:.3f} f1 {val['f1']:.3f}")
        if val["accuracy"] > best_acc:
            best_acc, best_epoch, best_state = val["accuracy"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    info = {"model": key, "arch": name, "history": history, "best_epoch": best_epoch,
            "seconds_per_epoch": float(np.mean(history["seconds"])), "parameters": count_parameters(model),
            "settings": dict(box=box, contrast=contrast, warmup=warmup, temperature=temperature, margin=margin)}
    if results_dir is not None:
        save_model(model, key, results_dir)
        save_json(info, Path(results_dir) / "final" / f"{key}.json")
    return model.eval(), info


def best_on_val(final, keys):
    """The model with the best val F1 at its kept epoch (never chosen on the test set)."""
    return max(keys, key=lambda k: final[k]["history"]["val_f1"][final[k]["best_epoch"] - 1])


def comparison(keys, test, localization, deletion, labels=None):
    """Plain vs guided: classification, hit rate of the explanations, faithfulness; one row per model."""
    labels = labels or {}
    rows = {}
    for k in keys:
        hits = localization[k].drop("Random")["hit rate (%)"]
        rows[labels.get(k, k)] = {
            "test accuracy (%)": 100 * test[k]["accuracy"], "test F1 (%)": 100 * test[k]["f1"],
            "fractures found (%)": 100 * test[k]["recall"], "test AUC (%)": 100 * test[k]["auc"],
            "hit rate, mean of methods (%)": hits.mean(), "hit rate, best method (%)": hits.max(),
            "best method": hits.idxmax(), "Grad-CAM hit rate (%)": hits.get("Grad-CAM", np.nan),
            "deletion AUC, mean (lower = better)": deletion[k].drop("Random")["deletion AUC (mean)"].mean()}
    return pd.DataFrame(rows).T.round(3)
