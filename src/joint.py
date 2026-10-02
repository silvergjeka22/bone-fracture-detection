"""Our method: one ResNet18 that classifies, finds the fracture box and is taught where to look, all trained together.

loss = cross-entropy (label)
     + detect * CenterNet loss of the built-in detector (radiologist's boxes)
     + point  * Grad-CAM must point inside the radiologist's box
     + agree  * Grad-CAM <-> detector map, and X-ray <-> mirrored X-ray must agree (SimCLR-like: two views, one answer)
     + erase  * fracture blurred -> "not fractured", a random patch blurred -> still "fractured" (the focus is real)
A = cross-entropy only (its detector is YOLO), B = + detect, C = everything. Batches are half fractured.
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import localize
from .data import COLOR, GEOMETRY, MEAN, STD, make_loader
from .models import JointNet
from .training import best_threshold, compute_metrics, predict, save_json, save_model
from .utils import count_parameters, set_seed

STRIDE = 16  # one cell of the layer3 grid = 16 x 16 px
TAU = 0.5    # temperature of the maps turned into probabilities over the cells


def boxes_from_masks(masks):
    """(k, S, S) masks -> (k', 4) boxes around the non-empty ones."""
    out = []
    for m in masks > 0.5:
        rows, cols = torch.nonzero(m, as_tuple=True)
        if len(rows):
            out.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
    return torch.tensor(out, dtype=torch.float32).reshape(-1, 4)


def targets(boxes, size):
    """CenterNet targets on the size/16 grid: centre heatmap (1 at each centre), offset + size there, centre mask."""
    g = size // STRIDE
    heat, reg, centre = torch.zeros(g, g), torch.zeros(4, g, g), torch.zeros(g, g, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    for x0, y0, x1, y1 in boxes.tolist():
        cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
        i, j = min(int(cy), g - 1), min(int(cx), g - 1)
        sigma = max(0.5, ((x1 - x0) * (y1 - y0)) ** 0.5 / STRIDE / 6)
        heat = torch.maximum(heat, torch.exp(-((xx - j) ** 2 + (yy - i) ** 2) / (2 * sigma ** 2)))
        reg[:, i, j] = torch.tensor([cx - j, cy - i, (x1 - x0) / STRIDE, (y1 - y0) / STRIDE])
        centre[i, j] = True
    return heat, reg, centre


class BoxDataset(Dataset):
    """X-ray, label, mask of its boxes grown by `margin` px and the detector targets; flips and rotations move the boxes."""

    def __init__(self, image_set, augment=False, margin=None):
        self.set, self.augment = image_set, augment
        self.size = image_set.images.shape[1]
        self.margin = self.size // 28 if margin is None else margin  # 16 px at 448

    def __len__(self):
        return len(self.set)

    def __getitem__(self, i):
        x = torch.from_numpy(self.set.images[i]).float().div(255)[None]
        boxes = torch.as_tensor(self.set.boxes[i], dtype=torch.float32).reshape(-1, 4)
        if self.augment:
            masks = torch.zeros(len(boxes), self.size, self.size)
            for m, (x0, y0, x1, y1) in zip(masks, boxes.tolist()):
                m[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = 1
            x, masks = GEOMETRY(torch.cat([x, masks])).split([1, len(boxes)])
            x, boxes = COLOR(x).clamp(0, 1), boxes_from_masks(masks)
        mask, m = torch.zeros(1, self.size, self.size), self.margin
        for x0, y0, x1, y1 in boxes.tolist():
            mask[0, max(int(y0) - m, 0):int(np.ceil(y1)) + m, max(int(x0) - m, 0):int(np.ceil(x1)) + m] = 1
        return ((x - MEAN) / STD, int(self.set.labels[i]), mask, *targets(boxes, self.size))


def _fractured(image_set):
    return image_set.subset([i for i in range(len(image_set)) if image_set.labels[i] == 0 and len(image_set.boxes[i])])


def blur_inside(x, mask):
    """Blur the X-ray where mask = 1: thin details such as a fracture line disappear, the rough shape stays."""
    r = x.shape[-1] // 28
    blurred = F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
    return torch.where(mask > 0, blurred, x)


def moved(mask):
    """The same boxes shifted by half the image (wrapping around), minus the real ones: a patch without the fracture."""
    s = mask.shape[-1] // 2
    return torch.roll(mask, (s, s), dims=(-2, -1)) * (1 - mask)


def grad_cam(logits, a, create_graph=True):
    """Grad-CAM of 'fractured' (score fractured - not fractured) on the layer3 grid, before the ReLU: (B, G, G)."""
    g = torch.autograd.grad((logits[:, 0] - logits[:, 1]).sum(), a, create_graph=create_graph)[0]
    return (g.mean((2, 3), keepdim=True) * a).sum(1)


def spread(maps):
    """Maps (B, G, G) -> probabilities over the cells (scaled to unit std, so the size of the map does not matter)."""
    flat = maps.flatten(1)
    flat = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True).clamp_min(1e-6)
    return (flat / TAU).softmax(1)


def js(p, q):
    """Jensen-Shannon divergence between the rows of p and q: 0 = same map, at most log 2."""
    m = (p + q) / 2
    kl = lambda a, b: (a * (a.clamp_min(1e-8).log() - b.clamp_min(1e-8).log())).sum(1)
    return ((kl(p, m) + kl(q, m)) / 2).mean()


def detection_loss(out, heat, reg, centre):
    """CenterNet: focal loss on the centre heatmap + L1 on offset and size at the centres."""
    p = out[:, 0].sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = centre.float()
    n = pos.sum().clamp_min(1)
    focal = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - heat) ** 4 * p ** 2 * (1 - p).log()).sum() / n
    pred = torch.cat([out[:, 1:3].sigmoid(), out[:, 3:5]], 1)
    weight = torch.tensor([1.0, 1.0, 0.1, 0.1], device=out.device)[:, None, None]
    return focal + ((pred - reg).abs() * weight * pos[:, None]).sum() / n


def joint_losses(model, x, y, mask, heat, reg, centre, detect=True, point=True, agree=True, erase=True):
    """Logits and the five losses of a batch (a loss is 0 when it is off or has nothing to use)."""
    a = model.grid(x)
    logits = model.classify(a)
    zero = logits.new_zeros(())
    losses = {"ce": F.cross_entropy(logits, y), "detect": zero, "point": zero, "agree": zero, "erase": zero}
    out = model.detector(a) if detect or agree else None
    if detect:
        losses["detect"] = detection_loss(out, heat, reg, centre)
    box = F.adaptive_max_pool2d(mask, a.shape[-2:])[:, 0]   # grid cells touched by a (grown) box
    f = (y == 0) & (box.flatten(1).sum(1) > 0)               # fractured X-rays with a box
    if point or agree:
        p = spread(grad_cam(logits, a))
        if point and f.any():
            losses["point"] = -(p[f] * box[f].flatten(1)).sum(1).clamp_min(1e-8).log().mean()
        if agree:
            a2 = model.grid(x.flip(-1))                                  # second view: the mirrored X-ray
            losses["agree"] = js(p, spread(grad_cam(model.classify(a2), a2).flip(-1)))
            if f.any():
                losses["agree"] = losses["agree"] + js(p[f], spread(out[f, 0]))  # classifier <-> detector
    if erase and f.any():
        xf, mf = x[f], mask[f]
        logits_e = model(torch.cat([blur_inside(xf, mf), blur_inside(xf, moved(mf))]))
        losses["erase"] = F.cross_entropy(logits_e, torch.cat([torch.ones_like(y[f]), torch.zeros_like(y[f])]))
    return logits, losses


def cam_hit_rate(model, image_set, device, batch_size=32):
    """Share (%) of fractured X-rays whose Grad-CAM peak (layer3 grid) is in a cell of the radiologist's box."""
    frac = _fractured(image_set)
    if len(frac) == 0:
        return float("nan")
    model.eval()
    hits = []
    for x, _, mask, *_ in DataLoader(BoxDataset(frac, margin=0), batch_size=batch_size):
        x, mask = x.to(device), mask.to(device)
        with torch.enable_grad():
            a = model.grid(x)
            cam = grad_cam(model.classify(a), a, create_graph=False)
        box = F.adaptive_max_pool2d(mask, cam.shape[-2:])[:, 0].flatten(1)
        hits += box.gather(1, cam.flatten(1).argmax(1, keepdim=True))[:, 0].bool().tolist()
    return 100 * float(np.mean(hits))


def train_joint(key, train_set, val_set, device, detect=0.0, point=0.0, agree=0.0, erase=0.0, warmup=2, epochs=20,
                batch_size=16, lr=1e-4, weight_decay=1e-4, seed=42, num_workers=2, results_dir=None):
    """Train one version (A / B / C by its loss weights) saved as `key`; val AUC picks the epoch, val F1 the threshold."""
    print(f"=== {key} | detect {detect} | point {point} | agree {agree} | erase {erase} | warm-up {warmup} ===")
    set_seed(seed)
    model = JointNet().to(device)
    labels = train_set.labels
    share = np.where(labels == 0, 0.5 / max((labels == 0).sum(), 1), 0.5 / max((labels != 0).sum(), 1))
    sampler = WeightedRandomSampler(share.tolist(), len(labels), generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(BoxDataset(train_set, augment=True), batch_size=batch_size, sampler=sampler, drop_last=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    val_loader = make_loader(val_set, batch_size=2 * batch_size, num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = dict(detect=detect, point=point, agree=agree, erase=erase)
    history = {k: [] for k in ("ce", *weights, "val_loss", "val_acc", "val_f1", "val_auc", "val_cam_hit", "seconds")}
    best_auc, best_epoch, best_state = -1.0, epochs, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        ramp = 1.0 if warmup == 0 else min(1.0, max(0.0, (epoch - warmup) / warmup))  # XAI losses: 0, then up to 1
        on = {k: w > 0 and (k == "detect" or ramp > 0) for k, w in weights.items()}
        model.train()
        sums, seen = dict.fromkeys(("ce", *weights), 0.0), 0
        for batch in loader:
            x, y, mask, heat, reg, centre = (t.to(device, non_blocking=True) for t in batch)
            _, losses = joint_losses(model, x, y, mask, heat, reg, centre, **on)
            loss = (losses["ce"] + detect * losses["detect"]
                    + ramp * (point * losses["point"] + agree * losses["agree"] + erase * losses["erase"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for k, v in losses.items():
                sums[k] += v.item() * len(y)
            seen += len(y)
        probs, val_labels, val_loss = predict(model, val_loader, device)
        val = compute_metrics(val_labels, probs)
        scheduler.step()
        row = {**{k: v / seen for k, v in sums.items()}, "val_loss": val_loss, "val_acc": val["accuracy"],
               "val_f1": val["f1"], "val_auc": val["auc"], "val_cam_hit": cam_hit_rate(model, val_set, device),
               "seconds": time.time() - start}
        for k, v in row.items():
            history[k].append(float(v))
        print(f"[{epoch:2d}/{epochs}] ce {row['ce']:.3f} detect {row['detect']:.3f} point {row['point']:.3f} "
              f"agree {row['agree']:.3f} erase {row['erase']:.3f} | val AUC {val['auc']:.3f} F1 {val['f1']:.3f} "
              f"Grad-CAM hit {row['val_cam_hit']:.0f}% ({row['seconds']:.0f}s)")
        if best_state is None or val["auc"] > best_auc:
            best_auc, best_epoch, best_state = val["auc"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    probs, val_labels, _ = predict(model, val_loader, device)
    info = {"model": key, "arch": "joint", "history": history, "best_epoch": best_epoch,
            "threshold": best_threshold(val_labels, probs[:, 0]),
            "seconds_per_epoch": float(np.mean(history["seconds"])), "parameters": count_parameters(model),
            "settings": dict(weights, warmup=warmup, batch_size=batch_size, image_size=train_set.images.shape[1])}
    if results_dir is not None:
        save_model(model, key, results_dir)
        save_json(info, Path(results_dir) / "final" / f"{key}.json")
    return model.eval(), info


def best_on_val(final, keys):
    """The version with the best val AUC at its kept epoch (never chosen on the test set)."""
    return max(keys, key=lambda k: final[k]["history"]["val_auc"][final[k]["best_epoch"] - 1])


@torch.no_grad()
def find_boxes(model, images, device, conf=0.01, top=10, batch_size=32):
    """Boxes of the built-in detector: per image a (k, 5) array (x0, y0, x1, y1, score) in pixels, best first."""
    model.eval()
    found = []
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device)
        out = model.detector(model.grid(x))
        heat = out[:, 0].sigmoid()
        peaks = heat * (heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]) 
        for h, o in zip(peaks, out):
            scores, cells = h.flatten().topk(min(top, h.numel()))
            keep = scores >= conf
            i, j = cells[keep] // h.shape[1], cells[keep] % h.shape[1]
            cx, cy = (j + o[1, i, j].sigmoid()) * STRIDE, (i + o[2, i, j].sigmoid()) * STRIDE
            w, hh = o[3, i, j].clamp_min(0) * STRIDE, o[4, i, j].clamp_min(0) * STRIDE
            boxes = torch.stack([cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2], 1).clamp(0, x.shape[-1])
            found.append(torch.cat([boxes, scores[keep, None]], 1).cpu().numpy())
    return found


def detector_scores(found, true_boxes, size):
    """AP@0.5 and the pointing test of the most confident box, on the fractured X-rays (same scoring for every detector)."""
    return {"AP@0.5 (%)": round(100 * localize.average_precision(found, true_boxes), 1),
            **localize.score_boxes(found, true_boxes, (size, size))}


@torch.no_grad()
def erase_test(model, image_set, device, batch_size=32):
    """Mean p(fractured) (%) of the fractured X-rays: as they are, with the fracture blurred, with a random patch blurred."""
    model.eval()
    p = {"as they are": [], "fracture blurred": [], "random patch blurred": []}
    for x, _, mask, *_ in DataLoader(BoxDataset(_fractured(image_set)), batch_size=batch_size):
        x, mask = x.to(device), mask.to(device)
        for kind, view in zip(p, (x, blur_inside(x, mask), blur_inside(x, moved(mask)))):
            p[kind] += model(view).softmax(1)[:, 0].tolist()
    return {kind: 100 * float(np.mean(v)) for kind, v in p.items()}


def comparison(keys, test, localization, deletion, detectors, agreement, erase, labels=None):
    """A vs B vs C on the same test X-rays: classification, detector, where the explanations point, faithfulness."""
    labels = labels or {}
    rows = {}
    for k in keys:
        hits = localization[k].drop("Random")["hit rate (%)"]
        rows[labels.get(k, k)] = {
            "test accuracy (%)": 100 * test[k]["accuracy"], "test F1 (%)": 100 * test[k]["f1"],
            "fractures found (%)": 100 * test[k]["recall"], "test AUC (%)": 100 * test[k]["auc"],
            "detector AP@0.5 (%)": detectors[k]["AP@0.5 (%)"], "detector hit rate (%)": detectors[k]["hit rate (%)"],
            "Grad-CAM hit rate (%)": hits.get("Grad-CAM", np.nan), "best XAI method": hits.idxmax(),
            "best XAI hit rate (%)": hits.max(), "XAI hit rate, mean of 6 (%)": hits.mean(),
            "Grad-CAM inside the detector's box (%)": agreement[k],
            "deletion AUC, mean (lower = better)": deletion[k].drop("Random")["deletion AUC (mean)"].mean(),
            "p(fractured) drop, fracture blurred": erase[k]["as they are"] - erase[k]["fracture blurred"],
            "p(fractured) drop, random patch blurred": erase[k]["as they are"] - erase[k]["random patch blurred"]}
    return pd.DataFrame(rows).T.infer_objects().round(3)
