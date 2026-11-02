"""Reference RGB + target-mask baseline. Supply your own local VINS download."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from evaluate import summarize, write_scores
from validate_release import ROOT, load_release, require


class Components(Dataset):
    def __init__(self, keys, screens, boxes, labels, vins, size):
        self.keys, self.screens, self.boxes = keys, screens, boxes
        self.labels, self.vins, self.size = labels, vins, size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        with Image.open(self.vins / self.screens[k[0]]["vins_image_path"]) as src:
            resized = src.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        l, t, r, b = (float(self.boxes[k][f]) * self.size for f in ("x_min", "y_min", "x_max", "y_max"))
        mask = np.zeros((1, self.size, self.size), dtype=np.float32)
        mask[0, math.floor(t):math.ceil(b), math.floor(l):math.ceil(r)] = 1.
        x = torch.from_numpy(np.concatenate((rgb, mask), axis=0))
        y = torch.tensor(float(self.labels[k]["rating_final"]) / 5., dtype=torch.float32)
        return x, y


def make_model(imagenet):
    net = resnet18(weights=ResNet18_Weights.DEFAULT if imagenet else None)
    old = net.conv1
    net.conv1 = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    if imagenet:
        with torch.no_grad():
            net.conv1.weight[:, :3].copy_(old.weight)
            net.conv1.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
    else:
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vins-root", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, help="new file for held-out predictions (0–5 scale)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imagenet", action="store_true", help="use torchvision ImageNet weights; may download weights")
    p.add_argument("--smoke", action="store_true", help="one training batch and one held-out batch, no output file")
    args = p.parse_args()
    require(args.epochs > 0 and args.batch_size >= 2 and args.image_size >= 64 and args.lr > 0,
            "invalid training parameters")
    require(args.smoke or (args.output and not args.output.exists()), "provide a new --output file")
    screens, boxes, labels = load_release(args.data, args.vins_root)
    keys, scores = sorted(boxes), {}
    for fold in range(5):
        random.seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        torch.manual_seed(args.seed + fold)
        train = [k for k in keys if int(screens[k[0]]["reference_fold"]) != fold]
        test = [k for k in keys if int(screens[k[0]]["reference_fold"]) == fold]
        require({k[0] for k in train}.isdisjoint(k[0] for k in test), "screen leaked across folds")
        if args.smoke:
            train, test = train[:2], test[:2]
        def dataset(ks):
            return Components(ks, screens, boxes, labels, args.vins_root, args.image_size)
        train_loader = DataLoader(dataset(train), batch_size=args.batch_size, shuffle=True)
        model = make_model(args.imagenet).to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        for _ in range(1 if args.smoke else args.epochs):
            for x, y in train_loader:
                opt.zero_grad()
                pred = model(x.to(args.device)).flatten()
                loss = nn.functional.mse_loss(pred, y.to(args.device))
                require(torch.isfinite(loss).item(), "nonfinite training loss")
                loss.backward()
                opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in DataLoader(dataset(test), batch_size=args.batch_size):
                preds.extend((5.0 * model(x.to(args.device)).flatten()).cpu().tolist())
        require(len(preds) == len(test) and all(map(math.isfinite, preds)), "invalid held-out predictions")
        scores.update(zip(test, preds))
        if args.smoke:
            print("OK: trained on one batch and predicted on a disjoint UI fold; no benchmark calculated")
            return
    write_scores(args.output, scores, screens)
    print(json.dumps(summarize(scores, labels, screens), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
