"""
STAGE 5 — CPU-friendly CNN transfer learning on BUSI.

Design choices, all driven by the "no GPU" constraint:
  - pretrained ImageNet ResNet18 (features already learned, no from-scratch training)
  - convolutional backbone FROZEN (no gradient computed through it -> most of
    the compute cost disappears)
  - only the final Linear layer is trained (a few thousand parameters, trains
    in seconds/minutes per epoch on CPU even on a laptop)
  - small image size (160x160) and small batch size to keep memory/CPU light
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["benign", "malignant", "normal"]
IMG_SIZE = 160  # smaller than the usual 224 -> noticeably faster on CPU


# ---------------------------------------------------------------------------
# Dataset discovery (handles BUSI's "<label> (<n>).png" + "<label> (<n>)_mask.png")
# ---------------------------------------------------------------------------
def discover_busi(root="data/BUSI"):
    items = []
    for label in CLASSES:
        folder = os.path.join(root, label)
        for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
            if "_mask" in os.path.basename(path).lower():
                continue  # never treat a mask file as an input image
            mask_path = path.replace(".png", "_mask.png")
            mask_path = mask_path if os.path.exists(mask_path) else None
            items.append({"path": path, "mask": mask_path, "label": label})
    return items


def stratified_split(items, val_frac=0.15, test_frac=0.15, seed=42):
    labels = [it["label"] for it in items]
    train_val, test = train_test_split(
        items, test_size=test_frac, stratify=labels, random_state=seed
    )
    tv_labels = [it["label"] for it in train_val]
    val_size = val_frac / (1 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=tv_labels, random_state=seed
    )
    return train, val, test


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------
class BusiDataset(Dataset):
    def __init__(self, items, train=False):
        self.items = items
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["path"]).convert("RGB")
        img = self.tf(img)
        label = CLASSES.index(it["label"])
        return img, label


def class_weights(items):
    counts = np.array([sum(1 for it in items if it["label"] == c) for c in CLASSES])
    weights = counts.sum() / (len(counts) * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model: pretrained ResNet18, frozen backbone, trainable head
# ---------------------------------------------------------------------------
def build_model(num_classes=len(CLASSES)):
    # NOTE: weights download requires internet access to download.pytorch.org.
    # Set env var CNN_NO_PRETRAINED=1 only for offline sandbox testing where
    # that host is blocked -- on your own machine, leave this unset so you
    # get REAL ImageNet-pretrained weights (required by the assignment).
    weights = None if os.environ.get("CNN_NO_PRETRAINED") else ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False  # freeze the ENTIRE pretrained backbone
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # only this layer trains
    return model


def train_model(model, train_items, val_items, epochs=8, batch_size=16, lr=1e-3,
                 patience=3, ckpt_path="models/cnn_best.pt", seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    train_loader = DataLoader(BusiDataset(train_items, train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BusiDataset(val_items, train=False), batch_size=batch_size, shuffle=False)

    weights = class_weights(train_items)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)  # only fc has grads anyway

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss, train_acc = run_loss / total, correct / total

        model.eval()
        vrun_loss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                vrun_loss += loss.item() * x.size(0)
                vcorrect += (out.argmax(1) == y).sum().item()
                vtotal += x.size(0)
        val_loss, val_acc = vrun_loss / vtotal, vcorrect / vtotal

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(torch.load(ckpt_path))
    return model, history


def evaluate_model(model, test_items, batch_size=16):
    loader = DataLoader(BusiDataset(test_items, train=False), batch_size=batch_size, shuffle=False)
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            probs = torch.softmax(out, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(out.argmax(1).tolist())
            y_prob.extend(probs.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)
