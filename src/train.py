import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
import json

# ─────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=False, default="data")
parser.add_argument("--epochs",    type=int,   default=15)
parser.add_argument("--lr",        type=float, default=1e-4)
parser.add_argument("--batch_size",type=int,   default=32)
args = parser.parse_args()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CLASSES = [
    "complex1","complex2","complex3","complex4",
    "blurred","rain","fog","night",
    "occluded","truncated","lp_blurred"
]
# Note: 'snow' and 'no_annotation' were removed as per requirements.

NUM_LABELS = len(CLASSES)
IMG_DIR    = "data/raw/images"
CSV_PATH   = "data/processed/cleaned_dataset.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class LPDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])
        img      = Image.open(img_path).convert("RGB")
        labels   = torch.tensor(row[CLASSES].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {CSV_PATH}. Check your paths!")

    df = pd.read_csv(CSV_PATH)

    assert all(c in df.columns for c in CLASSES), f"Label mismatch! Missing columns: {[c for c in CLASSES if c not in df.columns]}"

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    split = int(0.8 * len(df))
    train_df = df[:split]
    val_df   = df[split:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(
        LPDataset(train_df, IMG_DIR, train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        LPDataset(val_df, IMG_DIR, val_transform),
        batch_size=args.batch_size, num_workers=2
    )


    WEIGHTS_PATH = "data/processed/pos_weights.npy"
    if os.path.exists(WEIGHTS_PATH):
        pos_weight_vals = np.load(WEIGHTS_PATH)
        if len(pos_weight_vals) == len(CLASSES):
            print(f"Loading pos_weights from {WEIGHTS_PATH}...")
            pos_weight = torch.tensor(pos_weight_vals, dtype=torch.float32).to(DEVICE)
        else:
            print(f"Warning: {WEIGHTS_PATH} shape {pos_weight_vals.shape} does not match CLASSES {len(CLASSES)}. Recomputing...")
            pos = df[CLASSES].sum().replace(0, 1)
            neg = len(df) - pos
            pos_weight = torch.tensor((neg / pos).values, dtype=torch.float32).to(DEVICE)
    else:
        print(f"Warning: {WEIGHTS_PATH} not found. Recomputing weights...")
        pos = df[CLASSES].sum().replace(0, 1)
        neg = len(df) - pos
        pos_weight = torch.tensor((neg / pos).values, dtype=torch.float32).to(DEVICE)

    print("\nClass distribution:")
    pos_counts = df[CLASSES].sum()
    for c, p in zip(CLASSES, pos_counts.values):
        print(f"  {c:<20}: {int(p)} positives")


    # MODEL (EfficientNet-B0)

    model = models.efficientnet_b0(weights='DEFAULT')

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last feature block + classifier
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.features[-2].parameters():
        param.requires_grad = True

    # Replace head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_LABELS)
    model = model.to(DEVICE)


    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    # ─────────────────────────────────────────
    # TRAIN LOOP
    # ─────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    best_val_loss = float("inf")

    mlflow.start_run()
    mlflow.log_params({
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "model": "efficientnet_b0",
        "num_labels": NUM_LABELS
    })

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_labels= []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs)
                all_preds.append((probs > 0.5).int().cpu().numpy())
                all_labels.append(labels.int().cpu().numpy())

        val_loss  /= len(val_loader)
        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        pre = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"F1: {f1:.4f} | P: {pre:.4f} | R: {rec:.4f}")

        # ── Log Metrics ──
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "f1_macro":   f1,
            "precision":  pre,
            "recall":     rec
        }, step=epoch)

        # ── LR Scheduler ──
        scheduler.step(val_loss)

        # ── Save best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")

    # ─────────────────────────────────────────
    # PER-CLASS THRESHOLD TUNING
    # ─────────────────────────────────────────
    print("\nTuning per-class thresholds on validation set...")
    model.load_state_dict(torch.load("outputs/best_model.pth"))
    model.eval()

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.int().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    thresholds = {}
    for i, cls in enumerate(CLASSES):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs[:, i] > t).astype(int)
            f1    = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[cls] = round(best_t, 2)
        print(f"  {cls:<20}: threshold={best_t:.2f}  F1={best_f1:.4f}")

    # Save thresholds
    with open("outputs/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    mlflow.log_artifact("outputs/thresholds.json")
    mlflow.log_artifact("outputs/best_model.pth")
    mlflow.end_run()

    print("\nDone. Files in outputs/:")
    print("  best_model.pth")
    print("  thresholds.json")

