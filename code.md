# 🧠 ML Model Training — Linux Terminal Command Reference
**License Plate Multi-Label Classifier | EfficientNet-B0 | Azure ML**

---

## 📁 1. Navigation & File Management

```bash
# Go into project directory
cd ML-Model-1/ML-Model-1/

# List all files in current folder
ls

# List files with size and details
ls -lh

# Check file content (read-only, scrollable)
less train.log

# View last N lines of a file
tail -20 train.log

# Monitor a file live as it updates
tail -f train_50.log

# Exit live monitoring
# Press: Ctrl + C
```

---

## 🔍 2. Searching Inside Files

```bash
# Search for a keyword in a file (shows line number)
grep -n "epoch" train.py

# Search for multiple keywords at once
grep -n "CLASSES\|snow" train.py

# Search for optimizer and scheduler lines
grep -n "optimizer\|scheduler\|lr" train.py

# Check how many lines match a pattern
grep -c ",1," data/cleaned_dataset.csv
```

---

## ✏️ 3. Editing Files with Nano

```bash
# Open a file in nano editor
nano train.py

# Open file and jump directly to a line number
nano +43 train.py

# Inside nano — useful shortcuts:
# Ctrl + W       → Search for text
# Ctrl + _       → Go to specific line number
# Ctrl + O       → Save file (Write Out)
# Enter          → Confirm save
# Ctrl + X       → Exit nano
# Page Down      → Scroll down
# Page Up        → Scroll up
# Ctrl + V       → Scroll down (alternative)
# Ctrl + Y       → Scroll up (alternative)
```

---

## 🚀 4. Running the Training Script

```bash
# Basic run (foreground — blocks terminal)
python3 train.py

# Run with custom epochs
python3 train.py --epochs 50

# Run in background using nohup (keeps running after terminal closes)
nohup python3 train.py --epochs 50 > train_50.log 2>&1 &

# What each part means:
# nohup          → Don't stop if terminal closes
# python3        → Use python3 interpreter
# train.py       → Script to run
# --epochs 50    → Pass argument: 50 epochs
# > train_50.log → Save all output to this log file
# 2>&1           → Also save errors to same log file
# &              → Run in background

# Save the Process ID for later use
nohup python3 train.py --epochs 50 > train_50.log 2>&1 &
echo $! > train.pid
```

---

## 📊 5. Monitoring Training Progress

```bash
# Watch training log live
tail -f train_50.log

# Check if training is still running
ps aux | grep train.py

# Stop monitoring (training keeps running)
# Press: Ctrl + C

# Kill the training process if needed
kill <PID>
# Example:
kill 164322
```

---

## 🔧 6. Checking System & GPU

```bash
# Check available python
which python3

# Check if Azure CLI is installed
az --version

# Check GPU usage during training
nvidia-smi

# Check disk space
df -h

# Check file sizes in outputs folder
ls -lh outputs/
```

---

## ☁️ 7. Installing Azure CLI

```bash
# Install Azure CLI on Ubuntu VM
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure account
az login
# → Opens browser link + code to authenticate
```

---

## 💾 8. Uploading to Azure Blob Storage

```bash
# Upload entire outputs folder to Blob Storage
az storage blob upload-batch \
  --account-name aakashml3430935648 \
  --destination vmcopy/Mini-model \
  --source outputs/

# Upload a single file to Blob Storage
az storage blob upload \
  --account-name aakashml3430935648 \
  --container-name vmcopy \
  --name Mini-model/train.log \
  --file train.log

# Upload multiple specific files
az storage blob upload \
  --account-name aakashml3430935648 \
  --container-name vmcopy \
  --name Mini-model/train_50.log \
  --file train_50.log

# List all blobs in a container/folder
az storage blob list \
  --account-name aakashml3430935648 \
  --container-name vmcopy \
  --prefix Mini-model \
  --output table

# Upload using batch (all outputs at once)
az storage blob upload-batch \
  --account-name aakashml3430935648 \
  --destination vmcopy/Mini-model \
  --source outputs/
```

---

## 🤖 9. Submitting Job to Azure ML

```bash
# Submit training job via job.yml
az ml job create -f job.yml

# Check job status
az ml job list --output table
```

---

## 📝 10. Model Changes Made

### Remove snow class (0 samples in dataset)
```python
# In train.py — Line 43, CLASSES list
# Before:
"blurred", "rain", "fog", "snow", "night",

# After:
"blurred", "rain", "fog", "night",
```

### Remove snow pos_weight (Line 120)
```python
# Before:
0.0,      # snow   — no labeled samples, loss zeroed out

# After:
# (line deleted entirely)
```

### LR Scheduler (already present in train.py)
```python
# Line 243-244 — already configured
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)
# Line 294 — already being called
scheduler.step(val_loss)
```

---

## 🔁 11. Full Workflow Summary

```
1. SSH into VM
2. cd ML-Model-1/ML-Model-1/
3. nano +43 train.py          → make changes
4. nohup python3 train.py --epochs 50 > train_nosnow.log 2>&1 &
5. tail -f train_nosnow.log   → monitor training
6. Wait for "Done ✓"
7. az storage blob upload-batch → upload outputs to Blob
```

---

## 📦 12. Key Files Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `job.yml` | Azure ML job config |
| `data/cleaned_dataset.csv` | Dataset CSV |
| `data/images/` | Image folder |
| `outputs/best_model.pth` | Best saved model |
| `outputs/thresholds.json` | Per-class thresholds |
| `outputs/training_curves.png` | Loss/F1 plots |
| `outputs/model_info.json` | Model metadata |
| `train.log` | 30 epoch run log |
| `train_50.log` | 50 epoch run log |
| `train_nosnow.log` | Snow-removed run log |

---

*Storage Account: `aakashml3430935648` | Container: `vmcopy` | Folder: `Mini-model`*
