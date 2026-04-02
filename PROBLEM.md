# Project: Multi-Label License Plate Classification

## 1. What We Were Trying to Do
Train a **ResNet-18** deep learning model to perform **multi-label classification** on license plate images. 
- **Goal**: Detect up to 11 different labels (e.g., `plate_no`, `double_row`, `person_in_car`, etc.) simultaneously for each image.
- **Data**: ~2,800 images with labels stored in a CSV file.

---

## 2. The Cloud Journey (How We Did It)
We transitioned from local training to **Azure Machine Learning** to leverage a powerful **NVIDIA A10 GPU**.

### The Steps We Took:
1.  **Infrastructure**: Attached an existing Azure Linux VM (`CVATGPU`) as an "Attached Compute" resource.
2.  **Authentication**: Used `InteractiveBrowserCredential` to securely log in from VS Code.
3.  **Optimization**: Registered the 2,800 images as an **Azure ML Data Asset** (`tagging_project_data:1`) to avoid re-uploading them every time.
4.  **Environment**: Created a custom **Conda YAML** to ensure the cloud machine had exactly the same libraries as your local laptop.

---

## 3. Roadblocks & Problems Faced

| Problem | Root Cause | Solution |
| :--- | :--- | :--- |
| **TypeError (Verbose)** | PyTorch updated `ReduceLROnPlateau`. | Removed the retired `verbose` parameter. |
| **`ResourceNotFoundError`** | Azure couldn't find the "Curated" env. | Switched to a custom **`conda.yaml`** file. |
| **Unable to establish SSH** | Firewall blocked the Cloud -> VM connection. | Opened **Port 22** in the Azure NSG to Service Tags. |
| **`Quota Exceeded`** | Tried to create a new Managed Cluster. | Updated the script to use **Serverless** or **Attached** compute instead. |
| **"Failed" Compute State** | Azure ML "stuck" on a bad connection. | Detached and Re-attached with a **Fresh Name** (`training-v3`). |

---

## 4. File Structure Explained

```text
├── configs/
│   ├── conda.yaml        <-- The "Recipe" for the cloud machine libraries.
│   └── requirements.txt  <-- Simple list of libraries for Pip.
├── src/
│   ├── train.py          <-- The main training logic (ResNet, Loss, Optim).
│   ├── predict.py        <-- Script to test the model on new images.
│   └── submit_job.py     <-- The "Launcher" that sends work to the cloud.
├── README.md             <-- Basic project intro.
├── .gitignore            <-- Crucial! Prevents pushing 2,800 images to GitHub.
└── PROBLEM.md            <-- This document (Roadmap).
```

---

## 6. The Cloud ML "Masterclass" (Behind the Scenes)

Below is the deep-dive into exactly how our new system works:

### **A. The Model: `train.py` (The Heart)**
- **ResNet-18 Architecture**: We chose a "Residual Network" that is powerful enough to learn image features but small enough to train quickly.
- **The Multi-Label Secret**: Instead of picking one label, we use **Sigmoid** activation. This allows the model to say "Yes/No" to 11 different questions (like "Double row?" or "Person in car?") for every single image.
- **BCEWithLogitsLoss**: This is the specialized "Binary Cross Entropy" loss that supports these multiple independent binary tests.

### **B. The Connection: `submit_job.py` (The Launcher)**
This script acts as the "Secure Bridge":
- **Identity (`InteractiveBrowserCredential`)**: It triggers a browser login to ensure only YOU can launch jobs.
- **The Snapshot**: It zips only your `src/` code folder, keeping the upload size tiny and the launch instant.
- **Entry Point**: It tells Azure precisely how to start the engine: `python train.py`.

### **C. The Environment: `configs/conda.yaml` (The Recipe)**
Unlike a local laptop, cloud machines start "empty". This YAML is the recipe that Azure reads to automatically install **Python, PyTorch, and MLflow** every time a job starts. By using a custom YAML, we are 100% in control of our library versions.

### **D. The Cluster: Managed CPU (`cpu-testing`)**
- **Why it works**: Unlike "Attached" VMs (which need Port 22 SSH), a Managed Cluster is "Cloud-Native". 
- **The Advantage**: Microsoft builds and manages the connection for us "behind the scenes". We never have to worry about firewalls or SSH keys again!

### **E. The Data: Registered Assets**
- **The Local Problem**: Uploading 2,800 images every time you test a small change is very slow.
- **The Asset Solution**: We registered `tagging_project_data:1` in the cloud. Now, Azure "Mounts" that cloud disk directly to your machine at lightning speed. It's essentially a high-speed "Cloud USB Stick".

### **F. The Logs: MLflow**
Even when the training is 10,000 miles away, **MLflow** streams your "Loss" and "Accuracy" curves back to your Azure Studio Dashboard. You can watch your model "learn" in real-time from your browser.
