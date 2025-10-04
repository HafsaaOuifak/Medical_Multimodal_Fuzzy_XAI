# 🧠 Medical Multimodal Fuzzy XAI  
> An explainable AI pipeline that fuses **tabular**, **image**, and **fuzzy surrogate models** for transparent multimodal learning — all inside Docker.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📚 Table of Contents
- [🚀 Step 0: One-Time Preparation](#-step-0-one-time-preparation)
- [⚙️ Step 1: Build & Start Containers](#️-step-1-build--start-containers)
- [🔐 Step 2: Mount Google Drive](#-step-2-mount-google-drive-in-the-container)
- [🧪 Step 3: Sanity Check the Mount](#-step-3-sanity-check-the-mount)
- [📊 Step 4: Train Tabular Models](#-step-4-train-tabular-models)
- [🖼️ Step 5: Train Image Models](#️-step-5-train-image-models)
- [🔗 Step 6: Fuse Multimodal Results](#-step-6-fuse-multimodal-results)
- [🧩 Explainability Options](#-explainability-options)
- [🧱 Developer Workflow](#-developer-workflow)
- [💎 Tips](#-tips)

---

## 🚀 Step 0: One-Time Preparation

1. **Ensure Docker Desktop is running.**  
2. **Share your project folder** in Docker Desktop  
   > *Preferences → Resources → File Sharing*  
3. **Clean previous results** (run before every new experiment):

   ```bash
   rm -rf tabular_models image_models multimodal_results surrogates
   ```

---

## ⚙️ Step 1: Build & Start Containers

```bash
docker compose build
docker compose up -d
```

---

## 🔐 Step 2: Mount Google Drive in the Container

Open a **new terminal** and run (keep it open while working):

```bash
docker exec -it mmp-python bash -lc "bash /workspace/scripts/mount_gdrive.sh"
```

---

## 🧪 Step 3: Sanity Check the Mount

Run inside the container:

```bash
docker exec -it mmp-python bash -lc "echo mounttest > /workspace/tabular_models/hostmount_test.txt"
cat tabular_models/hostmount_test.txt
```

✅ If you see `mounttest`, the mount works.  
❌ If not, check:
- Folder sharing permissions  
- Restart Docker Desktop  
- Rerun the mount script

---

## 📊 Step 4: Train Tabular Models

```bash
docker exec -it mmp-python bash -lc "python -m src.python.train_tabular"
```

**Check outputs:**
```bash
ls tabular_models
ls tabular_models/DecisionTree
```

---

## 🖼️ Step 5: Train Image Models

```bash
docker exec -it mmp-python bash -lc "python -m src.python.train_image"
```

**Check outputs:**
```bash
ls image_models
ls image_models/ResNet50
```

---

## 🔗 Step 6: Fuse Multimodal Results

```bash
docker exec -it mmp-python bash -lc "python -m src.python.fuse_multimodal"
```

**Check outputs:**
```bash
ls multimodal_results
cat multimodal_results/fused_metrics.csv
```

---

## 🧩 Explainability Options

**Explain with all features (default):**
```bash
docker exec -it mmp-python bash -lc "python -m src.python.explain --id 23"
```

**Explain using only the top-10 surrogate features:**
```bash
docker exec -it mmp-python bash -lc "python -m src.python.explain --id 23 --top_k 10"
```

**Compare XAI methods:**
```bash
docker exec -it mmp-python bash -lc "python -m src.python.compare_xai --id 23"
```

---

## 🧱 Developer Workflow

Whenever you modify code or configuration files:

```bash
# Check what changed
git status

# Stage all modified files
git add .

# Commit with a clear message
git commit -m "Update <describe what you changed>"

# Push to GitHub
git push
```

If the change affects Docker:
```bash
docker compose up --build -d
```

---

## 💎 Tips

- 🧹 **Always delete old result folders** before new runs.  
- 🧠 **Keep Docker Desktop updated.**  
- 🪶 **Monitor containers:**
  ```bash
  docker compose logs -f
  ```
- 🐚 **Enter container shell manually:**
  ```bash
  docker exec -it mmp-python bash
  ```


