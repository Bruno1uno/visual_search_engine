---
title: Visual Search Engine
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Visual Search Engine: Metric Learning & Multi-Modal Retrieval

Full-stack Machine Learning project demonstrating an end-to-end vector search pipeline: **Metric Learning Encoder Training -> Offline Zero-Shot Evaluation -> FAISS Indexing -> FastAPI REST API -> React Frontend & Streamlit Dashboard**.


---

## Key Features & Architecture

* **Dataset & Zero-Shot Split:** CUB-200-2011 (11,788 bird images across 200 classes).
  * **Seen Classes (1–100):** Used for model training, validation checkpoint selection, and 1-NN sanity checking.
  * **Unseen Classes (101–200):** Strictly held out zero-shot test set for evaluating embedding space generalization.
* **Metric Learning Models (`src/model.py`, `src/loss.py`):**
  * ResNet34 backbone with custom linear embedding head and L2 unit-norm output vectors.
  * **Proxy Anchor Loss** (Proxy-based loss treating class proxies as anchors).
  * **Triplet Margin Loss** with online hard-negative pair mining (`pytorch_metric_learning.miners`).
* **Multimodal Embedding (`src/clip_encoder.py`):**
  * Frozen **OpenCLIP (ViT-B-32)** vision and text encoders (512D).
* **Vector Indexing (`src/indexer.py`):**
  * High-performance exact nearest neighbor search via **FAISS `IndexFlatIP`** (Cosine Similarity on L2 unit vectors).
  * Separate indices for ResNet and OpenCLIP embeddings.
* **Comprehensive Metrics (`src/evaluate.py`):**
  * Evaluates **Recall@K (K=1, 2, 4, 8)**, **mAP (mean Average Precision)**, **NMI (Normalized Mutual Information)**, 1-NN classification accuracy, Confusion Matrix heatmaps, and t-SNE scatter projections.

---

## Experimental Results (Proxy Anchor vs. Triplet Loss)

Evaluated on the **Unseen Test Split (Classes 101–200)** after Optuna hyperparameter optimization:

| Metric | Proxy Anchor Loss (256D) | Triplet Loss + Hard Mine (256D) | Difference |
| :--- | :--- | :--- | :--- |
| **Seen Test 1-NN Accuracy** | **78.01%** | 73.03% | **+4.98 pp** |
| **Recall@1** | **52.43%** | 44.26% | **+8.17 pp** |
| **Recall@2** | **64.77%** | 56.38% | **+8.39 pp** |
| **Recall@4** | **75.79%** | 67.78% | **+8.01 pp** |
| **Recall@8** | **84.64%** | 78.02% | **+6.62 pp** |
| **mAP** | **0.2615** | 0.2227 | **+0.0388** |
| **NMI** | **0.6037** | 0.5507 | **+0.0530** |

### Benchmark Comparison with Published Literature (CUB-200 Zero-Shot)

| Method / Reference Paper | Backbone | Embedding Dim | Recall@1 | Recall@2 | NMI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[Lifted Structured Feature](https://arxiv.org/abs/1511.06452)** (CVPR 2016) | GoogLeNet | 512D | 47.2% | 58.9% | 56.2% |
| **[Smart Mining](https://arxiv.org/abs/1704.01285)** (ICCV 2017) | Inception-V1 | 64D | 49.8% | 62.3% | 59.9% |
| **[Proxy Anchor Loss](https://arxiv.org/abs/2003.13911)** (CVPR 2020) | Inception-BN | 512D | 68.4% | 79.2% | N/A |
| **My Pipeline: Triplet + Hard Mine** | **ResNet-34** | **256D** | **44.26%** | **56.38%** | **55.07%** |
| **My Pipeline: Proxy Anchor Loss** | **ResNet-34** | **256D** | **52.43%** | **64.77%** | **60.37%** |

> **Analysis:** ResNet-34 Proxy Anchor implementation (256D) achieves 52.43% Recall@1, outperforming classical metric learning baselines (Lifted Structure 47.2%, Smart Mining 49.8%) and demonstrating a +8.17 pp absolute improvement over our own Triplet + Hard Mining baseline (44.26%). This validates the superior convergence and retrieval accuracy of proxy-based loss functions over conventional pair-based hard mining under identical evaluation setups.

---

## Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA support (tested on NVIDIA RTX 3060)

### 2. Environment Activation & Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

> [!NOTE]
> **Automatic Dataset Management:** The CUB-200-2011 dataset does not need to be downloaded or extracted manually. All pipeline CLI scripts (`src/train.py`, `src/hpo.py`, `src/evaluate.py`, `src/indexer.py`, `main.py`) automatically verify the presence of `data/CUB_200_2011/images` and download/extract the dataset on first execution if missing.


---

## Pipeline Execution Commands (How to Run)

All pipeline stages are designed as modular Python CLI scripts that can be executed independently.

### 1. Hyperparameter Optimization (`src/hpo.py`)
Run Optuna HPO studies to find optimal learning rate, margin, alpha, and embedding dimension:
```powershell
# HPO for Proxy Anchor Loss
python -m src.hpo --loss_type proxy_anchor --n_trials 15 --epochs_per_trial 5

# HPO for Triplet Loss
python -m src.hpo --loss_type triplet --n_trials 15 --epochs_per_trial 5
```
*Outputs:* Saves best configs to `configs/best_config_proxy_anchor.yaml` and SQLite database `hpo_study.db`.

---

### 2. Encoder Training (`src/train.py`)
Train the ResNet34 embedding encoder using optimal hyperparameters:
```powershell
# Train Proxy Anchor model
python -m src.train --loss_type proxy_anchor --num_epochs 25 --checkpoint_path checkpoints/best_resnet34_proxy_anchor.pt

# Train Triplet Loss model
python -m src.train --loss_type triplet --num_epochs 25 --checkpoint_path checkpoints/best_resnet34_triplet.pt
```
*Outputs:* Saves model weights to `checkpoints/*.pt` and training loss history to `metrics/history_*.json`.

---

### 3. Offline FAISS Indexing (`src/indexer.py`)
Extract feature embeddings for all 11,788 CUB-200 images and construct FAISS search indices:
```powershell
# Build FAISS indices using trained Proxy Anchor checkpoint
python -m src.indexer --checkpoint_path checkpoints/best_resnet34_proxy_anchor.pt --batch_size 64
```
*Outputs:*
- `indices/resnet_cub200.faiss` (Custom ResNet 256D index)
- `indices/clip_cub200.faiss` (OpenCLIP 512D index)
- `indices/id_to_metadata.json` (Numerical ID to image path and label mapping)

---

### 4. Offline Evaluation & Plotting (`src/evaluate.py`)
Run zero-shot evaluation on unseen test classes (101–200) and sanity check on seen test classes (1–100):
```powershell
# Evaluate Proxy Anchor model
python -m src.evaluate --checkpoint_path checkpoints/best_resnet34_proxy_anchor.pt --loss_name proxy_anchor

# Evaluate Triplet Loss model
python -m src.evaluate --checkpoint_path checkpoints/best_resnet34_triplet.pt --loss_name triplet
```
*Outputs:*
- `metrics/eval_results_proxy_anchor.json` & `metrics/eval_results_triplet.json`
- `plots/confusion_seen_proxy_anchor.png` & `plots/confusion_seen_triplet.png`
- `plots/tsne_seen_proxy_anchor.png` & `plots/tsne_unseen_proxy_anchor.png`
- `plots/tsne_seen_triplet.png` & `plots/tsne_unseen_triplet.png`

---

### 5. FastAPI REST API Serving (`main.py`)
Start the FastAPI vector retrieval backend server:
```powershell
uvicorn main:app --reload --port 8000
```
- Swagger UI / API Documentation: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

---

### 6. Interactive ML Presentation Dashboard (`dashboard.py`)
Launch the Streamlit internal dashboard for metrics inspection and live retrieval queries:
```powershell
streamlit run dashboard.py --server.port 8501
```
- Streamlit UI: `http://localhost:8501`

---

### 7. React Frontend Web Application (`frontend/`)
Launch the modern, minimalist React web client for end-user visual search:
```powershell
cd frontend
npm install
npm run dev
```
- React UI: `http://localhost:3000`
- Detailed frontend architecture and disclaimer: [frontend/README.md](file:///d:/PRG/ML/visual_search_engine/frontend/README.md)

---

### 8. Full-Stack Docker Deployment (`Dockerfile`)
Build and run the entire unified full-stack application (FastAPI backend + React frontend) in a single isolated Docker container:
```powershell
# Build multi-stage Docker image
docker build -t visual-search-engine .

# Run container on port 7860 (Hugging Face Spaces default)
docker run -p 7860:7860 visual-search-engine
```
- Web Application & API: `http://localhost:7860`
- Swagger UI Documentation: `http://localhost:7860/docs`

---

### 9. Automated Test Suite (`pytest`)
Run unit and integration tests across the codebase:
```powershell
# Run fast unit test suite (excludes slow integration tests)
pytest -m "not slow"


# Run full test suite including end-to-end dataset integration test
pytest
```


---

## Repository Structure

```text
visual_search_engine/
├── checkpoints/              # Saved model weights (.pt)
│   ├── best_resnet34_proxy_anchor.pt
│   └── best_resnet34_triplet.pt
├── configs/                  # Optimal hyperparameter configurations (.yaml)
├── data/                     # CUB-200-2011 dataset directory (gitignored)
├── indices/                  # Saved FAISS search indices & metadata
│   ├── resnet_cub200.faiss
│   ├── clip_cub200.faiss
│   └── id_to_metadata.json
├── metrics/                  # Saved evaluation metrics & training history (JSON)
├── plots/                    # Generated t-SNE & confusion matrix plots (PNG)
├── src/
│   ├── dataset.py            # CUB-200 parser, disjoint sampler & dataloaders
│   ├── model.py              # ResNet34 backbone + L2-normalized head + checkpoint loader
│   ├── loss.py               # Triplet Loss + miner & Proxy Anchor Loss
│   ├── clip_encoder.py       # Frozen OpenCLIP ViT-B-32 image & text encoder wrapper
│   ├── train.py              # PyTorch training loop & validation Recall@1 tracking
│   ├── hpo.py                # Optuna hyperparameter tuning CLI
│   ├── evaluate.py           # Evaluation pipeline (Recall@K, mAP, NMI, k-NN)
│   ├── indexer.py            # Offline FAISS index builder
│   ├── inference.py          # Unified visual search engine (ResNet & CLIP retrieval)
│   └── visualization.py      # Plotting utilities (loss curves, confusion matrix, t-SNE)
├── tests/                    # Comprehensive unit and integration test suite
├── frontend/                 # React Web Application (Vite + React single-page UI)
│   ├── src/                  # React components (Header, Search, Grid, Lightbox) & CSS
│   ├── vite.config.js        # Local proxy to FastAPI backend (/api, /static -> :8000)
│   └── README.md             # Frontend documentation & AI attribution disclaimer
├── main.py                   # FastAPI REST API server
├── dashboard.py              # Streamlit ML presentation dashboard
├── pytest.ini
├── requirements.txt
├── AGENTS.md
├── PLAN.md
└── README.md
```

---

## References & Disclaimer

### Benchmark Papers & Datasets
- **Dataset:** [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) - Wah et al. (2011)
- **Proxies:** [*Proxy Anchor Loss for Deep Metric Learning*](https://arxiv.org/abs/2003.13911) - Kim et al. (CVPR 2020) & [*No Fuss Distance Metric Learning using Proxies*](https://arxiv.org/abs/1703.07464) - Hermans et al. (2017)
- **Triplet Loss & Mining:** [*Smart Mining for Deep Metric Learning*](https://arxiv.org/abs/1704.01285) - Harwood et al. (ICCV 2017)
- **Lifted Structured Feature:** [*Deep Metric Learning via Lifted Structured Feature Embedding*](https://arxiv.org/abs/1511.06452) - Song et al. (CVPR 2016)
- **Multimodal Representation:** [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020) - Radford et al. / CLIP (2021)

### Development & Methodology
This project was developed by studying the published metric learning literature above alongside:
- **Official Documentation:** Technical specifications from PyTorch, FAISS Wiki, OpenCLIP, Optuna, FastAPI, and React.
- **Codebase Reuse:** Reusable data loading abstractions adapted from prior personal ML projects.
- **AI Pair-Programming:** Collaboration with an AI agent (**Gemini 3.6 Flash / Antigravity**) for architecture planning, refactoring, test suite generation, and performance optimization.


