# PLAN.md – Metric Learning & Multi-Modal Retrieval Pipeline

## Project Overview
The goal of the project is to build a production-clean **Full ML pipeline (Train → Evaluation → Indexing → Inference → Serving)** focused on **Metric Learning** and **Multi-Modal Retrieval** (Image-to-Image & Text-to-Image search).

* **Dataset:** CUB-200-2011 (11,788 images, 200 bird classes).
* **Split:** Disjoint class split – Classes 1–100 (Train/Val + Seen Test subset), Classes 101–200 (Unseen Test set for Zero-Shot evaluation).
* **Hardware:** RTX 3060 12GB, PyTorch, CUDA.

---

## Architecture and Key Interfaces

```text
               ┌────────────────────────────────────────────────────────┐
               │                OFFLINE STAGE (Train & Indexing)        │
               └────────────────────────────────────────────────────────┘
                                            │
   ┌────────────────────────┐    ┌──────────┴───────────┐    ┌────────────────────────┐
   │  1. ResNet Training    │    │ 2. Evaluation & Plots│    │ 3. Indexer (FAISS)     │
   │  (Triplet + Hard Mine) │───>│  (Recall@K, t-SNE)   │    │ (ResNet 128D & CLIP)  │
   └────────────────────────┘    └──────────────────────┘    └───────────┬────────────┘
                                                                         │
                                                          Saved indices (.faiss) + metadata (.json)
                                                                         │
               ┌────────────────────────────────────────────────────────┴┐
               │                 ONLINE STAGE (Serving & User UI)        │
               └─────────────────────────────────────────────────────────┘
                                            │
                                 ┌──────────┴───────────┐
                                 │   src/inference.py   │  <-- Stable Interface (Engine)
                                 └──────────┬───────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      │                                           │
           ┌──────────┴──────────┐                     ┌──────────┴──────────┐
           │  FastAPI (main.py)  │                     │  Streamlit (dev)    │
           │  REST API           │                     │  Metrics Dashboard  │
           └──────────┬──────────┘                     └─────────────────────┘
                      │
           ┌──────────┴──────────┐
           │  React Frontend     │
           │  (Product Web UI)   │
           └──────────┬──────────┘
```

---

## Development Phases (Detailed Step by Step)

### Phase 1: Data Pipeline & Disjoint Split (`src/dataset.py`)
- [x] **Dataset Downloader/Parser:** Script to download and extract CUB-200-2011.
- [x] **Seen classes (1–100)** divided into three disjoint image subsets:
  - `train_dataset` (~70% of images) — model training
  - `val_dataset` (~10% of images) — checkpoint selection during training (validation Recall@1 after each epoch)
  - `seen_test_dataset` (~20% of images) — final sanity-check evaluation AFTER training (k-NN accuracy, confusion matrix).
- [x] **Unseen test classes (101–200)** — all images form `unseen_test_dataset` for zero-shot evaluation.
- [x] **Sampler for Triplet Mining:** `MPerClassSampler` from `pytorch_metric_learning.samplers` ($M=4$ images per class per batch).
- [x] **Data Augmentations:** Train transforms (Resize, RandomCrop, RandomHorizontalFlip, ColorJitter) and Eval transforms (Resize, CenterCrop).

---

### Phase 2: Model, Loss Functions & Offline Training (`src/model.py`, `src/loss.py`, `src/train.py`)
- [x] **Model Backbone (`src/model.py`):**
  - ResNet34 (or ResNet18/50) pretrained on ImageNet (`models.ResNet34_Weights.DEFAULT`).
  - Replace classification layer `model.fc` with `nn.Identity()` and custom head: `nn.Linear(in_features, 128)`.
  - **L2 Normalization:** Apply `F.normalize(x, p=2, dim=1)` to the output 128D vector in the forward pass.
- [] **Loss Functions & Mining (`src/loss.py`):**
  - [x] **Baseline:** `pytorch_metric_learning.losses.TripletMarginLoss` + `miners.TripletMarginMiner` (hard/semi-hard pair mining).
  - [] **Proxy Variant:** `pytorch_metric_learning.losses.ProxyAnchorLoss(num_classes=100, embedding_size=128)` (proxy-based loss without miner, treating class proxies as anchors).
- [ ] **Data Loader Parameterization (`src/dataset.py`):**
  - Add `use_m_per_class_sampler: bool = True` to `get_cub200_dataloaders()`.
  - `True` for Triplet Loss (`MPerClassSampler`), `False` for Proxy Anchor (`shuffle=True` random sampler).
- [ ] **Training Loop (`src/train.py`):**
  - Optimizer: AdamW (must include both `model.parameters()` and `loss_func.parameters()` when using Proxy Anchor).
  - CLI flag `--loss_type` (`triplet` | `proxy_anchor`).
  - Checkpointing: Saving distinct weights (`best_resnet34_triplet.pt` vs. `best_resnet34_proxy_anchor.pt`) based on validation Recall@1.
  - History logging (`history_triplet.json`, `history_proxy.json`).

---

### Phase 3: Offline Evaluation, FAISS Indexer & Metrics (`src/evaluate.py`, `src/indexer.py`)

#### A. Offline FAISS Indexer (`src/indexer.py`)
- **Dataset Pass:** Extraction of 128D embeddings from all CUB-200 images using the trained ResNet, and 512D embeddings using frozen CLIP (`clip-ViT-B-32`).
- **FAISS Indexing:**
  - Creation of two separate FAISS indices:
    1. `resnet_index.faiss`: `faiss.IndexFlatIP(128)` (Inner Product for L2-normalized vectors = Cosine Similarity).
    2. `clip_index.faiss`: `faiss.IndexFlatIP(512)` (for CLIP embeddings).
  - `index.add(embeddings)` adds vectors under numerical IDs ($0 \dots N-1$).
- **Save to Disk:**
  - `indices/resnet_cub200.faiss`
  - `indices/clip_cub200.faiss`
  - `indices/id_to_metadata.json`: Dictionary mapping `int_id -> { "image_path": "...", "class_id": 105, "class_name": "black_footed_albatross" }`.

#### B. Offline Evaluation & Plot Generation (`src/evaluate.py`)
- **Zero-Shot Evaluation on Unseen Classes (101–200):**
  - **Recall@K (K=1, 2, 4, 8):** Iterates through each test query image, fetches Top-K results from FAISS, and checks if at least 1 result has the same class.
  - **mAP (mean Average Precision):** Evaluation of relevant image ranking order.
  - **NMI (Normalized Mutual Information):** K-Means clustering on embeddings vs. actual class labels.
- **Sanity Check on Seen Test Subset (1–100):**
  - k-NN classification accuracy and Confusion Matrix (saved as a PNG image).
- **Ablation Study Experiments:**
  1. **Mining Strategy Ablation:** Hard Negative Mining vs. Random Triplet Sampling.
  2. **Loss Function Ablation:** Triplet Loss + Hard Mining vs. Proxy Anchor Loss (comparing Recall@K, mAP, NMI on unseen classes).
- **Embedding Space Visualization:**
  - Computation of t-SNE / UMAP projections (pre-training vs post-training, seen vs unseen, Triplet vs Proxy Anchor).
- **Output Storage:**
  - All metrics saved to `metrics/eval_results.json` and `metrics/ablation.json`.
  - Plots saved to `plots/tsne_unseen.png`, `plots/confusion_seen.png`.

---

### Phase 4: Serving Engine & FastAPI REST API (`src/inference.py`, `main.py`)
- **Stable Engine (`src/inference.py`):**
  - Loads trained ResNet model, CLIP model, FAISS indices, and `id_to_metadata.json` upon initialization.
  - Method `search_by_image(image_bytes, engine_type='resnet'|'clip', top_k=8)`:
    1. Computes embedding of uploaded image.
    2. Calls `faiss_index.search(query_vec, top_k)`.
    3. Maps retrieved numerical indices via `id_to_metadata.json` to image paths and similarity scores.
  - Method `search_by_text(text_query, top_k=8)`:
    1. Computes text embedding using CLIP text encoder.
    2. Calls `clip_faiss_index.search(text_vec, top_k)`.
    3. Returns list of most relevant images.
  - *Important:* Uploaded query image is **NOT SAVED** into FAISS index (FAISS index is a read-only database).
- **FastAPI REST API (`main.py`):**
  - Endpoints:
    - `POST /api/search/image` (form-data: file, engine_type, top_k)
    - `POST /api/search/text` (json: text_query, top_k)
    - `GET /api/metrics` (returns precomputed data from `metrics/eval_results.json`)
    - Static file serving to display source images of CUB-200 dataset.
  - `CORSMiddleware` configuration for React frontend communication.

---

### Phase 5: Internal Dev Dashboard (Streamlit) & Product Web UI (React)

#### A. Internal ML Dashboard (`dashboard.py` - Streamlit)
- Internal tool for presenting development results.
- Displays precomputed data from Phase 3:
  - Training progress (loss curves).
  - Recall@K, mAP, NMI table on unseen classes.
  - Ablation study table (Random vs. Hard Mining).
  - t-SNE projection images and Confusion Matrix.

#### B. User-Facing Product Web UI (`frontend/` - React)
- Clean, modern SPA interface.
- Two main tabs / modes:
  1. **Image Similarity Search:** Drag-and-drop image upload + model toggle (Custom Metric ResNet vs. CLIP Visual) + slider for Top-K. Results displayed in a grid with similarity scores.
  2. **Text-to-Image Search:** Text search input (e.g., *"yellow bird with black wings"*) -> query to CLIP text encoder -> display retrieved images.

---

## Summary of Metrics & File Infrastructure

| File | Purpose |
| :--- | :--- |
| `data/CUB_200_2011/` | Downloaded CUB-200-2011 dataset |
| `checkpoints/best_resnet34_metric.pt` | Saved weights of trained encoder |
| `indices/resnet_cub200.faiss` | FAISS index (128D) |
| `indices/clip_cub200.faiss` | FAISS index (512D) |
| `indices/id_to_metadata.json` | FAISS ID mapping to image path and label |
| `metrics/eval_results.json` | Saved computed metrics (Recall@K, mAP, NMI) |
| `plots/` | Generated plots (t-SNE, loss, confusion matrix) |
