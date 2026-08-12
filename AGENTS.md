# Metric Learning & Multi-Modal Retrieval – Project Context

## Project Goal
Portfolio ML engineering project demonstrating a full pipeline: **Metric learning encoder training -> Offline evaluation on unseen classes -> FAISS indexing -> FastAPI serving -> React frontend + Streamlit dashboard**.

The goal is to learn to work with embedding spaces, nearest neighbor search (Vector Search), and multi-modal retrieval (Text-to-Image via CLIP).

---

## Dataset & Split (CRITICAL RULE)
- **Dataset:** CUB-200-2011 (200 bird classes, ~11,788 images).
  - **Raw CUB-200-2011 Directory Structure:**
    ```text
    data/CUB_200_2011/
    ├── images/                               # 200 bird class subdirectories
    │   ├── 001.Black_footed_Albatross/
    │   └── ...
    ├── classes.txt                           # class_id <-> class_name ("1 001.Black_footed_Albatross")
    ├── images.txt                            # image_id <-> rel_path ("1 001.Black_footed_Albatross/...")
    └── image_class_labels.txt                # image_id <-> class_id ("1 1")
    ```
  - **Metadata Parsing:** `parse_cub200_metadata()` performs a relational JOIN on `classes.txt`, `images.txt`, and `image_class_labels.txt` to produce a clean `list[dict]` of image records (`image_id`, `rel_path`, `abs_path`, `class_id`, `class_name`).
- **Disjoint Class Split (Zero-Shot Setup):**
  - **Classes 1–100 (Seen classes):** Used for training (`train_dataset`), validation, and a small subset for `seen_test_dataset` (for k-NN & Confusion Matrix sanity check).
  - **Classes 101–200 (Unseen test classes):** The model NEVER sees these classes during training. Search and evaluation (Recall@K, mAP, NMI) on these classes demonstrate embedding space generalization.
- **Hardware:** RTX 3060 12GB, PyTorch, CUDA.

---

## Architecture and Rules
1. **Two Separate FAISS Indices:**
   - ResNet custom embedding (128D) and CLIP embedding (512D) have **different geometries and dimensions**.
   - They are never mixed in a single index! `indices/resnet_cub200.faiss` and `indices/clip_cub200.faiss` are created separately.
2. **Offline vs. Online Separation:**
   - **FAISS indexing and metrics evaluation happen OFFLINE** (scripts `src/indexer.py` and `src/evaluate.py`).
   - Metric results and plots are saved to `metrics/` and `plots/`.
   - **FastAPI loads saved `.faiss` indices and `id_to_metadata.json` into RAM only once upon startup**.
   - Uploaded query images from the frontend are **NOT SAVED** into the FAISS index – searching is a read-only operation.
3. **Stable Inference Engine (`src/inference.py`):**
   - FastAPI server (`main.py`) and Gradio/Streamlit do not perform mathematical operations with FAISS directly in controllers.
   - They call a clean interface `src/inference.py`:
     ```python
     def search_by_image(image_bytes: bytes, engine_type: str = "resnet", top_k: int = 8) -> list[dict]: ...
     def search_by_text(text_query: str, top_k: int = 8) -> list[dict]: ...
     ```
4. Important: Confusion matrix and k-NN classification accuracy are calculated ONLY on the seen test split (classes 1–100, unseen images), never on unseen (101–200), because a fixed label set on both sides makes no sense there.

---

## Repository Structure
```text
visual_search_engine/
├── data/                      # CUB-200-2011 dataset (gitignored)
├── checkpoints/               # Trained ResNet encoder weights (.pt)
├── indices/                   # FAISS indices (.faiss) and metadata mapping (.json)
├── metrics/                   # Saved metric results (JSON)
├── plots/                     # Generated plots (t-SNE, loss, confusion matrix)
├── src/
│   ├── dataset.py             # CUB loader, disjoint sampler (MPerClassSampler)
│   ├── model.py               # ResNet34 backbone + L2-normalized 128D head + optionally ResNet18 ...
│   ├── loss.py                # Triplet loss with online batch-hard mining
│   ├── train.py               # Training loop, validation Recall@1 tracking
│   ├── evaluate.py            # Computation of Recall@K, mAP, NMI, t-SNE and Ablation study
│   ├── indexer.py             # Offline FAISS index generation
│   └── inference.py           # Stable retrieval engine for API
├── main.py                    # FastAPI REST API server
├── dashboard.py               # Streamlit internal ML presentation dashboard
├── frontend/                  # React Web Application (Product UI)
├── requirements.txt
├── PLAN.md
├── AGENTS.md
└── README.md
```

## Testing

Tests are written continuously alongside implementation, not at the end. Mirroring the `src/` structure:

| Test | What it verifies | Why it is critical |
| :--- | :--- | :--- |
| `test_dataset.py` | No class overlap between seen (1–100)/unseen (101–200); no file overlap between train/val/seen_test | Error here = entire Phase 3 evaluation generates nonsensical numbers without being obvious at first glance |
| `test_sampler.py` | `MPerClassSampler` actually returns P classes × M instances per batch, not a random distribution | Without this test, hard-mining silently "turns off" (missing positive pairs) and no one notices |
| `test_model.py` | Output embedding has an L2 norm of exactly 1.0 | Normalization sanity check - if it fails, cosine similarity in FAISS stops matching reality |
| `test_train.py` | Smoke test: single training step executes without error, loss is a finite number (not NaN/inf) | Cheap test, catches broken gradient flow right at the start |
| `test_indexer.py` | Number of vectors in FAISS index == number of items in `id_to_metadata.json` | Most common real-world error: desynchronized index and mapping after rebuild |
| `test_inference.py` | `search_by_image`/`search_by_text` returns top_k results in the correct format, correctly distinguishes `engine_type` | Verifies the core stable API that all serving relies on |
| `test_serving.py` | FastAPI endpoints return 200 and expected JSON shape | Basic check that the serving layer didn't break during refactoring |

---

## How to Help Me (Communication and Coding Rules)
- **ASCII Character Rule (CRITICAL):** Do NOT use non-standard unicode characters, em-dashes (e.g. `—`), or fancy unicode arrows (e.g. `→`). Use standard keyboard ASCII characters only: standard hyphen `-`, ASCII arrow `->`, standard ASCII dots `...`, standard single quote `'`.
- If you write a code snippet, always **explain why** it is written that way (e.g., *why L2 normalization is necessary at the end of the forward pass*).
- **Do not affirm unnecessarily, do not compliment.** If I suggest an architecturally bad or unnecessarily complicated approach, state it directly and explain the risks.
- **Guard the scope:** If I start inventing complex extensions (relevance feedback, re-ranking), remind me that it distracts from completing the core pipeline.

