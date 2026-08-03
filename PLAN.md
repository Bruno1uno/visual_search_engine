# PLAN.md – Metric Learning & Multi-Modal Retrieval Pipeline

## Přehled projektu
Cílem projektu je vytvořit produkčně čistou **Full ML pipeline (Train → Evaluation → Indexing → Inference → Serving)** zaměřenou na **Metric Learning** a **Multi-Modal Retrieval** (Image-to-Image & Text-to-Image vyhledávání).

* **Dataset:** CUB-200-2011 (11 788 obrázků, 200 tříd ptáků).
* **Split:** Disjoint class split – Třídy 1–100 (Train/Val + Seen Test subset), Třídy 101–200 (Unseen Test set pro Zero-Shot evaluation).
* **Hardware:** RTX 3060 12GB, PyTorch, CUDA.

---

## Architektura a Klíčové Rozhraní

```text
               ┌────────────────────────────────────────────────────────┐
               │                OFFLINE STAGE (Trénink & Indexace)       │
               └────────────────────────────────────────────────────────┘
                                           │
  ┌────────────────────────┐    ┌──────────┴───────────┐    ┌────────────────────────┐
  │  1. Trénink ResNet     │    │  2. Evaluace & Plots │    │ 3. Indexer (FAISS)     │
  │  (Triplet + Hard Mine) │───>│  (Recall@K, t-SNE)   │    │ (ResNet 128D & CLIP)  │
  └────────────────────────┘    └──────────────────────┘    └───────────┬────────────┘
                                                                        │
                                                         Uložené indexy (.faiss) + metadata (.json)
                                                                        │
               ┌────────────────────────────────────────────────────────┴┐
               │                 ONLINE STAGE (Serving & User UI)        │
               └─────────────────────────────────────────────────────────┘
                                           │
                                ┌──────────┴───────────┐
                                │   src/inference.py   │  <-- Stabilní rozhraní (Engine)
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
          └─────────────────────┘
```

---

## Fáze Vývoje (Detailní Krok za Krokem)

### Fáze 1: Data Pipeline & Disjoint Split (`src/dataset.py`)
- **Dataset Downloader/Parser:** Skript pro stáhnutí a rozbalení CUB-200-2011.
- **Seen classes (1–100)** rozděleny na tři disjoint podmnožiny obrázků (ne tříd — třídy jsou stejné napříč všemi třemi):
  - `train_dataset` (~70 % obrázků) — trénink modelu
  - `val_dataset` (~10 % obrázků) — checkpoint selection za běhu tréninku (validation Recall@1 po každé epoše)
  - `seen_test_dataset` (~20 % obrázků) — finální sanity-check evaluace PO tréninku (k-NN accuracy, confusion matrix). Nikdy se nepoužívá k výběru checkpointu — jinak by byl reportovaný výsledek zkreslený tím, že checkpoint byl vybrán podle dat, na kterých se pak reportuje.
- **Unseen test classes (101–200)** — všechny obrázky tvoří `unseen_test_dataset`, model je během tréninku nikdy nevidí. Hlavní zero-shot benchmark (Recall@K, mAP, NMI).
- **Sampler pro Triplet Mining:** `MPerClassSampler` z knihovny `pytorch-metric-learning` (nebo vlastního sampleru):
  - Zajistí, že každá batch obsahuje $P$ tříd po $M$ obrázcích (např. 16 tříd $\times$ 4 obrázky = batch size 64). Bez tohoto sampleru nelze v rámci batch dělat efektivní online hard-negative mining!
- **Data Augmentations:** standardní PyTorch transforms (Resize, RandomHorizontalFlip, ColorJitter pro train; CenterCrop + Normalize pro eval).

---

### Fáze 2: Model, Triplet Loss & Offline Trénink (`src/model.py`, `src/loss.py`, `src/train.py`)
- **Model Backbone (`src/model.py`):**
  - ResNet34 (příp. ResNet18) předtrénovaný na ImageNetu (`models.ResNet34_Weights.DEFAULT`).
  - Nahrazení klasifikační vrstvy `model.fc` lineární projekční vrstvou: `nn.Linear(in_features, 128)`.
  - **L2 Normalizace:** V dopředném průchodu aplikovat `F.normalize(x, p=2, dim=1)` na výstupní 128D vektor. Výstup leží na jednotkové sféře, což zaručuje ekvivalentnost Cosine Similarity a L2 vzdálenosti.
- **Loss & Mining (`src/loss.py`):**
  - **Triplet Loss:** Margin loss nad trojicemi (Anchor, Positive, Negative).
  - **Online Batch-Hard / Semi-Hard Mining:** Mining je celý delegovaný na `pytorch_metric_learning.miners.TripletMarginMiner(type_of_triplets="hard")` — knihovna interně spočítá pairwise distance matici B×B a pro každý anchor vybere nejtěžší positive/negative pár, nic z tohohle se neimplementuje ručně. Miner vrací seznam (anchor, positive, negative) indexů, které se pak předají do `losses.TripletMarginLoss`.
  - Knihovna: `pytorch_metric_learning.losses.TripletMarginLoss` a `miners.TripletMarginMiner(type_of_triplets="hard")`.
- **Tréninková Smyčka (`src/train.py`):**
  - Optimizer: AdamW (learning rate ~1e-4 pro backbone, ~1e-3 pro embedding head).
  - Checkpointing: Ukládání modelu s nejnižší validation loss a nejvyšším validation Recall@1.
  - Ukládání tréninkové historie (`history.json`: train loss, val loss, validation Recall@1).

---

### Fáze 3: Offline Evaluace, FAISS Indexer & Metriky (`src/evaluate.py`, `src/indexer.py`)

#### A. Offline FAISS Indexer (`src/indexer.py`)
- **Průchod Datasetem:** Extrakce 128D embeddingů ze všech obrázků CUB-200 pomocí natrénovaného ResNetu a 512D embeddingů pomocí frozen CLIP (`clip-ViT-B-32`).
- **FAISS Indexace:**
  - Vytvoření dvou samostatných FAISS indexů:
    1. `resnet_index.faiss`: `faiss.IndexFlatIP(128)` (Inner Product pro L2-normované vektory = Cosine Similarity).
    2. `clip_index.faiss`: `faiss.IndexFlatIP(512)` (pro CLIP embeddingy).
  - `index.add(embeddings)` přidá vektory pod číselnými ID ($0 \dots N-1$).
- **Uložení na Disk:**
  - `indices/resnet_cub200.faiss`
  - `indices/clip_cub200.faiss`
  - `indices/id_to_metadata.json`: Slovník mapující `int_id -> { "image_path": "...", "class_id": 105, "class_name": "black_footed_albatross" }`.

#### B. Offline Evaluace & Generování Grafů (`src/evaluate.py`)
- **Zero-Shot Evaluace na Unseen Třídách (101–200):**
  - **Recall@K (K=1, 2, 4, 8):** Pročte každý testovací query obrázek, získá Top-K výsledků z FAISS a ověří, zda alespoň 1 výsledek má stejnou třídu.
  - **mAP (mean Average Precision):** Vyhodnocení pořadí relevantních obrázků.
  - **NMI (Normalized Mutual Information):** K-Means clustering na embeddingách vs. skutečné labely tříd.
- **Sanity Check na Seen Test Subsetu (1–100):**
  - k-NN klasifikační přesnost a Confusion Matrix (uložena jako PNG image).
- **Ablation Study Experiment:**
  - Porovnání vyhodnocení metrik modelu trénovaného s **Hard Negative Mining** vs. modelu trénovaného s **Random Triplet Sampling**.
- **Vizualizace Embedding Prostoru:**
  - Výpočet t-SNE / UMAP projekcí (před tréninkem vs po tréninku, seen vs unseen).
- **Uložení Výstupů:**
  - Všechny metriky uloženy do `metrics/eval_results.json` a `metrics/ablation.json`.
  - Grafy uloženy do `plots/tsne_unseen.png`, `plots/confusion_seen.png`.

---

### Fáze 4: Serving Engine & FastAPI REST API (`src/inference.py`, `main.py`)
- **Stabilní Engine (`src/inference.py`):**
  - Načte při inicializaci natrénovaný ResNet model, CLIP model, FAISS indexy a `id_to_metadata.json`.
  - Metoda `search_by_image(image_bytes, engine_type='resnet'|'clip', top_k=8)`:
    1. Spočítá embedding nahraného obrázku.
    2. Zavolá `faiss_index.search(query_vec, top_k)`.
    3. Vyhledané číselné indexy namapuje přes `id_to_metadata.json` na cesty k obrázkům a skóre podobnosti.
  - Metoda `search_by_text(text_query, top_k=8)`:
    1. Spočítá textový embedding pomocí CLIP text encoderu.
    2. Zavolá `clip_faiss_index.search(text_vec, top_k)`.
    3. Vrátí seznam nejrelevantnějších obrázků.
  - *Důležité:* Uploadovaný query obrázek se **NEUKLÁDÁ** do FAISS indexu (FAISS index je read-only databáze).
- **FastAPI REST API (`main.py`):**
  - Endpoints:
    - `POST /api/search/image` (form-data: file, engine_type, top_k)
    - `POST /api/search/text` (json: text_query, top_k)
    - `GET /api/metrics` (vrací předpočítaná data z `metrics/eval_results.json`)
    - Static file serving pro zobrazování zdrojových obrázků datasetu CUB-200.
  - Nastavení `CORSMiddleware` pro komunikaci s React frontendem.

---

### Fáze 5: Internal Dev Dashboard (Streamlit) & Product Web UI (React)

#### A. Internal ML Dashboard (`dashboard.py` - Streamlit)
- Interní nástroj pro prezentaci výsledků vývoje.
- Zobrazuje předpočítaná data z Fáze 3:
  - Průběh tréninku (loss křivky).
  - Tabulka Recall@K, mAP, NMI na unseen třídách.
  - Ablation study tabulka (Random vs. Hard Mining).
  - Obrázky t-SNE projekcí a Confusion Matrix.

#### B. User-Facing Product Web UI (`frontend/` - React)
- Čisté, moderní SPA rozhraní.
- Dva hlavní záložky / režimy:
  1. **Image Similarity Search:** Drag-and-drop upload obrázku + přepínač modelu (Custom Metric ResNet vs. CLIP Visual) + slider pro Top-K. Zobrazení výsledků v gridu se skóre podobnosti.
  2. **Text-to-Image Search:** Textové vyhladávací pole (např. *"yellow bird with black wings"*) -> dotaz na CLIP text encoder -> zobrazení nalezených obrázků.

---

## Summary Metrik & Souborové Infrastruktury

| Soubor | Účel |
| :--- | :--- |
| `data/CUB_200_2011/` | Stažený dataset CUB-200-2011 |
| `checkpoints/best_resnet34_metric.pt` | Uložené váhy natrénovaného encoderu |
| `indices/resnet_cub200.faiss` | FAISS index (128D) |
| `indices/clip_cub200.faiss` | FAISS index (512D) |
| `indices/id_to_metadata.json` | Mapování FAISS ID na cestu k obrázku a label |
| `metrics/eval_results.json` | Uložené vypočítané metriky (Recall@K, mAP, NMI) |
| `plots/` | Vygenerované grafy (t-SNE, loss, confusion matrix) |
