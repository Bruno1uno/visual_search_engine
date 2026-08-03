# Metric Learning & Multi-Modal Retrieval – Kontext Projektu

## Cíl Projektu
Portfolio ML engineering projekt demonstrující plný pipeline: **Trénink metric learning encoderu → Offline evaluace na unseen třídách → FAISS indexace → FastAPI serving → React frontend + Streamlit dashboard**.

Cílem je naučit se pracovat s embedding prostory, vyhledáváním nejbližších sousedů (Vector Search) a multi-modálním vyhladáváním (Text-to-Image přes CLIP).

---

## Dataset & Split (KRITICKÉ PRAVIDLO)
- **Dataset:** CUB-200-2011 (200 tříd ptáků, ~11 788 obrázků).
- **Disjoint Class Split (Zero-Shot Setup):**
  - **Třídy 1–100 (Seen classes):** Slouží pro trénink (`train_dataset`), validaci a malou část pro `seen_test_dataset` (pro k-NN & Confusion Matrix sanity check).
  - **Třídy 101–200 (Unseen test classes):** Model tyto třídy během tréninku NIKDY nevidí. Vyhledávání a evaluace (Recall@K, mAP, NMI) na těchto třídách prokazuje generalizaci embedding prostoru.
- **Hardware:** RTX 3060 12GB, PyTorch, CUDA.

---

## Architektura a Pravidla
1. **Dva Samostatné FAISS Indexy:**
   - ResNet custom embedding (128D) a CLIP embedding (512D) mají **odlišnou geometrii a dimenzi**.
   - Nikdy se nemíchají v jednom indexu! Vytváří se `indices/resnet_cub200.faiss` a `indices/clip_cub200.faiss`.
2. **Offline vs. Online Oddělení:**
   - **FAISS indexace a vyhodnocení metrik probíhá OFFLINE** (skripty `src/indexer.py` a `src/evaluate.py`).
   - Výsledky metrik a grafů se ukládají do `metrics/` a `plots/`.
   - **FastAPI při startu načte uložené `.faiss` indexy a `id_to_metadata.json` pouze jednou do paměti RAM**.
   - Nahraný query obrázek z frontendu se **NEUKLÁDÁ** do FAISS indexu – vyhledávání je read-only operation.
3. **Stabilní Inference Engine (`src/inference.py`):**
   - FastAPI server (`main.py`) a Gradio/Streamlit neprovádí matematické operace s FAISS přímo v kontrolerech.
   - Volají čisté rozhraní `src/inference.py`:
     ```python
     def search_by_image(image_bytes: bytes, engine_type: str = "resnet", top_k: int = 8) -> list[dict]: ...
     def search_by_text(text_query: str, top_k: int = 8) -> list[dict]: ...
     ```
4. Důležité: confusion matrix a k-NN klasifikační accuracy se počítají pouze na seen test splitu (třídy 1–100, neviděné obrázky), nikdy na unseen (101–200), protože tam nedává smysl fixní label set na obou stranách.

---

## Struktura Repozitáře
```text
visual_search_engine/
├── data/                      # CUB-200-2011 dataset (gitignored)
├── checkpoints/               # Váhy natrénovaného ResNet encoderu (.pt)
├── indices/                   # FAISS indexy (.faiss) a metadata mapping (.json)
├── metrics/                   # Uložené výsledky metrik (JSON)
├── plots/                     # Vygenerované grafy (t-SNE, loss, confusion matrix)
├── src/
│   ├── dataset.py             # CUB loader, disjoint sampler (MPerClassSampler)
│   ├── model.py               # ResNet34 backbone + L2-normalized 128D head + případně ResNet18 ...
│   ├── loss.py                # Triplet loss s online batch-hard miningem
│   ├── train.py               # Tréninková smyčka, validation Recall@1 tracking
│   ├── evaluate.py            # Výpočet Recall@K, mAP, NMI, t-SNE a Ablation study
│   ├── indexer.py             # Offline generování FAISS indexů
│   └── inference.py           # Stabilní vyhledávací engine pro API
├── main.py                    # FastAPI REST API server
├── dashboard.py               # Streamlit internal ML presentation dashboard
├── frontend/                  # React Web Application (Product UI)
├── requirements.txt
├── PLAN.md
├── AGENTS.md
└── README.md
```
## Testování

Testy vznikají průběžně s implementací, ne až na konci. Mirror struktury `src/`:

| Test | Co ověřuje | Proč je kritický |
| :--- | :--- | :--- |
| `test_dataset.py` | Žádný přesah tříd mezi seen (1–100)/unseen (101–200); žádný přesah souborů mezi train/val/seen_test | Chyba tady = celá evaluace ve Fázi 3 vyrábí nesmyslná čísla, aniž by to bylo vidět na první pohled |
| `test_sampler.py` | `MPerClassSampler` skutečně vrací P tříd × M instancí na batch, ne náhodné rozložení | Bez tohohle testu se hard-mining tiše "vypne" (chybí positive páry) a nikdo si toho nevšimne |
| `test_model.py` | Výstupní embedding má L2 normu přesně 1.0 | Sanity check normalizace — pokud selže, cosine similarity ve FAISS přestane odpovídat realitě |
| `test_train.py` | Smoke test: jeden trénovací krok proběhne bez chyby, loss je konečné číslo (ne NaN/inf) | Levný test, chytá rozbité gradient flow hned na začátku |
| `test_indexer.py` | Počet vektorů ve FAISS indexu == počet položek v `id_to_metadata.json` | Nejčastější reálná chyba: rozjetý index a mapping po rebuildu |
| `test_inference.py` | `search_by_image`/`search_by_text` vrací top_k výsledků ve správném formátu, správně rozlišuje `engine_type` | Ověřuje samotné stabilní API, na kterém stojí celý serving |
| `test_serving.py` | FastAPI endpointy vrací 200 a očekávaný JSON tvar | Základní kontrola, že se serving vrstva nerozbila při refaktoru |
---

## Jak mi pomáhat (Pravidla komunikace a kódování)
- Pokud napíšeš úryvek kódu, vždy **vysvětli proč** je napsaný daným způsobem (např. *proč je nutná L2 normalizace na konci dopředného průchodu*).
- **Neafirmuj zbytečně, nekomplimentuj.** Pokud navrhnu architektonicky špatný nebo zbytečně komplikovaný přístup, řekni to rovnou a vysvětli rizika.
- **Hlídej scope:** Pokud začnu vymýšlet další složitá rozšíření (relevance feedback, re-ranking), upozorni mě, že to odvádí pozornost od dokončení core pipeline.
