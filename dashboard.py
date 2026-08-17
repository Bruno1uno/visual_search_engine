import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Global Configuration
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Visual Search Engine - ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for larger fonts, spacious padding, and prominent metric cards
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 1.12rem !important;
    }
    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }
    h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    h3 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    .stTextInput input {
        font-size: 1.2rem !important;
        padding: 10px !important;
    }
    .stButton button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics_json(filepath: str) -> dict:
    """Helper function to load evaluation metrics JSON files."""
    path = Path(filepath)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def check_api_health() -> tuple[bool, int]:
    """Checks if FastAPI backend server is online and returns ready status & vector count."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("engine_ready", False), data.get("num_indexed_vectors", 0)
    except Exception:
        pass
    return False, 0


# Sidebar Setup
st.sidebar.title("Visual Search Engine")
st.sidebar.caption("Metric Learning & Multi-Modal Retrieval Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview & Benchmark",
        "Loss Curves & Progress",
        "Embedding Space & t-SNE",
        "Interactive Search Playground",
    ],
)

# API Status Badge in Sidebar Caption
api_online, num_vectors = check_api_health()
st.sidebar.markdown("---")
if api_online:
    st.sidebar.caption(f"Backend API: Online ({num_vectors:,} vectors)")
else:
    st.sidebar.caption("Backend API: Offline (Run uvicorn main:app)")

st.sidebar.info(
    "**Dataset:** CUB-200-2011 (11,788 images)\n\n"
    "**Split:** Seen Classes 1-100 | Unseen Classes 101-200\n\n"
    "**Backbone:** ResNet34 (256D) & OpenCLIP (512D)"
)


# Page 1: Overview & Benchmark
if page == "Overview & Benchmark":
    st.title("Model Evaluation & Literature Benchmark")
    st.markdown(
        "Zero-shot evaluation results calculated on the **Unseen Test Split (Classes 101-200)** "
        "after Optuna hyperparameter optimization."
    )

    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Recall@1 (Proxy Anchor)", value="52.43%", delta="+8.17% (vs Triplet)")
    col2.metric(label="Recall@2 (Proxy Anchor)", value="64.77%", delta="+8.39% (vs Triplet)")
    col3.metric(label="mAP (Proxy Anchor)", value="0.2615", delta="+0.0388 (vs Triplet)")
    col4.metric(label="NMI (Proxy Anchor)", value="0.6037", delta="+0.0530 (vs Triplet)")

    st.markdown("---")
    st.subheader("1. Internal Pipeline Loss Function Comparison (My Trained Models)")
    st.markdown(
        "*Direct comparison between two variants of fine-tuned ResNet-34 encoders on the CUB-200 unseen test set "
        "(Classes 101-200), evaluating the impact of Proxy Anchor Loss vs. Triplet Loss + Hard Negative Mining.*"
    )

    comparison_data = {
        "Metric": ["Seen Test 1-NN Accuracy", "Recall@1", "Recall@2", "Recall@4", "Recall@8", "mAP", "NMI"],
        "Proxy Anchor Loss (256D)": ["78.01%", "52.43%", "64.77%", "75.79%", "84.64%", "0.2615", "0.6037"],
        "Triplet Loss + Hard Mining (256D)": ["73.03%", "44.26%", "56.38%", "67.78%", "78.02%", "0.2227", "0.5507"],
        "Difference": ["+4.98 pp", "+8.17 pp", "+8.39 pp", "+8.01 pp", "+6.62 pp", "+0.0388", "+0.0530"],
    }
    df_comparison = pd.DataFrame(comparison_data)

    def highlight_proxy(col):
        if "Proxy Anchor" in col.name:
            return ["font-weight: 600;"] * len(col)
        return [""] * len(col)

    st.dataframe(df_comparison.style.apply(highlight_proxy, axis=0), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("2. Published Literature Benchmark Comparison (CUB-200 Zero-Shot)")
    st.markdown(
        "Comparing my pipeline implementations against published metric learning literature on the CUB-200 zero-shot benchmark. "
    )

    literature_data = {
        "Method / Reference Paper": [
            "Lifted Structured Feature (CVPR 2016)",
            "Smart Mining (ICCV 2017)",
            "Proxy Anchor Loss Paper (CVPR 2020)",
            "[My Implementation] ResNet-34 Triplet + Hard Mining",
            "[My Implementation] ResNet-34 Proxy Anchor Loss",
        ],
        "Backbone": ["GoogLeNet", "Inception-V1", "Inception-BN", "ResNet-34", "ResNet-34"],
        "Dim": ["512D", "64D", "512D", "256D", "256D"],
        "Recall@1": ["47.20%", "49.80%", "68.40%", "44.26%", "52.43%"],
        "Recall@2": ["58.90%", "62.30%", "79.20%", "56.38%", "64.77%"],
        "NMI": ["56.20%", "59.90%", "N/A", "55.07%", "60.37%"],
    }
    df_literature = pd.DataFrame(literature_data)

    def style_literature(df):
        style_df = pd.DataFrame("", index=df.index, columns=df.columns)
        
        # Highlight last 2 implementation rows with distinct cyan text color
        for r_idx in [3, 4]:
            for col in df.columns:
                style_df.loc[r_idx, col] = "color: #38bdf8; font-weight: 600;"

        # Highlight globally best values in bold for metric columns
        style_df.loc[2, "Recall@1"] += " font-weight: 800; text-decoration: underline;"
        style_df.loc[2, "Recall@2"] += " font-weight: 800; text-decoration: underline;"
        style_df.loc[4, "NMI"] += " font-weight: 800; text-decoration: underline;"
        
        return style_df

    st.dataframe(df_literature.style.apply(style_literature, axis=None), width="stretch", hide_index=True)


# Page 2: Loss Curves & Progress
elif page == "Loss Curves & Progress":
    st.title("Training Loss Curves & Validation Progression")
    st.markdown(
        "Comparison of training loss reduction and validation Recall@1 tracking across epochs."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Proxy Anchor Loss (256D)")
        plot_pa = Path("plots/training_history_proxy_anchor.png")
        if plot_pa.exists():
            st.image(str(plot_pa), width="stretch")
        else:
            st.warning("Proxy Anchor training history plot not found at plots/training_history_proxy_anchor.png")

    with col2:
        st.subheader("Triplet Loss + Hard Mining (256D)")
        plot_tr = Path("plots/training_history_triplet.png")
        if plot_tr.exists():
            st.image(str(plot_tr), width="stretch")
        else:
            st.warning("Triplet Loss training history plot not found at plots/training_history_triplet.png")


# Page 3: Embedding Space & t-SNE
elif page == "Embedding Space & t-SNE":
    st.title("Embedding Space Projection & Confusion Matrix")

    tab1, tab2 = st.tabs(["t-SNE Projections", "Seen Classes Confusion Matrix"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Unseen Test Set t-SNE (Classes 101-200)")
            tsne_unseen = Path("plots/tsne_unseen_proxy_anchor.png")
            if tsne_unseen.exists():
                st.image(str(tsne_unseen), width="stretch")
            else:
                st.info("t-SNE unseen plot not found.")

        with col2:
            st.subheader("Seen Test Set t-SNE (Classes 1-100)")
            tsne_seen = Path("plots/tsne_seen_proxy_anchor.png")
            if tsne_seen.exists():
                st.image(str(tsne_seen), width="stretch")
            else:
                st.info("t-SNE seen plot not found.")

    with tab2:
        st.subheader("1-NN Confusion Matrix (Seen Test Classes 1-100)")
        cm_plot = Path("plots/confusion_seen_proxy_anchor.png")
        if cm_plot.exists():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(str(cm_plot), width=650)
        else:
            st.info("Confusion matrix plot not found.")


# Page 4: Interactive Search Playground
elif page == "Interactive Search Playground":
    st.title("Interactive Search Playground (Live REST API Queries)")

    if not api_online:
        st.warning(
            f"FastAPI Backend Server is OFFLINE ({API_BASE_URL}). "
            "Please start the REST API server in a terminal using: "
            "`python -m uvicorn main:app --reload --port 8000`"
        )

    # Expandable Bird Species Browser
    with st.expander("Browse CUB-200 Bird Species List & Sample Prompts"):
        st.markdown("**Sample Text Prompts to try:**")
        st.markdown("- `yellow bird with black wings` (e.g. American Goldfinch)")
        st.markdown("- `red bird with black mask` (e.g. Cardinal)")
        st.markdown("- `large seabird with white belly` (e.g. Albatross)")
        st.markdown("- `small hummingbird with green feathers` (e.g. Anna Hummingbird)")

        meta_file = Path("indices/id_to_metadata.json")
        if meta_file.exists():
            meta_dict = load_metrics_json(str(meta_file))
            unique_species = sorted(list(set(v.get("class_name", "") for v in meta_dict.values() if "class_name" in v)))
            st.markdown(f"**Indexed Bird Species ({len(unique_species)} Total):**")
            st.caption(", ".join(unique_species[:40]) + f"... and {len(unique_species)-40} more.")

    search_tab1, search_tab2 = st.tabs(["Image-to-Image Search", "Text-to-Image Search (OpenCLIP)"])

    # Tab 1: Image Similarity Search
    with search_tab1:
        st.info(
            "**Dataset Information:** This search engine is fine-tuned and indexed on the **CUB-200-2011** dataset "
            "(Caltech-UCSD Birds - 200 bird species, 11,788 images). For best results, query with bird images."
        )
        col_ctrl, col_prev = st.columns([1, 1])

        with col_ctrl:
            uploaded_file = st.file_uploader("Choose a Query Image...", type=["jpg", "jpeg", "png"])
            engine_option = st.radio("Select Search Engine", ["ResNet34 (256D Custom Metric)", "OpenCLIP ViT-B-32 (512D)"])
            engine_type = "resnet" if "ResNet34" in engine_option else "clip"
            top_k = st.slider("Top-K Neighbors", min_value=1, max_value=16, value=8)

            search_btn = st.button("Search Similar Birds", disabled=not uploaded_file or not api_online)

        with col_prev:
            if uploaded_file is not None:
                st.caption("Query Image Preview:")
                query_img = Image.open(uploaded_file)
                st.image(query_img, width=220)

        if search_btn and uploaded_file is not None and api_online:
            st.markdown("---")
            st.subheader(f"Top-{top_k} Search Results ({engine_option})")

            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {"engine_type": engine_type, "top_k": top_k}

            with st.spinner("Querying FAISS vector index via FastAPI..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/search/image", files=files, params=params, timeout=10)
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if not results:
                            st.warning(
                                "No matching birds found. The search engine is trained specifically on the CUB-200-2011 dataset. "
                                "Try uploading a clear photo of a bird!"
                            )
                        else:
                            # Display results in grid columns
                            cols_per_row = 4
                            for i in range(0, len(results), cols_per_row):
                                grid_cols = st.columns(cols_per_row)
                                for j, res in enumerate(results[i : i + cols_per_row]):
                                    with grid_cols[j]:
                                        abs_path = Path(res["abs_path"])
                                        if abs_path.exists():
                                            st.image(str(abs_path), width="stretch")
                                        else:
                                            st.caption(f"[Image file not found: {res['rel_path']}]")
                                        st.markdown(f"**{res['class_name']}**")
                                        st.caption(f"Score: `{res['score']:.4f}` | Class ID: {res['class_id']}")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as err:
                    st.error(f"Connection Error: {err}")

    # Tab 2: Text-to-Image Search
    with search_tab2:
        st.info(
            "**OpenCLIP Multimodal Vector Search:** Converts your text prompt into a 512D embedding vector and "
            "searches FAISS index for images with matching visual attributes. It performs true neural vector search "
            "(NOT string/text matching on file names)."
        )
        st.caption("Try prompts matching visual scenes (without species names): `bird flying in blue sky`, `swimming in water`, `red feathers`, `close up portrait`")

        text_prompt = st.text_input("Enter Text Prompt", value="bird flying in blue sky")
        top_k_text = st.slider("Top-K Text Results", min_value=1, max_value=16, value=8, key="text_topk")

        text_search_btn = st.button("Search Prompt (OpenCLIP)", disabled=not text_prompt.strip() or not api_online)

        if text_search_btn and api_online:
            st.markdown("---")
            st.subheader(f"Top-{top_k_text} Results for prompt: '{text_prompt}'")

            payload = {"text_query": text_prompt.strip(), "top_k": top_k_text}
            with st.spinner("Encoding text prompt via OpenCLIP..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/search/text", json=payload, timeout=10)
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if not results:
                            st.warning(
                                f"No matching birds found for '{text_prompt}'. "
                                "Try describing bird visual traits (e.g. 'yellow bird with black wings', 'small hummingbird with green feathers')."
                            )
                        else:
                            cols_per_row = 4
                            for i in range(0, len(results), cols_per_row):
                                grid_cols = st.columns(cols_per_row)
                                for j, res in enumerate(results[i : i + cols_per_row]):
                                    with grid_cols[j]:
                                        abs_path = Path(res["abs_path"])
                                        if abs_path.exists():
                                            st.image(str(abs_path), width="stretch")
                                        else:
                                            st.caption(f"[Image file not found: {res['rel_path']}]")
                                        st.markdown(f"**{res['class_name']}**")
                                        st.caption(f"Cosine Similarity: `{res['score']:.4f}`")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as err:
                    st.error(f"Connection Error: {err}")
