# ⚙️ `omniembed` Python Package Core

This folder acts as the central hub encompassing our heavily isolated, mathematically robust utilities used to communicate with neural networks and map topological geometries!

### 📦 Core Modules

1. **`extractor.py` (The Interceptor)**
   - Houses the `HiddenStateExtractor` which acts as a PyTorch context hook wrapper (`__enter__` and `__exit__`). It safely registers forward listeners inside deeply isolated Model objects natively grabbing tensors at sequence evaluation time.

2. **`alignment.py` (The Mathematical Tuner)**
   - Operates the `ModalityAligner` tool mapping geometric clusters. Operates strict Zero-Mean scaling rules and calculates singular value decomposition (SVD) constraints perfectly to deploy robust **ZCA Whitening**.

3. **`compression.py` (The Dimensionality Reducer)**
   - Houses `PCACompressor` alongside raw skeleton implementations for dynamic `Matryoshka Projection` pipelines lowering overall tensor footprint sizes for scalable vector database support mappings.

4. **`retrieval.py` (The Evaluation Core)**
   - Standardized statistical analysis functions executing `Recall@1, 5, 10` tracking limits alongside Modality Gap $L_2$ Euclidean distance monitoring thresholds.

5. **`viz.py` (The Dashboard)**
   - Standard graph plotting parameters executing native `UMAP` point clusters mapping structural densities, and implementing isolated Seaborn interaction frameworks!
