# 🧪 OmniEmbed Analysis Notebooks

This directory contains our interactive Jupyter Notebook suite. We utilize these files to rapidly experiment with, visualize, and test the hidden state distributions of the Qwen3-Omni multidimensional landscape.

All notebooks are designed out-of-the-box to be executed in **Google Colab** ☁️ as well as standard IDE container setups.

### 📓 The Suite

* **[01_omni_pipeline_core.ipynb](./01_omni_pipeline_core.ipynb)**  
  This is the complete linear backend pipeline. It covers:
  - **Model Exploration**: Architectural diagrams of the hook strategy.
  - **Baseline Extraction**: Harvesting representations for Text and Images.
  - **Gap Analysis**: Visualizing distribution drifts and executing ZCA-Whitening.
  - **Compression**: PCA variance curves to determine the ideal dimensionality (512d).

* **🌟 [02_interactive_omni_search.ipynb](./02_interactive_omni_search.ipynb)**  
  Our premium interactive front-end layout. 
  - **Multi-modal Rendering**: Natively plays audio clips 🔊 and renders HTML document snippets.
  - **4-Modality Extraction**: Processes Text, Image, Audio, and Video in parallel.
  - **Universal Search**: Implements a unified Cosine-Similarity search engine mapping one modality directly to another in a shared vector space.
