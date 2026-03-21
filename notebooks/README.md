# 🧪 OmniEmbed Analysis Notebooks

This directory contains our interactive Jupyter Notebook suite. We utilize these files to rapidly experiment with, visualize, and test the hidden state distributions of the Qwen3-Omni multidimensional landscape.

All notebooks are designed out-of-the-box to be executed in **Google Colab** ☁️ as well as standard IDE container setups! (They safely instantiate auto-environment git clones and handle required 4-bit configurations for smaller VRAM environments).

### 📓 The Suite

* **[00_model_exploration.ipynb](./00_model_exploration.ipynb)**  
  Provides visualization and structural diagrams mapping out our fundamental strategy to attach mathematical Hooks to the inner LM layers before logits are printed.
  
* **[01_baseline_extraction.ipynb](./01_baseline_extraction.ipynb)**  
  A straightforward command-line wrapped notebook executing default textual and image representation gathering locally.
  
* **🌟 [01b_extended_extraction_and_search.ipynb](./01b_extended_extraction_and_search.ipynb)**  
  Our premium interactive universal layout. It natively renders complex **Audio Waves** 🔊 alongside Image files and executes parallel embeddings over *Images, Texts, Audio, and Video*. It features our built-in `Cosine-Similarity Universal Search Algorithm` mapping audio clips directly back to corresponding text blocks.

* **[02_modality_gap_analysis.ipynb](./02_modality_gap_analysis.ipynb)**  
  Renders clear geometric scatter-plots showcasing the strict "Modality Gap" offset and visually tracks our `ZCA-Whitening` functions physically pulling the clusters securely together.

* **[03_compression_tradeoffs.ipynb](./03_compression_tradeoffs.ipynb)**  
  Renders standard evaluation tracking vectors. Features an explained variance plotting boundary to visually represent exactly how we select dimensionality ceilings limits (e.g. why 512 dimensions is preferred over 4096-d strings).
