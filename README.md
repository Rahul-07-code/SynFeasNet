# SynFeasNet 🧬

> A GNN-based deep learning system for predicting the synthetic feasibility of drug-like molecules.

Generative chemistry models often produce molecular structures that look promising *in silico* but are notoriously difficult or impossible to actually synthesize in the lab. SynFeasNet bridges this gap by scoring molecules with a robust Synthetic Practicality Index (SPI) prior to expensive wet-lab validation. By filtering out chemically impractical candidates early in the drug discovery pipeline, researchers can focus resources on manufacturable, high-potential compounds.

## How It Works

*   **Graph Representation**: Molecules are ingested as SMILES strings and converted into rich atom and bond feature graphs using RDKit.
*   **The Synthetic Practicality Index (SPI)**: A multi-dimensional composite score (0 to 1) that evaluates molecules across factors like synthetic complexity, route practicality, scalability, and medicinal chemistry realism.
*   **GNN Architecture**: A Graph Neural Network regresses the SPI score and classifies the molecule into one of five synthesis difficulty tiers (from *trivial* to *intractable*).
*   **Feasibility Gating**: An initial Stage 1 filter aggressively removes clearly intractable molecules (SPI < 0.25) before further downstream evaluation or model training.
*   **Integrated Retrosynthesis**: A built-in BFS-based retrosynthesis engine generates and scores multi-step synthetic routes to validate predicted feasibility.

## Project Structure

```text
backend/
├── api/              # FastAPI REST endpoints for real-time prediction
├── app/              # Streamlit interactive web interface
├── models/           # GNN model architectures and definitions
├── training/         # Training loops, loss functions, and scaffold-aware splitting
├── inference/        # End-to-end inference pipeline for new SMILES
├── retrosynthesis/   # BFS retrosynthesis engine with reaction templates
├── explainability/   # GNN attribution and saliency maps for predictions
├── data/processed/   # Preprocessed molecular datasets and features
├── checkpoints/      # Saved model weights
├── tests/            # Unit and integration test suites
├── scscore/          # SCScore integration for heuristic baselines
├── preprocess.py     # Data preprocessing and featurization scripts
└── spi_labels.py     # SPI label generation from raw dataset features
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI Server**
   Start the REST API for model inference:
   ```bash
   uvicorn api.main:app --reload
   ```

3. **Run the Streamlit Dashboard**
   Launch the interactive web UI:
   ```bash
   streamlit run app/main.py
   ```

## Tech Stack

*   **Core**: Python 3.x
*   **Deep Learning**: PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4, HuggingFace Transformers + PEFT
*   **Cheminformatics**: RDKit, SCScore
*   **Data & ML**: scikit-learn, XGBoost, NumPy, Pandas
*   **API & UI**: FastAPI, Uvicorn, Streamlit
*   **Visualization**: Matplotlib, Seaborn

## Status

**Status: Active Research Project** 🔬
SynFeasNet is currently under active development as a research tool for computational chemistry and drug discovery. Features, APIs, and model architectures are subject to iteration.
