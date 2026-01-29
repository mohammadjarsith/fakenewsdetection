# Project Run Notes

## Status
- **Analyzed**: The project is a Deep Learning based Fake News Detection system using Bi-GRU and Attention.
- **Environment**: Python 3.14.2 detected.
- **Dependencies**: 
    - Installed: `numpy`, `pandas`, `nltk`, `beautifulsoup4`, `requests`, `scrapy`, `selenium`.
    - **Failed**: `tensorflow` (not supported on Python 3.14 yet), `gensim` (build failure).
- **Data**:
    - Project data (`snopes.csv`, text files) seems present.
    - **Missing**: `GoogleNews-vectors-negative300.bin.gz` (Required for Word2Vec embeddings).

## Execution
Since TensorFlow could not be installed, the Deep Learning model training (`biGRU_attention.ipynb` / `train_model.py`) **cannot be executed** in this environment.

However, the **Data Preprocessing Pipeline** was successfully set up and can be run.

### Running Preprocessing
To run the preprocessing steps (Top words selection, Tokenization, Dataset splitting):

```bash
python run_pipeline.py
```

## To Run Full Model
To run the full training, you need:
1.  A Python environment compatible with **TensorFlow 1.x or 2.x** (e.g., Python 3.7 - 3.11).
2.  Install `tensorflow`, `gensim`.
3.  Download `GoogleNews-vectors-negative300.bin.gz` and place it in the project root.
