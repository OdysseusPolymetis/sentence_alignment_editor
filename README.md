# LaBSE Aligner

A lightweight local interface for exploring, reviewing, and manually correcting sentence-level alignments based on multilingual sentence embeddings.

The application was initially designed for intertextuality and translation-alignment experiments, using [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), but it can load any compatible `sentence-transformers` model from Hugging Face.

The interface is built with Streamlit and can use an NVIDIA GPU through PyTorch/CUDA.

## Features

- Load or paste two texts to compare
- Use `sentence-transformers/LaBSE` by default
- Load another compatible Sentence Transformer from Hugging Face
- GPU inference with CUDA when available
- FP16 model inference for faster embedding generation
- Segment texts by:
  - sentences
  - lines
  - paragraphs
- Compute normalized sentence embeddings
- Compare segments using cosine similarity
- Filter candidate alignments using:
  - a cosine similarity threshold
  - a configurable top-*k* per source segment
- Optionally display all pairs above the selected threshold
- Review alignments one by one in a dedicated reading interface
- Display surrounding source and target segments
- Manually edit either side of an alignment
- Extend an alignment with the previous or following segment
- Recompute cosine similarity after manual correction
- Annotate candidate alignments as:
  - `Yes`
  - `No`
  - `To review`
- Assign a confidence score from 0 to 100
- Export reviewed alignments to CSV

The exported data preserves both the automatically detected alignment and the manually corrected version.

# Installation

## 1. Clone the repository

Open a terminal and clone the repository:

```bash
git clone https://github.com/OdysseusPolymetis/sentence_alignment_editor.git
```

Then enter the project directory:

```bash
cd sentence_alignment_editor
```

## 2. Create a Python virtual environment

Python 3.10 or later is recommended. The application has been tested with Python 3.12.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Once activated, the terminal should display something similar to:

```text
(.venv) user@computer:~/sentence_alignment_editor$
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch

PyTorch should be installed separately so that the version matches the hardware configuration.

### NVIDIA GPU

For an NVIDIA GPU, install a CUDA-enabled version of PyTorch.

For example, with a system compatible with the CUDA 12.8 PyTorch build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

You can check that PyTorch detects the GPU with:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GiB"
    )
PY
```

A working NVIDIA installation should return something similar to:

```text
PyTorch: 2.x.x+cu128
CUDA version: 12.8
CUDA available: True
GPU: NVIDIA ...
```

The CUDA version reported by PyTorch does not necessarily have to match the maximum CUDA version reported by `nvidia-smi`.

### CPU-only installation

If no compatible NVIDIA GPU is available, the application can also run on CPU.

Install the standard PyTorch package:

```bash
python -m pip install torch
```

Inference will be slower, especially on large corpora.

## 4. Install the remaining dependencies

Install the Python packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
scipy
scikit-learn
sentence-transformers
streamlit
```

## 5. Run the application

Start Streamlit from the repository directory:

```bash
streamlit run app.py
```

The application should automatically open in your web browser.

If it does not, open:

```text
http://localhost:8501
```

## Quick installation summary

On an Ubuntu/Linux machine with an NVIDIA GPU:

```bash
git clone https://github.com/OdysseusPolymetis/sentence_alignment_editor.git
cd sentence_alignment_editor

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

streamlit run app.py
```

For subsequent launches, you only need:

```bash
cd YOUR-REPOSITORY
source .venv/bin/activate
streamlit run app.py
```

# Usage

## Typical workflow

1. Load or paste two texts.
2. Choose a Sentence Transformer model.
3. Select a segmentation method.
4. Compute embeddings.
5. Set a cosine similarity threshold and a maximum number of candidates per source segment.
6. Review proposed alignments.
7. Adjust segment boundaries when necessary.
8. Recompute the similarity score for corrected pairs.
9. Validate or reject the candidate alignment.
10. Export the reviewed dataset as CSV.

# Models

The default model is:

```text
sentence-transformers/LaBSE
```

Any Sentence Transformer compatible with the `sentence-transformers` library can be entered in the model field.

For example:

```text
sentence-transformers/LaBSE
```

or a custom/fine-tuned Hugging Face model:

```text
username/my-finetuned-model
```

Local Sentence Transformer checkpoints can also be used by providing their path.

On first use, a Hugging Face model will be downloaded automatically and cached locally.

# Similarity calculation

Embeddings are normalized during encoding:

```python
normalize_embeddings=True
```

Cosine similarity can therefore be computed as a dot product between normalized vectors.

For candidate retrieval, the application processes similarity calculations in chunks rather than systematically materializing the complete source × target matrix in GPU memory.

This makes it possible to work with relatively large texts while keeping GPU memory usage limited.

# Manual alignment correction

Automatic sentence segmentation often produces boundaries that are not ideal for translation or intertextuality analysis.

For example, a source segment may correspond to:

- one target segment,
- part of a target segment,
- two consecutive target segments,
- or a manually reconstructed passage.

The review interface therefore allows both source and target segments to be edited.

Adjacent segments can be added using:

```text
+ previous
+ next
```

Segments can also be shortened directly in the editing area.

The corrected pair can then be re-embedded and its cosine similarity recomputed.

The original automatic alignment is preserved.

# CSV export

The exported CSV contains information such as:

```text
model
threshold
top_k
id_a
segments_a_utilises
texte_a_original
texte_a_edite
id_b
segments_b_utilises
texte_b_original
texte_b_edite
cosine_original
cosine_edite
decision
certitude
```

This makes it possible to distinguish between:

- automatic candidate generation,
- manual segmentation correction,
- model similarity scores,
- and expert validation.

The resulting CSV can therefore also be used as an expert-reviewed evaluation dataset or as the basis for later model training or fine-tuning.

# Performance

The application is designed to benefit from GPU acceleration when CUDA is available.

It has currently been tested with:

- Ubuntu 24.04
- Python 3.12
- PyTorch 2.11
- NVIDIA Quadro RTX 4000
- 8 GB VRAM

On this configuration, LaBSE inference works comfortably in FP16 for sentence-level alignment experiments.

The batch size can be configured in the interface. If a CUDA out-of-memory error occurs, the application automatically retries with a smaller batch size.

# Current limitations

The current version is intended as an experimental research tool.

In particular:

- sentence segmentation is currently relatively simple;
- candidate retrieval is based primarily on embedding similarity;
- no sequential alignment constraint is applied;
- annotations are stored in the current Streamlit session until exported;
- very large result sets may become cumbersome to inspect in the browser.

Future versions may include more sophisticated segmentation, persistent projects, additional retrieval strategies, and dedicated sequential alignment modes.

# Research use

The tool was developed for experiments involving multilingual textual alignment and intertextuality detection, especially in historical and literary corpora.

It is intended to support a human-in-the-loop workflow:

```text
automatic candidate generation
        ↓
semantic similarity
        ↓
manual boundary correction
        ↓
expert validation
        ↓
reviewed alignment dataset
```

The cosine score should be treated as a retrieval and ranking signal, not as evidence of intertextuality by itself.

# License

Add the license used for the repository here.

For example:

```text
MIT License
```

# Citation

If this software becomes part of a published research workflow, citation information will be added here in a future release.
