# AI Photo Album

An intelligent photo management system with semantic search, face recognition, and learning-to-rank capabilities.

## Features

- **Semantic Search**: Natural language queries powered by CLIP embeddings
- **Face Recognition**: Automatic face detection, clustering, and identity management
- **Learning-to-Rank**: XGBoost-based ranking model trained on user click feedback
- **Image Captioning**: Automatic caption generation using Qwen2-VL
- **Aesthetic Scoring**: Neural network-based image quality prediction
- **Analytics Dashboard**: Model performance metrics and A/B testing visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                              │
├─────────────────────────────────────────────────────────────┤
│  Search Engine  │  Face Clustering  │  Analytics Dashboard  │
├─────────────────────────────────────────────────────────────┤
│              Ranking Layer (XGBoost / Heuristic)            │
├─────────────────────────────────────────────────────────────┤
│  CLIP Embeddings  │  Caption Embeddings  │  Face Embeddings │
├─────────────────────────────────────────────────────────────┤
│         ChromaDB (Vectors)    │    SQLite (Metadata)        │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- OpenAI API key (for query parsing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/faceRec.git
cd faceRec

# Create virtual environment
conda create -n facerec python=3.10
conda activate facerec

# Install dependencies
pip install -r requirements.txt

# Download model weights (not included in repo)
# See "Model Downloads" section below
```

### Model Downloads

The following model weights must be downloaded separately:

1. **LAION Aesthetic Predictor**
   ```bash
   wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac_logos_ava1-l14-linearMSE.pth
   ```

2. **Qwen2-VL** (auto-downloads on first run, or pre-download):
   ```python
   from transformers import Qwen2VLForConditionalGeneration
   model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
   ```

3. **InsightFace** models auto-download on first run

4. **3DDFA_V2** weights are included in `third_party/3DDFA_V2/weights/`

### Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the App

```bash
streamlit run appv2.py
```

### Ingestion Pipeline

Process photos and save embeddings to the database:

```bash
# Default: parallel processing (batch_size=16)
python main.py ingest

# Sequential processing (batch_size=1)
python main.py ingest --sequential

# Custom batch size
python main.py ingest --batch-size 8
python main.py ingest --batch-size 32
```

### Verification Workflow

Verify that the pipeline produces correct results by comparing against a golden dataset.

**Step 1: Capture Golden Dataset (run BEFORE any refactoring)**

```bash
# Capture 300 images using the original implementation
python main.py capture-golden --count 300

# Custom count
python main.py capture-golden --count 500
```

This creates `golden_dataset.json` containing all outputs from the original implementation.

**Step 2: Run Verification (run AFTER changes)**

```bash
# Quick verification (50 images)
python main.py verify --sample 50

# Standard verification (100 images)
python main.py verify --sample 100

# Thorough verification (200 images max)
python main.py verify --sample 200
```

**Verification Options:**

```bash
# Custom batch size for experimentation
python main.py verify --sample 50 --batch-size 8
python main.py verify --sample 50 --batch-size 32

# Different random seed (for different sample selection)
python main.py verify --sample 50 --seed 123

# Disable shuffling (take first N images in order)
python main.py verify --sample 50 --no-shuffle
```

The verification compares all fields:
- Image dimensions, timestamps, ISO
- Global quality metrics (blur, brightness, contrast)
- CLIP semantic vectors (768D)
- Aesthetic scores
- Captions and caption embeddings
- Face data (bboxes, embeddings, pose, quality metrics)

### Benchmarking

Compare sequential vs parallel pipeline performance:

```bash
# Run benchmark with 100 images
python main.py benchmark --sample 100

# Custom sample size
python main.py benchmark --sample 200
```

Outputs `benchmark_report.json` with timing comparisons and speedup metrics.

### CLI Reference

| Command | Description |
|---------|-------------|
| `ingest` | Process photos and save to database |
| `cluster` | Run face clustering on ingested photos |
| `verify` | Verify pipeline against golden dataset |
| `benchmark` | Compare sequential vs parallel performance |
| `capture-golden` | Capture golden dataset for verification |

| Option | Applies To | Description |
|--------|------------|-------------|
| `--sequential` | ingest | Use batch_size=1 |
| `--parallel` | ingest | Use batch_size=16 (default) |
| `--batch-size N` | ingest, verify, benchmark | Override batch size |
| `--sample N` | verify, benchmark | Number of images to process |
| `--count N` | capture-golden | Number of images to capture |
| `--seed N` | verify | Random seed for shuffling (default: 42) |
| `--no-shuffle` | verify | Disable random shuffling |

### Training the Ranking Model

```bash
# Collect training data by using the app with the Heuristic ranker
# Then train the XGBoost model:
python -m src.training.trainer

# Options:
python -m src.training.trainer --help
```

## Project Structure

```
├── src/
│   ├── db/             # Database layer (SQLite + ChromaDB)
│   ├── ingestion/      # Photo processing pipeline
│   ├── model/          # ML models (aesthetic, captioning)
│   ├── rank/           # Ranking strategies (XGBoost, Heuristic)
│   ├── retrieval/      # Search engine
│   ├── evaluation/     # Offline metrics (NDCG, MRR, etc.)
│   ├── training/       # XGBoost trainer
│   └── ui/             # Streamlit components
├── third_party/        # 3DDFA_V2 for face pose
├── models/             # Downloaded model weights (gitignored)
├── appv2.py            # Main application entry
└── requirements.txt
```

## Performance Metrics

The analytics dashboard tracks:
- **CTR (Click-Through Rate)** by ranking model
- **NDCG@K** - Normalized Discounted Cumulative Gain
- **Precision@K** - Fraction of relevant results in top-K
- **MRR** - Mean Reciprocal Rank
- **Position Bias** - CTR decay by result position

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: This project uses third-party models with various licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details. Notably, InsightFace pre-trained models are restricted to non-commercial use.

## Acknowledgments

This project builds upon the work of many open-source projects:

- [OpenAI CLIP](https://github.com/openai/CLIP) - Image-text embeddings
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) - Vision-language model
- [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) - 3D face alignment
- [LAION Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) - Image aesthetics
- [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database

## Author

Jacob Hsiung
