# QalamAI — Abu Hashim AI Model

<div align="center">

**قلم الذكاء الاصطناعي**

An advanced Arabic AI assistant built on open-source language models.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](#)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)

</div>

---
## 🔗 Official Resources & Documentation

### 🧠 Base Models (LLMs)
- Aya Models (Aya-23-8B / Aya-23-35B): https://huggingface.co/CohereForAI
- JAIS-13B: https://huggingface.co/inception-mbzuai
- Qwen2-7B: https://huggingface.co/Qwen

### 🧰 Training & Fine‑Tuning Tools
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- PEFT (LoRA / QLoRA): https://huggingface.co/docs/peft
- BitsAndBytes (4‑bit quantization): https://github.com/TimDettmers/bitsandbytes
- Accelerate: https://huggingface.co/docs/accelerate
- PyTorch: https://pytorch.org

### 🧹 Data Processing & Cleaning
- Python: https://www.python.org
- Regex (re module): https://docs.python.org/3/library/re.html
- FastText (optional): https://fasttext.cc

### 🧪 Evaluation & Benchmarking
- HuggingFace Evaluate: https://huggingface.co/docs/evaluate
- BLEU / ROUGE / METEOR Metrics: https://huggingface.co/docs/evaluate
- CAMeL Arabic NLP Tools: https://camelcameltools.readthedocs.io

### 🚀 Inference & Serving
- FastAPI: https://fastapi.tiangolo.com
- Uvicorn: https://www.uvicorn.org
- Jinja2 Templates: https://jinja.palletsprojects.com
- Pydantic: https://docs.pydantic.dev

### 🔐 Safety & Filtering
- OpenAI Moderation (reference): https://platform.openai.com/docs/guides/moderation
- HuggingFace Safety Models: https://huggingface.co/models?pipeline_tag=text-classification&other=safety

### 🧭 QalamAI Integration
- QalamAI Platform: https://qalamai.net
- JSONL Format: https://jsonlines.org

### 🖥️ Dashboard & Branding
- HTML5: https://developer.mozilla.org/docs/Web/HTML
- CSS3: https://developer.mozilla.org/docs/Web/CSS
- TailwindCSS (optional): https://tailwindcss.com

### 📦 Project Environment
- Python 3.10+: https://www.python.org/downloads
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- NVIDIA Drivers: https://www.nvidia.com/download/index.aspx

### 📄 Licenses
- Aya Models License: https://huggingface.co/CohereForAI/aya-23-8b#license
- JAIS Models License: https://huggingface.co/inception-mbzuai/jais-13b#license
- Qwen Models License: https://huggingface.co/Qwen/Qwen2-7B#license
- Transformers License: https://github.com/huggingface/transformers/blob/main/LICENSE
- FastAPI License: https://github.com/tiangolo/fastapi/blob/master/LICENSE

## Overview

**Abu Hashim** is a standalone Arabic AI model project powered by QalamAI. It provides a complete framework for training, fine-tuning, evaluating, and serving an Arabic-capable language model — all using open-source LLMs with no proprietary API dependencies.

### Key Features

- **Arabic-First**: Built on Aya-23-8B, a multilingual model with strong Arabic capabilities
- **Efficient Fine-Tuning**: LoRA/QLoRA for training on consumer GPUs
- **Complete Pipeline**: Data processing, training, evaluation, and inference in one project
- **Self-Learning**: Feedback collection and incremental model improvement
- **Safety Filters**: Bilingual content safety for Arabic and English
- **QalamAI Bridge**: Automated data collection and learning from QalamAI.net interaction logs
- **Branded Dashboard**: Admin UI with QalamAI branding

## Project Structure

```
├── model_base/              # Base model configuration and download
│   ├── config.py            # Model settings (Aya-23-8B default)
│   └── download_model.py    # HuggingFace model downloader
│
├── dataset_raw/             # Raw training data (user-provided)
├── dataset_processed/       # Processed training data (generated)
│
├── training_scripts/        # Training and data pipeline
│   ├── data_pipeline.py     # Data ingestion pipeline
│   ├── text_cleaner.py      # Arabic text normalization
│   ├── data_formatter.py    # Instruction pair formatting
│   ├── pii_remover.py       # PII removal
│   ├── train.py             # LoRA/QLoRA fine-tuning
│   ├── train_config.py      # Training hyperparameters
│   ├── incremental_train.py # Incremental training
│   ├── self_learning.py     # Self-learning data validation
│   └── update_model.py      # Admin-triggered model update
│
├── model_finetune/          # Fine-tuned model checkpoints
│
├── model_inference/         # Inference utilities
│
├── evaluation/              # Evaluation and benchmarking
│   ├── evaluate.py          # Evaluation runner
│   ├── metrics.py           # Arabic fluency/quality metrics
│   ├── benchmarks.py        # Comparison benchmarks
│   └── report_generator.py  # HTML report generation
│
├── api_server/              # FastAPI inference server
│   ├── main.py              # Server application
│   ├── inference_engine.py  # Model loading and generation
│   ├── safety_filters.py    # Content safety filters
│   ├── schemas.py           # API request/response schemas
│   ├── static/              # CSS and JavaScript assets
│   └── templates/           # HTML templates (dashboard, docs)
│
├── learning_buffer/         # User feedback storage
│
├── branding/                # QalamAI brand assets
│   ├── colors.json          # Color palette
│   ├── typography.md        # Typography guidelines
│   └── theme.css            # CSS theme
│
├── qalam_bridge/            # QalamAI data collection bridge
│   ├── importer.py          # JSON/CSV/JSONL ingestion + cleaning
│   ├── quality_scorer.py    # Quality scoring and filtering
│   ├── dataset_builder.py   # Dataset building and train/eval splitting
│   └── update_dataset.py    # Continuous learning pipeline
│
├── docs/                    # Documentation
│   ├── model_architecture.md
│   ├── dataset_structure.md
│   ├── training_instructions.md
│   ├── inference_server.md
│   ├── update_model.md
│   ├── integration_guide.md
│   ├── branding_guidelines.md
│   └── qalam_bridge.md
│
└── main.py                  # Application entry point
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Base Model

```bash
python -m model_base.download_model --model aya-23-8b
```

### 3. Prepare Training Data

Place your data in `dataset_raw/` (see [Dataset Structure](docs/dataset_structure.md)), then run:

```bash
python -m training_scripts.data_pipeline
```

### 4. Train the Model

```bash
python -m training_scripts.train --preset standard
```

### 5. Start the Inference Server

```bash
python main.py
```

The server starts at `http://localhost:5000` with the admin dashboard.

### 6. Generate Text

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ما هو الذكاء الاصطناعي؟"}'
```

## Supported Models

| Model          | Parameters | Description                              |
|----------------|------------|------------------------------------------|
| Aya-23-8B      | 8B         | Recommended — Arabic-capable multilingual |
| Aya-23-35B     | 35B        | Larger multilingual model                |
| JAIS-13B       | 13B        | Arabic-English bilingual                 |
| Qwen2-7B       | 7B         | Multilingual with Arabic support         |

## Training Presets

| Preset        | Epochs | Batch Size | Seq Length | Use Case              |
|---------------|--------|------------|------------|-----------------------|
| `quick_test`  | 1      | 2          | 512        | Quick validation       |
| `standard`    | 3      | 4          | 2048       | Normal training        |
| `high_quality`| 5      | 2          | 4096       | Maximum quality        |

## API Endpoints

| Method | Endpoint    | Description              |
|--------|-------------|--------------------------|
| GET    | `/health`   | Server health check      |
| POST   | `/generate` | Text generation          |
| POST   | `/chat`     | Chat completion          |
| POST   | `/feedback` | Submit user feedback     |

## Documentation

| Document                                              | Description                    |
|-------------------------------------------------------|--------------------------------|
| [Model Architecture](docs/model_architecture.md)      | Base model and LoRA details    |
| [Dataset Structure](docs/dataset_structure.md)        | Data formats and pipeline      |
| [Training Instructions](docs/training_instructions.md)| How to train the model         |
| [Inference Server](docs/inference_server.md)          | API server documentation       |
| [Update Model](docs/update_model.md)                  | Self-learning and updates      |
| [Integration Guide](docs/integration_guide.md)        | How to integrate Abu Hashim    |
| [Branding Guidelines](docs/branding_guidelines.md)    | QalamAI brand assets           |
| [QalamAI Bridge](docs/qalam_bridge.md)                | Data collection from QalamAI   |

## Branding

QalamAI uses a Dark Navy and Gold color scheme:

- **Primary Dark**: `#0A1A2F`
- **Primary Gold**: `#D4AF37`

## Requirements

- Python 3.10+
- CUDA-capable GPU (for training and inference)
- 16GB+ VRAM recommended for standard training
- 8GB+ VRAM minimum with QLoRA quantization

## License

This project uses open-source language models. Please refer to each base model's license for usage terms.
