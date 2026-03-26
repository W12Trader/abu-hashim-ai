# QalamAI — Abu Hashim AI Model

<div align="center">

**قلم الذكاء الاصطناعي**

An advanced Arabic AI assistant built on open-source language models.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](#)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)

</div>

<div align="center">

<!-- Programming Language -->
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

<!-- Frameworks -->
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Uvicorn](https://img.shields.io/badge/Server-Uvicorn-4B8BBE.svg)](https://www.uvicorn.org)
[![Pydantic](https://img.shields.io/badge/Schema-Pydantic-0A66C2.svg)](https://docs.pydantic.dev)

<!-- Machine Learning -->
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/docs/transformers)
[![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-FF6F00.svg)](https://huggingface.co/docs/datasets)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-FF9900.svg)](https://huggingface.co/docs/peft)
[![BitsAndBytes](https://img.shields.io/badge/Quantization-BitsAndBytes-00A67E.svg)](https://github.com/TimDettmers/bitsandbytes)
[![Accelerate](https://img.shields.io/badge/HuggingFace-Accelerate-0099CC.svg)](https://huggingface.co/docs/accelerate)

<!-- Base Models -->
[![Aya-23](https://img.shields.io/badge/Model-Aya--23-8B%2F35B-8A2BE2.svg)](https://huggingface.co/CohereForAI)
[![JAIS-13B](https://img.shields.io/badge/Model-JAIS--13B-1E90FF.svg)](https://huggingface.co/inception-mbzuai)
[![Qwen2-7B](https://img.shields.io/badge/Model-Qwen2--7B-FF4500.svg)](https://huggingface.co/Qwen)

<!-- Evaluation -->
[![Evaluate](https://img.shields.io/badge/HuggingFace-Evaluate-5A4FCF.svg)](https://huggingface.co/docs/evaluate)
[![CAMeL Tools](https://img.shields.io/badge/ArabicNLP-CAMeL--Tools-006400.svg)](https://camelcameltools.readthedocs.io)

<!-- Safety -->
[![Safety](https://img.shields.io/badge/Safety-Content%20Filters-red.svg)](https://platform.openai.com/docs/guides/moderation)

<!-- QalamAI -->
[![QalamAI](https://img.shields.io/badge/QalamAI-Integration-DAA520.svg)](https://qalamai.net)

<!-- Environment -->
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NVIDIA Drivers](https://img.shields.io/badge/NVIDIA-Drivers-76B900.svg)](https://www.nvidia.com/download/index.aspx)

<!-- License -->
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](#)

</div>


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
