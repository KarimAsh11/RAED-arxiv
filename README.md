<div align="center">

 
<img src="raed-logo.png" alt="Logo" height="200" align="center"/>

# RAED: Retrieval-Augmented Entity Description Generation for Emerging Entity Linking and Disambiguation

<img src="https://github.com/Babelscape/FENICE/blob/master/Sapienza_Babelscape.png?raw=True" height="70">


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5?logo=lightning)](https://lightning.ai/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A PyTorch Lightning framework for generating entity descriptions using retrieval-augmented generation**

[Installation](#-installation) •
[Quick Start](#-usage) •
[Models](#-models) •
[Documentation](#-configuration)


</div>

---

## 📖 Overview

RAED combines language models (T5, SmolLM2, Llama-3.2) with a retrieval component to generate accurate entity descriptions. The system retrieves relevant context passages from Wikipedia and uses them to augment the generation process, improving performance on the entity description generation task.

### ✨ Key Features

- 🤖 **Multiple Model Support**: T5, FiD (Fusion-in-Decoder), SmolLM2, and Llama-3.2
- 🔎 **Retrieval-Augmented Generation**: Integrates retrieved contexts to improve entity description quality
- 🎯 **Entity Disambiguation and Emerging Entity Linking Evaluation**: Tested on AIDA and Tempel datasets
- ⚙️ **Flexible Training Modes**: Support for both encoder-decoder and decoder-only models
- ⚡ **PyTorch Lightning**: Easily extendible pipeline

## 🚀 Installation

### 📋 Prerequisites

- Python 3.10+
- CUDA 11.7+ (for GPU support)
- Conda (recommended)

### ⚡ Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RAED
```

2. Run the setup script:
```bash
bash scripts/setup.sh
```

This will:
- Create a conda environment
- Install PyTorch with CUDA support
- Install all required dependencies

### 🛠️ Manual Installation

```bash
conda create -n raed python=3.10
conda activate raed
conda install pytorch torchvision cudatoolkit=11.7 -c pytorch
pip install -r requirements.txt
```

## 📁 Project Structure

```
RAED/
├── conf/                       # Hydra configuration files
│   ├── data/                   # Dataset configurations
│   ├── model/                  # Model configurations
│   ├── train/                  # Training configurations
│   └── logging/                # Logging configurations
├── src/
│   ├── data/                   # Dataset classes and utilities
│   ├── models/                 # Model implementations
│   ├── callbacks/              # Training callbacks
│   ├── trainer/                # Training, testing, and prediction scripts
│   └── Retriever/              # Retrieval system components
├── scripts/                    # Utility scripts
└── requirements.txt
```

## 🎮 Usage

### 🏋️ Training

Train a model using the default configuration:

```bash
bash scripts/train.sh
```

Or with custom configuration:

```bash
PYTHONPATH='.' python src/trainer/train.py \
    model=emerge_T5 \
    data=Aida_RAG \
    logging.wandb_arg.name=my_experiment
```

### 🧪 Testing

Evaluate a trained model:

```bash
bash scripts/test.sh
```

Or specify a checkpoint:

```bash
PYTHONPATH='.' python src/trainer/test.py \
    train.best_rag_ckpt_path=path/to/checkpoint.ckpt
```

### 🔮 Prediction

Generate predictions on a dataset:

```bash
bash scripts/predict.sh
```

## ⚙️ Configuration

RAED uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in the `conf/` directory.

### 📝 Key Configuration Files

- `conf/raed.yaml`: Main configuration file
- `conf/model/emerge_T5.yaml`: T5 model configuration
- `conf/model/emerge_smollm2.yaml`: SmolLM2 configuration
- `conf/data/Aida_RAG.yaml`: AIDA dataset with retrieval
- `conf/train/rag_trainer.yaml`: Training hyperparameters

### Configuration Options

#### Model Selection

```yaml
model:
  model_name: 't5-large'  # or 'HuggingFaceTB/SmolLM2-360M'
  fid: False              # Enable Fusion-in-Decoder
```

#### Data Configuration

```yaml
data:
  batch_size: 8
  train_extra_contexts: 10  # Number of retrieved contexts
  test_extra_contexts: 10
  target: 'title_def'       # 'title', 'definition', or 'title_def'
```

#### Training Parameters

```yaml
train:
  seed: 42
  lr_scheduler:
    lr: 2e-05
    num_warmup_steps: 2000
  generation_params:
    num_beams: 3
    max_new_tokens: 200
```

## 📊 Data Preparation

### 🔍 Retrieval Index Creation

1. Create windows from Wikipedia pages:
```bash
python src/Retriever/windowization/create_windows.py \
    <index_file> \
    <wiki_pages> \
    <output_file>
```

2. Filter and rank contexts by similarity:
```bash
python src/Retriever/windowization/filter_cosine.py
```

3. Build the retrieval index:
```bash
python src/Retriever/retriever/create_index.py \
    --question-encoder-name-or-path <encoder> \
    --document-path <documents.jsonl> \
    --output-folder <output_dir>
```

4. Retrieve contexts for your dataset:
```bash
bash scripts/retrieve_contexts.sh
```

### 📄 Data Format

Input data should be in JSONL format with the following structure:

```json
{
  "id": "sample_id",
  "context": "Text with entity mention [DEF] entity [/DEF]",
  "wikipedia": "Entity_Title",
  "gold_definition_wikipedia": "Entity description",
  "candidates_WIKIPEDIA": [
    {"title": "Candidate_1", "text": "Description 1"},
    {"title": "Candidate_2", "text": "Description 2"}
  ],
  "candidates_RETRIEVER": [
    {"text": "Retrieved context 1"},
    {"text": "Retrieved context 2"}
  ]
}
```

## 📈 Evaluation

The system supports multiple evaluation modes:

1. **Standard Generation**: Generate entity descriptions
2. **Perplexity-based Ranking**: Rank candidates by perplexity
3. **Constrained Generation**: Generate with constrained vocabulary

Results are logged to:
- WandB (if configured)
- Local files (JSONL format)
- Console output

### 📊 Metrics

- NLG metrics (BLEU, Rouge, Semantic Similarity, BERTScore)
- Factuality metric (Factual-NLI)
- inKB F1-score (Entity Disambiguation)
- Accuracy@64 (Emerging Entity Linking)

## 🤖 Models

### 🏗️ Supported Architectures

1. **T5**: Text-to-Text Transfer Transformer
   - `t5-base`, `t5-large`
   - `google/flan-t5-large`

2. **FiD**: Fusion-in-Decoder
   - T5-based architecture for multi-document retrieval

3. **SmolLM2**: Small language model for efficient generation
   - `HuggingFaceTB/SmolLM2-360M`

4. **Llama-3.2**: 
   - `meta-llama/Llama-3.2-1B`

## 🔔 Callbacks

RAED includes several custom callbacks for evaluation:

- `EvalCallback`: Standard BLEU evaluation
- `PerplexCallback`: Perplexity-based candidate ranking
- `ConstrainedPerplexCallback`: Constrained generation
- `PredictCallback`: Save predictions to file

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{raed2024,
  title={RAED: Retrieval-Augmented Entity Descriptions},
  author={Your Name},
  year={2024}
}
```

## 📜 License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🔧 Troubleshooting

### ⚠️ CUDA Out of Memory

- Reduce `batch_size` in configuration
- Decrease `train_extra_contexts` and `test_extra_contexts`
- Use gradient accumulation: increase `accumulate_grad_batches`

### 🐌 Slow Training

- Increase `num_workers` for data loading
- Use mixed precision training: `precision: "16"`
- Reduce context window size

### ❌ Import Errors

Make sure to set `PYTHONPATH`:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## 🙏 Acknowledgments

- Built with PyTorch Lightning
- Uses Hugging Face Transformers
- Retrieval system based on Golden Retriever
- Inspired by FiD (Fusion-in-Decoder) architecture
